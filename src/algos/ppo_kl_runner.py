"""PPO-KL training runner."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.utils import explained_variance
from torch.nn import functional as F

from src.algos._sb3_helpers import rollout_diagnostics
from src.algos.common import TrainingJob, run_training_job
from src.policies.actor_critic import BenchmarkActorCriticPolicy, build_policy_kwargs, set_optimizer_lrs

NON_OPERATIVE_CLIP_RANGE = 1e9


class InstrumentedPPOKL(PPO):
    def __init__(self, *args: Any, actor_lr: float, critic_lr: float, **kwargs: Any) -> None:
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.last_training_info: dict[str, Any] = {}
        super().__init__(*args, **kwargs)

    def train(self) -> None:
        self.policy.set_training_mode(True)
        try:
            set_optimizer_lrs(self.policy, self.actor_lr, self.critic_lr)

            entropy_losses: list[float] = []
            pg_losses: list[float] = []
            value_losses: list[float] = []
            epochs_fully_completed = 0
            early_stopped = False

            for epoch in range(self.n_epochs):
                epoch_completed = True
                for rollout_data in self.rollout_buffer.get(self.batch_size):
                    actions = rollout_data.actions
                    if self.action_space.__class__.__name__ == "Discrete":
                        actions = rollout_data.actions.long().flatten()

                    values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                    values = values.flatten()
                    advantages = rollout_data.advantages
                    if self.normalize_advantage and len(advantages) > 1:
                        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                    ratio = th.exp(log_prob - rollout_data.old_log_prob)
                    policy_loss = -(advantages * ratio).mean()

                    if self.clip_range_vf is None:
                        values_pred = values
                    else:
                        values_pred = rollout_data.old_values + th.clamp(
                            values - rollout_data.old_values,
                            -self.clip_range_vf,
                            self.clip_range_vf,
                        )

                    value_loss = F.mse_loss(rollout_data.returns, values_pred)
                    entropy_loss = -th.mean(entropy) if entropy is not None else th.mean(-log_prob)
                    loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                    with th.no_grad():
                        log_ratio = log_prob - rollout_data.old_log_prob
                        approx_kl = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()

                    if self.target_kl is not None and approx_kl > self.target_kl:
                        early_stopped = True
                        epoch_completed = False
                        break

                    self.policy.optimizer.zero_grad()
                    loss.backward()
                    th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.policy.optimizer.step()

                    pg_losses.append(float(policy_loss.item()))
                    value_losses.append(float(value_loss.item()))
                    entropy_losses.append(float(entropy_loss.item()))

                if epoch_completed:
                    epochs_fully_completed += 1
                if early_stopped:
                    break

            self._n_updates += 1
            explained_var = explained_variance(
                self.rollout_buffer.values.flatten(),
                self.rollout_buffer.returns.flatten(),
            )
            diagnostics = rollout_diagnostics(self, batch_size=self.batch_size)
            diagnostics.update(
                {
                    "update_index": int(self._n_updates),
                    "policy_loss": float(sum(pg_losses) / len(pg_losses)) if pg_losses else math.nan,
                    "value_loss": float(sum(value_losses) / len(value_losses)) if value_losses else math.nan,
                    "entropy": float(-sum(entropy_losses) / len(entropy_losses)) if entropy_losses else math.nan,
                    "clip_fraction": math.nan,
                    "epochs_completed_before_early_stop": epochs_fully_completed,
                    "cg_iterations_used": math.nan,
                    "line_search_backtracks": math.nan,
                    "accepted_step_fraction": math.nan,
                    "epochs_completed": epochs_fully_completed,
                    "early_stopped": early_stopped,
                    "line_search_failed": False,
                    "explained_variance": float(explained_var),
                }
            )
            self.last_training_info = diagnostics
        finally:
            self.policy.set_training_mode(False)


def build_model(job: TrainingJob, train_env: Any, device: str) -> InstrumentedPPOKL:
    config = job.config
    if not bool(config["algo"].get("early_stop_on_target_kl", False)):
        raise AssertionError("PPO-KL requires early_stop_on_target_kl=true")
    if bool(config["algo"].get("ratio_clipping", True)):
        raise AssertionError("PPO-KL requires ratio_clipping=false")
    return InstrumentedPPOKL(
        policy=BenchmarkActorCriticPolicy,
        env=train_env,
        learning_rate=config["algo"]["actor_lr"],
        n_steps=config["environment"]["steps_per_env"],
        batch_size=config["algo"]["minibatch_size"],
        n_epochs=config["algo"]["max_policy_epochs"],
        gamma=config["optimization"]["gamma"],
        gae_lambda=config["optimization"]["gae_lambda"],
        normalize_advantage=config["optimization"]["advantage_normalization"],
        # SB3 requires a clip-range argument, but PPO-KL uses the unclipped ratio surrogate.
        clip_range=NON_OPERATIVE_CLIP_RANGE,
        target_kl=config["algo"]["target_kl"],
        seed=job.seed,
        device=device,
        verbose=0,
        tensorboard_log=str(job.output_dir / "tensorboard"),
        policy_kwargs=build_policy_kwargs(config),
        actor_lr=config["algo"]["actor_lr"],
        critic_lr=config["algo"]["critic_lr"],
    )


def load_model(path: Path, env: Any, device: str) -> InstrumentedPPOKL:
    return InstrumentedPPOKL.load(path, env=env, device=device)


def run(job: TrainingJob) -> None:
    run_training_job(job, build_model=build_model)
