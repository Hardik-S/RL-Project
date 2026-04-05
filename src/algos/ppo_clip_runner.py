"""PPO-Clip training runner."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from stable_baselines3 import PPO

from src.algos._sb3_helpers import logger_value, rollout_diagnostics
from src.algos.common import TrainingJob, run_training_job
from src.policies.actor_critic import BenchmarkActorCriticPolicy, build_policy_kwargs, set_optimizer_lrs


class InstrumentedPPOClip(PPO):
    def __init__(self, *args: Any, actor_lr: float, critic_lr: float, **kwargs: Any) -> None:
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.last_training_info: dict[str, Any] = {}
        super().__init__(*args, **kwargs)

    def train(self) -> None:
        set_optimizer_lrs(self.policy, self.actor_lr, self.critic_lr)
        super().train()
        diagnostics = rollout_diagnostics(self, batch_size=self.batch_size)
        diagnostics.update(
            {
                "update_index": int(self._n_updates),
                "policy_loss": logger_value(self, "train/policy_gradient_loss"),
                "value_loss": logger_value(self, "train/value_loss"),
                "entropy": -logger_value(self, "train/entropy_loss"),
                "clip_fraction": logger_value(self, "train/clip_fraction"),
                "epochs_completed_before_early_stop": math.nan,
                "cg_iterations_used": math.nan,
                "line_search_backtracks": math.nan,
                "accepted_step_fraction": math.nan,
                "epochs_completed": self.n_epochs,
                "early_stopped": False,
                "line_search_failed": False,
            }
        )
        self.last_training_info = diagnostics


def build_model(job: TrainingJob, train_env: Any, device: str) -> InstrumentedPPOClip:
    config = job.config
    return InstrumentedPPOClip(
        policy=BenchmarkActorCriticPolicy,
        env=train_env,
        learning_rate=config["algo"]["actor_lr"],
        n_steps=config["environment"]["steps_per_env"],
        batch_size=config["algo"]["minibatch_size"],
        n_epochs=config["algo"]["policy_epochs"],
        gamma=config["optimization"]["gamma"],
        gae_lambda=config["optimization"]["gae_lambda"],
        normalize_advantage=config["optimization"]["advantage_normalization"],
        clip_range=config["algo"]["clip_epsilon"],
        target_kl=None,
        seed=job.seed,
        device=device,
        verbose=0,
        tensorboard_log=str(job.output_dir / "tensorboard"),
        policy_kwargs=build_policy_kwargs(config),
        actor_lr=config["algo"]["actor_lr"],
        critic_lr=config["algo"]["critic_lr"],
    )


def load_model(path: Path, env: Any, device: str) -> InstrumentedPPOClip:
    return InstrumentedPPOClip.load(path, env=env, device=device)


def run(job: TrainingJob) -> None:
    run_training_job(job, build_model=build_model)
