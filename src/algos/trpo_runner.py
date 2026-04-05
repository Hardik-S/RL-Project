"""TRPO training runner."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from sb3_contrib import TRPO

from src.algos._sb3_helpers import logger_value, rollout_diagnostics
from src.algos.common import TrainingJob, run_training_job
from src.policies.actor_critic import BenchmarkActorCriticPolicy, build_policy_kwargs


def _set_trpo_critic_lr(policy: Any, critic_lr: float) -> None:
    if not hasattr(policy, "optimizer") or not getattr(policy.optimizer, "param_groups", None):
        return
    if len(policy.optimizer.param_groups) == 1:
        policy.optimizer.param_groups[0]["lr"] = critic_lr
        return
    policy.optimizer.param_groups[1]["lr"] = critic_lr


class InstrumentedTRPO(TRPO):
    def __init__(self, *args: Any, critic_lr: float, **kwargs: Any) -> None:
        self.critic_lr = critic_lr
        self.last_training_info: dict[str, Any] = {}
        super().__init__(*args, **kwargs)

    def train(self) -> None:
        # TRPO only uses optimizer-based updates for the critic; keep actor-group LR untouched.
        _set_trpo_critic_lr(self.policy, self.critic_lr)
        super().train()
        diagnostics = rollout_diagnostics(self, batch_size=getattr(self, "batch_size", None))
        entropy_metric = -logger_value(self, "train/entropy_loss")
        if not math.isfinite(entropy_metric):
            fallback_entropy = diagnostics.get("policy_entropy_mean", math.nan)
            entropy_metric = fallback_entropy if math.isfinite(fallback_entropy) else None
        accepted_step_fraction = logger_value(
            self,
            "train/accepted_step_fraction",
            "train/line_search_accepted_step_fraction",
        )
        diagnostics.update(
            {
                "update_index": int(self._n_updates),
                "policy_loss": logger_value(self, "train/policy_objective", "train/policy_loss"),
                "value_loss": logger_value(self, "train/value_loss"),
                "entropy": entropy_metric,
                "clip_fraction": math.nan,
                "epochs_completed_before_early_stop": math.nan,
                "cg_iterations_used": logger_value(
                    self,
                    "train/cg_iterations_used",
                    "train/cg_iterations",
                    "train/cg_steps",
                ),
                "line_search_backtracks": logger_value(
                    self,
                    "train/line_search_backtracks",
                    "train/line_search_steps",
                ),
                "accepted_step_fraction": accepted_step_fraction,
                "epochs_completed": 1,
                "early_stopped": False,
                "line_search_failed": accepted_step_fraction == 0.0,
            }
        )
        self.last_training_info = diagnostics


def build_model(job: TrainingJob, train_env: Any, device: str) -> InstrumentedTRPO:
    config = job.config
    return InstrumentedTRPO(
        policy=BenchmarkActorCriticPolicy,
        env=train_env,
        learning_rate=config["algo"]["critic_lr"],
        n_steps=config["environment"]["steps_per_env"],
        batch_size=config["algo"]["critic_minibatch_size"],
        n_critic_updates=config["algo"]["critic_epochs"],
        cg_max_steps=config["algo"]["cg_steps"],
        target_kl=config["algo"]["max_kl_delta"],
        cg_damping=config["algo"]["damping"],
        line_search_max_iter=config["algo"]["line_search_steps"],
        line_search_shrinking_factor=config["algo"]["backtrack_coefficient"],
        gamma=config["optimization"]["gamma"],
        gae_lambda=config["optimization"]["gae_lambda"],
        normalize_advantage=config["optimization"]["advantage_normalization"],
        seed=job.seed,
        device=device,
        verbose=0,
        tensorboard_log=str(job.output_dir / "tensorboard"),
        policy_kwargs=build_policy_kwargs(config),
        critic_lr=config["algo"]["critic_lr"],
    )


def load_model(path: Path, env: Any, device: str) -> InstrumentedTRPO:
    return InstrumentedTRPO.load(path, env=env, device=device)


def run(job: TrainingJob) -> None:
    run_training_job(job, build_model=build_model)
