"""Evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Any

import numpy as np

from src.envs.normalization import sync_obs_rms


@dataclass(frozen=True)
class EvaluationSchedule:
    checkpoint_every_env_steps: int = 10_000
    episodes: int = 10
    deterministic: bool = True
    evaluate_at_step_zero: bool = True

    def should_evaluate(self, env_steps: int) -> bool:
        if env_steps == 0:
            return self.evaluate_at_step_zero
        return env_steps % self.checkpoint_every_env_steps == 0


def evaluate_policy_checkpoint(model: Any, train_env: Any, eval_env: Any, config: dict[str, Any]) -> dict[str, float]:
    sync_obs_rms(train_env, eval_env)
    episode_returns: list[float] = []
    for episode_index in range(config["evaluation"]["episodes"]):
        obs = eval_env.reset()
        done = False
        truncated = False
        episode_return = 0.0
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=config["evaluation"]["deterministic"])
            if not np.all(np.isfinite(np.asarray(action))):
                raise RuntimeError(f"non-finite evaluation action at episode {episode_index}")
            obs, reward, done_flags, info = eval_env.step(action)
            episode_return += float(np.asarray(reward).reshape(-1)[0])
            done = bool(np.asarray(done_flags).reshape(-1)[0])
            truncated = bool(info[0].get("TimeLimit.truncated", False))
        episode_returns.append(episode_return)
    return {
        "eval_return_mean": float(np.mean(episode_returns)),
        "eval_return_std": float(np.std(episode_returns)),
        "eval_return_median": float(median(episode_returns)),
    }
