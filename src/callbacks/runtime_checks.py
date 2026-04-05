"""Runtime checks and rollout-level diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class RolloutStatsCallback(BaseCallback):
    """Collect episode stats and stop on invalid numeric behavior."""

    def __init__(self, action_space: Any) -> None:
        super().__init__(verbose=0)
        self.action_space = action_space
        self.episode_returns: list[float] = []
        self.episode_lengths: list[float] = []
        self.invalid_action_detected = False
        self.invalid_action_reason: str | None = None
        self.invalid_observation_detected = False
        self.invalid_observation_reason: str | None = None

    def _on_step(self) -> bool:
        actions = self.locals.get("actions")
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards")
        new_obs = self.locals.get("new_obs")

        if actions is not None:
            action_array = np.asarray(actions)
            if not np.all(np.isfinite(action_array)):
                self.invalid_action_detected = True
                self.invalid_action_reason = "non_finite_action"
                return False

        if rewards is not None and not np.all(np.isfinite(np.asarray(rewards))):
            self.invalid_observation_detected = True
            self.invalid_observation_reason = "non_finite_reward"
            return False

        if new_obs is not None and not np.all(np.isfinite(np.asarray(new_obs))):
            self.invalid_observation_detected = True
            self.invalid_observation_reason = "non_finite_observation"
            return False

        for info in infos:
            episode = info.get("episode")
            if episode is not None:
                self.episode_returns.append(float(episode["r"]))
                self.episode_lengths.append(float(episode["l"]))
        return True
