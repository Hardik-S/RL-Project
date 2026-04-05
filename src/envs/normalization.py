"""Observation-normalization helpers."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.utils.serialization import load_json, save_json


@dataclass(frozen=True)
class ObservationNormalizationConfig:
    enabled: bool = True
    clip_obs: float = 10.0
    normalize_reward: bool = False
    epsilon: float = 1e-8


@dataclass
class FrozenObservationStats:
    mean: list[float]
    var: list[float]
    count: float
    clip_obs: float
    epsilon: float


def build_vecnormalize_kwargs(config: ObservationNormalizationConfig) -> dict[str, Any]:
    return {
        "norm_obs": config.enabled,
        "norm_reward": config.normalize_reward,
        "clip_obs": config.clip_obs,
        "epsilon": config.epsilon,
        "training": True,
    }


def freeze_stats(mean: list[float], var: list[float], count: float, config: ObservationNormalizationConfig) -> FrozenObservationStats:
    return FrozenObservationStats(
        mean=mean,
        var=var,
        count=count,
        clip_obs=config.clip_obs,
        epsilon=config.epsilon,
    )


def save_frozen_stats(path: Path, stats: FrozenObservationStats) -> None:
    save_json(path, asdict(stats))


def load_frozen_stats(path: Path) -> FrozenObservationStats:
    payload = load_json(path)
    return FrozenObservationStats(**payload)


def sync_obs_rms(source_env: Any, target_env: Any) -> None:
    if hasattr(source_env, "obs_rms") and hasattr(target_env, "obs_rms"):
        target_env.obs_rms = copy.deepcopy(source_env.obs_rms)
    target_env.training = False
    target_env.norm_reward = False
