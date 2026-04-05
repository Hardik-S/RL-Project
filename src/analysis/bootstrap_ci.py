"""Bootstrap and Wilson interval helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class Interval:
    mean: float
    lower: float
    upper: float


def bootstrap_mean_ci(values: Iterable[float], n_bootstrap: int = 2000, confidence: float = 0.95, seed: int = 0) -> Interval:
    values = np.asarray(list(values), dtype=float)
    if values.size == 0:
        raise ValueError("cannot bootstrap an empty sequence")
    rng = np.random.default_rng(seed)
    samples = np.array([
        rng.choice(values, size=values.size, replace=True).mean()
        for _ in range(n_bootstrap)
    ])
    alpha = (1.0 - confidence) / 2.0
    lower = float(np.quantile(samples, alpha))
    upper = float(np.quantile(samples, 1.0 - alpha))
    return Interval(mean=float(values.mean()), lower=lower, upper=upper)


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> Interval:
    if total <= 0:
        raise ValueError("total must be positive")
    phat = successes / total
    denom = 1 + z**2 / total
    center = (phat + z**2 / (2 * total)) / denom
    margin = z * math.sqrt((phat * (1 - phat) + z**2 / (4 * total)) / total) / denom
    return Interval(mean=phat, lower=center - margin, upper=center + margin)
