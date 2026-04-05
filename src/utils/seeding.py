"""Seed and RNG state helpers."""

from __future__ import annotations

import random
from typing import Any

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - optional during static scaffold work
    torch = None  # type: ignore[assignment]


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
    }
    if torch is not None:
        state["torch_random_state"] = torch.get_rng_state()
        if torch.cuda.is_available():
            state["torch_cuda_random_state"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python_random_state"])
    np.random.set_state(state["numpy_random_state"])
    if torch is not None and "torch_random_state" in state:
        torch.set_rng_state(state["torch_random_state"])
        if torch.cuda.is_available() and "torch_cuda_random_state" in state:
            torch.cuda.set_rng_state_all(state["torch_cuda_random_state"])
