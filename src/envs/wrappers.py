"""Environment wrapper helpers shared across training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class WrapperConfig:
    """Frozen wrapper choices from the experiment protocol."""

    bootstrap_on_time_limit_truncation: bool = True
    action_squash: str = "tanh"


def is_time_limit_truncation(info: dict[str, Any] | None) -> bool:
    if not info:
        return False
    return bool(info.get("TimeLimit.truncated", False))


def apply_common_wrapper_notes() -> list[str]:
    """Return the wrapper invariants that later implementation must respect."""

    return [
        "Use tanh-squashed actions consistently across algorithms.",
        "Bootstrap value targets on time-limit truncation only.",
        "Do not bootstrap on true terminal transitions.",
        "Keep reward scaling, clipping, and normalization disabled.",
    ]
