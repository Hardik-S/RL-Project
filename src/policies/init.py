"""Initialization policy metadata."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OrthogonalInitConfig:
    actor_hidden_gain: float = 2 ** 0.5
    actor_output_gain: float = 0.01
    critic_output_gain: float = 1.0
    log_std_init: float = -0.5


def init_notes() -> list[str]:
    return [
        "Use orthogonal initialization for all linear layers.",
        "Use actor hidden gain sqrt(2).",
        "Use actor output gain 0.01.",
        "Use critic output gain 1.0.",
        "Initialize learned state-independent log std to -0.5.",
    ]
