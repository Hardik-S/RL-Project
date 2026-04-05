"""Exact unstable-update definition from the protocol."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StabilityInputs:
    algorithm: str
    mean_kl_old_new: float
    nominal_kl_budget: float
    next_eval_return: float
    trailing_eval_mean_prev3: float
    max_eval_so_far: float
    initial_eval_mean_first3: float


@dataclass(frozen=True)
class StabilityDecision:
    unstable_update: bool
    large_step_no_drop: bool
    drop_without_large_step: bool
    kl_condition: bool
    performance_condition: bool


def kl_threshold_for_algorithm(algorithm: str, nominal_kl_budget: float) -> float:
    if algorithm in {"ppo_kl", "trpo"}:
        return 2.0 * nominal_kl_budget
    return 0.05


def evaluate_unstable_update(inputs: StabilityInputs) -> StabilityDecision:
    kl_condition = inputs.mean_kl_old_new > kl_threshold_for_algorithm(
        inputs.algorithm,
        inputs.nominal_kl_budget,
    )
    range_so_far = inputs.max_eval_so_far - inputs.initial_eval_mean_first3
    performance_condition = inputs.next_eval_return < (
        inputs.trailing_eval_mean_prev3 - 0.20 * range_so_far
    )
    return StabilityDecision(
        unstable_update=kl_condition and performance_condition,
        large_step_no_drop=kl_condition and not performance_condition,
        drop_without_large_step=(not kl_condition) and performance_condition,
        kl_condition=kl_condition,
        performance_condition=performance_condition,
    )
