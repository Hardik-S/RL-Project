"""Exact collapse-event definition from the protocol."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CollapseTracker:
    initial_eval_mean: float
    best_eval: float
    eligible: bool = False
    below_threshold_streak: int = 0

    def update(self, eval_return: float) -> tuple[bool, float]:
        self.best_eval = max(self.best_eval, eval_return)
        eligibility_threshold = self.initial_eval_mean + 0.60 * (self.best_eval - self.initial_eval_mean)
        collapse_threshold = self.initial_eval_mean + 0.25 * (self.best_eval - self.initial_eval_mean)

        if eval_return >= eligibility_threshold:
            self.eligible = True

        if self.eligible and eval_return < collapse_threshold:
            self.below_threshold_streak += 1
        else:
            self.below_threshold_streak = 0

        return self.below_threshold_streak >= 5, collapse_threshold


def immediate_collapse_reason(
    *,
    policy_loss_nan: bool = False,
    value_loss_nan: bool = False,
    action_params_nan: bool = False,
    permanent_line_search_failure: bool = False,
    numerically_invalid_environment_interaction: bool = False,
) -> str | None:
    if policy_loss_nan:
        return "policy_loss_nan"
    if value_loss_nan:
        return "value_loss_nan"
    if action_params_nan:
        return "action_distribution_nan"
    if permanent_line_search_failure:
        return "line_search_failed"
    if numerically_invalid_environment_interaction:
        return "environment_numeric_failure"
    return None
