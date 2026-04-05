"""SB3-specific helper functions for instrumented algorithms."""

from __future__ import annotations

import math
from typing import Any

import torch as th

from src.policies.actor_critic import actor_critic_parameter_groups, grad_norm


def rollout_diagnostics(model: Any, batch_size: int | None = None) -> dict[str, float]:
    ratios: list[th.Tensor] = []
    approx_kls: list[th.Tensor] = []
    advantages: list[th.Tensor] = []
    returns: list[th.Tensor] = []
    entropies: list[th.Tensor] = []

    policy = model.policy
    for rollout_data in model.rollout_buffer.get(batch_size=batch_size):
        actions = rollout_data.actions
        if model.action_space.__class__.__name__ == "Discrete":
            actions = rollout_data.actions.long().flatten()
        with th.no_grad():
            values, log_prob, entropy = policy.evaluate_actions(rollout_data.observations, actions)
        del values
        log_ratio = log_prob - rollout_data.old_log_prob
        ratios.append(th.exp(log_ratio))
        approx_kls.append(((th.exp(log_ratio) - 1) - log_ratio))
        advantages.append(rollout_data.advantages)
        returns.append(rollout_data.returns)
        if entropy is not None:
            entropies.append(entropy.reshape(-1))

    if ratios:
        ratio_tensor = th.cat([ratio.reshape(-1) for ratio in ratios])
        kl_tensor = th.cat([value.reshape(-1) for value in approx_kls])
        adv_tensor = th.cat([value.reshape(-1) for value in advantages])
        return_tensor = th.cat([value.reshape(-1) for value in returns])
        entropy_mean = (
            float(th.cat(entropies).mean().item())
            if entropies
            else math.nan
        )
        mean_kl = float(kl_tensor.mean().item())
        max_kl = float(kl_tensor.max().item())
        policy_ratio_mean = float(ratio_tensor.mean().item())
        policy_ratio_std = float(ratio_tensor.std(unbiased=False).item())
        advantage_mean = float(adv_tensor.mean().item())
        advantage_std = float(adv_tensor.std(unbiased=False).item())
        value_target_mean = float(return_tensor.mean().item())
    else:
        mean_kl = math.nan
        max_kl = math.nan
        policy_ratio_mean = math.nan
        policy_ratio_std = math.nan
        advantage_mean = math.nan
        advantage_std = math.nan
        value_target_mean = math.nan
        entropy_mean = math.nan

    actor_params, critic_params = actor_critic_parameter_groups(policy)
    return {
        "mean_kl_old_new": mean_kl,
        "max_kl_old_new": max_kl,
        "policy_ratio_mean": policy_ratio_mean,
        "policy_ratio_std": policy_ratio_std,
        "advantage_mean": advantage_mean,
        "advantage_std": advantage_std,
        "value_target_mean": value_target_mean,
        "policy_entropy_mean": entropy_mean,
        "grad_norm_actor": grad_norm(actor_params),
        "grad_norm_critic": grad_norm(critic_params),
        "log_std_mean": float(policy.log_std.mean().item()) if getattr(policy, "log_std", None) is not None else math.nan,
    }


def logger_value(model: Any, *keys: str, default: float = math.nan) -> float:
    values = getattr(getattr(model, "logger", None), "name_to_value", {})
    for key in keys:
        if key in values:
            try:
                return float(values[key])
            except (TypeError, ValueError):
                return default
    return default
