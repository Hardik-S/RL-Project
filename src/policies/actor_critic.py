"""Shared actor-critic policy helpers."""

from __future__ import annotations

import math
from typing import Any, Iterable

import torch as th
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy


class BenchmarkActorCriticPolicy(ActorCriticPolicy):
    """Matched actor-critic policy with separate optimizer parameter groups."""

    def __init__(self, *args: Any, actor_lr: float = 3e-4, critic_lr: float = 1e-3, **kwargs: Any) -> None:
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        kwargs.setdefault("net_arch", {"pi": [64, 64], "vf": [64, 64]})
        kwargs.setdefault("activation_fn", nn.Tanh)
        kwargs.setdefault("ortho_init", True)
        kwargs.setdefault("log_std_init", -0.5)
        kwargs.setdefault("share_features_extractor", False)
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule: Any) -> None:
        super()._build(lr_schedule)
        actor_params, critic_params = actor_critic_parameter_groups(self)
        self.optimizer = th.optim.Adam(
            [
                {"params": list(actor_params), "lr": self.actor_lr},
                {"params": list(critic_params), "lr": self.critic_lr},
            ]
        )


def build_policy_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "activation_fn": nn.Tanh,
        "net_arch": {
            "pi": list(config["architecture"]["actor"]["hidden_sizes"]),
            "vf": list(config["architecture"]["critic"]["hidden_sizes"]),
        },
        "ortho_init": True,
        "log_std_init": config["architecture"]["actor"]["log_std_init"],
        "share_features_extractor": False,
        "optimizer_class": th.optim.Adam,
        "actor_lr": config["algo"].get("actor_lr", 3e-4),
        "critic_lr": config["algo"].get("critic_lr", config["algo"].get("actor_lr", 3e-4)),
    }


def actor_critic_parameter_groups(policy: Any) -> tuple[list[Any], list[Any]]:
    actor_params: list[Any] = []
    critic_params: list[Any] = []

    if hasattr(policy, "pi_features_extractor"):
        actor_params.extend(policy.pi_features_extractor.parameters())
    if hasattr(policy, "vf_features_extractor"):
        critic_params.extend(policy.vf_features_extractor.parameters())
    actor_params.extend(policy.mlp_extractor.policy_net.parameters())
    critic_params.extend(policy.mlp_extractor.value_net.parameters())
    actor_params.extend(policy.action_net.parameters())
    critic_params.extend(policy.value_net.parameters())
    if getattr(policy, "log_std", None) is not None:
        actor_params.append(policy.log_std)
    return actor_params, critic_params


def set_optimizer_lrs(policy: Any, actor_lr: float, critic_lr: float) -> None:
    if len(policy.optimizer.param_groups) == 1:
        policy.optimizer.param_groups[0]["lr"] = actor_lr
        return
    policy.optimizer.param_groups[0]["lr"] = actor_lr
    policy.optimizer.param_groups[1]["lr"] = critic_lr


def grad_norm(parameters: Iterable[Any]) -> float:
    total = 0.0
    found = False
    for parameter in parameters:
        if getattr(parameter, "grad", None) is None:
            continue
        total += float(parameter.grad.detach().norm(2).item() ** 2)
        found = True
    if not found:
        return math.nan
    return math.sqrt(total)
