"""Central schema definitions for benchmark logging artifacts."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

METRICS_COLUMNS = [
    "env_steps",
    "wall_clock_seconds",
    "episodes_seen",
    "eval_return_mean",
    "eval_return_std",
    "eval_return_median",
    "train_episode_return_mean",
    "train_episode_length_mean",
    "mean_kl_old_new",
    "max_kl_old_new",
    "policy_ratio_mean",
    "policy_ratio_std",
    "unstable_update_flag",
    "cumulative_unstable_updates",
    "collapse_flag",
    "nan_or_divergence_flag",
    "policy_loss",
    "value_loss",
    "entropy",
    "advantage_mean",
    "advantage_std",
    "value_target_mean",
    "grad_norm_actor",
    "grad_norm_critic",
    "clip_fraction",
    "epochs_completed_before_early_stop",
    "cg_iterations_used",
    "line_search_backtracks",
    "accepted_step_fraction",
]

UPDATES_COLUMNS = [
    "update_index",
    "start_env_steps",
    "end_env_steps",
    "wall_clock_seconds",
    "mean_kl_old_new",
    "max_kl_old_new",
    "policy_ratio_mean",
    "policy_ratio_std",
    "large_step_no_drop",
    "drop_without_large_step",
    "unstable_update_flag",
    "epochs_completed",
    "early_stopped",
    "policy_loss",
    "value_loss",
    "entropy",
]

COLLAPSE_FIELDS = [
    "collapse_flag",
    "collapse_step",
    "collapse_reason",
    "R_init",
    "R_best",
    "collapse_threshold",
]

RUN_CONFIG_FIELDS = [
    "run_id",
    "algorithm",
    "env_id",
    "seed",
    "total_timesteps",
    "n_envs",
    "steps_per_env",
    "rollout_batch_size",
    "gamma",
    "gae_lambda",
    "obs_norm_enabled",
    "obs_clip",
    "reward_norm_enabled",
    "adv_norm_enabled",
    "policy_arch",
    "value_arch",
    "activation",
    "init_scheme",
]


def _missing_keys(payload: dict[str, Any], required_keys: list[str]) -> list[str]:
    return [key for key in required_keys if key not in payload]


def validate_metrics_row(row: dict[str, Any]) -> None:
    missing = _missing_keys(row, METRICS_COLUMNS)
    if missing:
        raise ValueError(f"metrics row missing keys: {missing}")


def validate_updates_row(row: dict[str, Any]) -> None:
    missing = _missing_keys(row, UPDATES_COLUMNS)
    if missing:
        raise ValueError(f"updates row missing keys: {missing}")


def validate_collapse_payload(payload: dict[str, Any]) -> None:
    missing = _missing_keys(payload, COLLAPSE_FIELDS)
    if missing:
        raise ValueError(f"collapse payload missing keys: {missing}")


def validate_run_config(payload: dict[str, Any]) -> None:
    missing = _missing_keys(payload, RUN_CONFIG_FIELDS)
    if missing:
        raise ValueError(f"run config missing keys: {missing}")


def _write_header_if_missing(path: Path, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()


def append_metrics_row(path: Path, row: dict[str, Any]) -> None:
    validate_metrics_row(row)
    _write_header_if_missing(path, METRICS_COLUMNS)
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRICS_COLUMNS)
        writer.writerow(row)


def append_updates_row(path: Path, row: dict[str, Any]) -> None:
    validate_updates_row(row)
    _write_header_if_missing(path, UPDATES_COLUMNS)
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=UPDATES_COLUMNS)
        writer.writerow(row)
