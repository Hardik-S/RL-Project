"""Shared training and evaluation harness for the benchmark."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import platform
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from src.metrics.collapse import CollapseTracker, immediate_collapse_reason
from src.metrics.logging_schema import (
    METRICS_COLUMNS,
    UPDATES_COLUMNS,
    validate_collapse_payload,
    validate_run_config,
)
from src.metrics.stability import StabilityInputs, evaluate_unstable_update
from src.utils.manifests import stable_config_hash, utc_now_iso
from src.utils.paths import repo_root, run_dir
from src.utils.seeding import capture_rng_state, set_global_seeds
from src.utils.serialization import load_json, save_json

CONFIG_ROOT = repo_root() / "configs"

ALGO_MODULES = {
    "a2c": "src.algos.a2c_runner",
    "ppo_clip": "src.algos.ppo_clip_runner",
    "ppo_kl": "src.algos.ppo_kl_runner",
    "trpo": "src.algos.trpo_runner",
}

ALGO_OVERRIDE_PATHS = {
    "actor_lr": ("algo", "actor_lr"),
    "critic_lr": ("algo", "critic_lr"),
    "clip_epsilon": ("algo", "clip_epsilon"),
    "target_kl": ("algo", "target_kl"),
    "policy_epochs": ("algo", "policy_epochs"),
    "max_policy_epochs": ("algo", "max_policy_epochs"),
    "critic_epochs": ("algo", "critic_epochs"),
    "minibatch_size": ("algo", "minibatch_size"),
    "critic_minibatch_size": ("algo", "critic_minibatch_size"),
    "max_kl_delta": ("algo", "max_kl_delta"),
    "cg_steps": ("algo", "cg_steps"),
    "damping": ("algo", "damping"),
    "line_search_steps": ("algo", "line_search_steps"),
    "backtrack_coefficient": ("algo", "backtrack_coefficient"),
    "total_timesteps": ("env", "total_timesteps"),
    "eval_every": ("evaluation", "checkpoint_every_env_steps"),
    "eval_episodes": ("evaluation", "episodes"),
    "n_envs": ("environment", "n_envs"),
    "steps_per_env": ("environment", "steps_per_env"),
}


@dataclass(frozen=True)
class TrainingJob:
    algorithm: str
    env_key: str
    seed: int
    config: dict[str, Any]
    output_dir: Path
    suite_name: str | None = None
    run_tag: str | None = None
    device: str = "auto"
    dry_run: bool = False
    resume: bool = False


def ensure_runtime_dependencies() -> None:
    missing: list[str] = []
    for module_name in ("yaml", "torch", "gymnasium", "stable_baselines3", "sb3_contrib"):
        try:
            importlib.import_module(module_name)
        except ImportError:
            missing.append(module_name)
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(
            f"Missing runtime dependencies: {joined}. Install them with "
            "'python -m pip install -r requirements.txt'."
        )


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dependency availability depends on local setup
        raise RuntimeError(
            "PyYAML is required to load benchmark configs. Install dependencies with "
            "'python -m pip install -r requirements.txt'."
        ) from exc
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _set_nested(config: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    cursor = config
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value


def apply_overrides(config: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    updated = json.loads(json.dumps(config))
    for override_key, value in overrides.items():
        if value is None or override_key not in ALGO_OVERRIDE_PATHS:
            continue
        _set_nested(updated, ALGO_OVERRIDE_PATHS[override_key], value)
    updated["environment"]["rollout_batch_size"] = (
        updated["environment"]["n_envs"] * updated["environment"]["steps_per_env"]
    )
    return updated


def load_run_config(
    algorithm: str,
    env_key: str,
    seed: int,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    base = _load_yaml(CONFIG_ROOT / "base.yaml")
    env_cfg = _load_yaml(CONFIG_ROOT / "envs" / f"{env_key}.yaml")
    algo_cfg = _load_yaml(CONFIG_ROOT / "algos" / f"{algorithm}.yaml")
    config = deep_merge(base, env_cfg)
    config = deep_merge(config, algo_cfg)
    config["run"] = {
        "algorithm": algorithm,
        "env_key": env_key,
        "seed": seed,
    }
    if overrides:
        config = apply_overrides(config, overrides)
    return config


def build_run_metadata(config: dict[str, Any]) -> dict[str, Any]:
    metadata = {
        "run_id": f"{config['run']['algorithm']}:{config['run']['env_key']}:{config['run']['seed']}",
        "algorithm": config["run"]["algorithm"],
        "env_id": config["env"]["env_id"],
        "seed": config["run"]["seed"],
        "total_timesteps": config["env"]["total_timesteps"],
        "n_envs": config["environment"]["n_envs"],
        "steps_per_env": config["environment"]["steps_per_env"],
        "rollout_batch_size": config["environment"]["rollout_batch_size"],
        "gamma": config["optimization"]["gamma"],
        "gae_lambda": config["optimization"]["gae_lambda"],
        "obs_norm_enabled": config["environment"]["observation_normalization"],
        "obs_clip": config["environment"]["observation_clip"],
        "reward_norm_enabled": config["environment"]["reward_normalization"],
        "adv_norm_enabled": config["optimization"]["advantage_normalization"],
        "policy_arch": config["architecture"]["actor"]["hidden_sizes"],
        "value_arch": config["architecture"]["critic"]["hidden_sizes"],
        "activation": config["architecture"]["actor"]["activation"],
        "init_scheme": config["architecture"]["initialization"]["scheme"],
    }
    validate_run_config(metadata)
    return metadata


def create_training_job(
    algorithm: str,
    env_key: str,
    seed: int,
    *,
    suite_name: str | None = None,
    run_tag: str | None = None,
    device: str = "auto",
    dry_run: bool = False,
    resume: bool = False,
    overrides: dict[str, Any] | None = None,
) -> TrainingJob:
    config = load_run_config(algorithm, env_key, seed, overrides=overrides)
    config["run"]["device"] = device
    config["run"]["suite_name"] = suite_name
    config["run"]["run_tag"] = run_tag or ("default" if suite_name else None)
    return TrainingJob(
        algorithm=algorithm,
        env_key=env_key,
        seed=seed,
        config=config,
        output_dir=run_dir(algorithm, env_key, seed, suite_name=suite_name, run_tag=run_tag),
        suite_name=suite_name,
        run_tag=run_tag,
        device=device,
        dry_run=dry_run,
        resume=resume,
    )


def common_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--algo", choices=sorted(ALGO_MODULES.keys()))
    parser.add_argument(
        "--env",
        dest="env_key",
        choices=["pendulum_v1", "hopper_v4", "walker2d_v4", "halfcheetah_v4"],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--total-timesteps", type=int)
    parser.add_argument("--eval-every", type=int)
    parser.add_argument("--eval-episodes", type=int)
    parser.add_argument("--n-envs", type=int)
    parser.add_argument("--steps-per-env", type=int)
    parser.add_argument("--actor-lr", type=float)
    parser.add_argument("--critic-lr", type=float)
    parser.add_argument("--clip-epsilon", type=float)
    parser.add_argument("--target-kl", type=float)
    parser.add_argument("--policy-epochs", type=int)
    parser.add_argument("--max-policy-epochs", type=int)
    parser.add_argument("--critic-epochs", type=int)
    parser.add_argument("--minibatch-size", type=int)
    parser.add_argument("--critic-minibatch-size", type=int)
    parser.add_argument("--max-kl-delta", type=float)
    parser.add_argument("--cg-steps", type=int)
    parser.add_argument("--damping", type=float)
    parser.add_argument("--line-search-steps", type=int)
    parser.add_argument("--backtrack-coefficient", type=float)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--suite-name")
    parser.add_argument("--run-tag")
    return parser


def overrides_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "total_timesteps": args.total_timesteps,
        "eval_every": args.eval_every,
        "eval_episodes": args.eval_episodes,
        "n_envs": args.n_envs,
        "steps_per_env": args.steps_per_env,
        "actor_lr": args.actor_lr,
        "critic_lr": args.critic_lr,
        "clip_epsilon": args.clip_epsilon,
        "target_kl": args.target_kl,
        "policy_epochs": args.policy_epochs,
        "max_policy_epochs": args.max_policy_epochs,
        "critic_epochs": args.critic_epochs,
        "minibatch_size": args.minibatch_size,
        "critic_minibatch_size": args.critic_minibatch_size,
        "max_kl_delta": args.max_kl_delta,
        "cg_steps": args.cg_steps,
        "damping": args.damping,
        "line_search_steps": args.line_search_steps,
        "backtrack_coefficient": args.backtrack_coefficient,
    }


def job_to_json(job: TrainingJob) -> str:
    payload = {
        "algorithm": job.algorithm,
        "env_key": job.env_key,
        "seed": job.seed,
        "output_dir": str(job.output_dir),
        "suite_name": job.suite_name,
        "run_tag": job.run_tag,
        "config": job.config,
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def get_algorithm_module(algorithm: str) -> Any:
    return importlib.import_module(ALGO_MODULES[algorithm])


def _safe_float(value: Any, default: float = math.nan) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric


def _all_finite(values: list[float]) -> bool:
    return all(math.isfinite(value) for value in values)


def _metric_is_non_finite(metrics: dict[str, Any], key: str) -> bool:
    value = metrics.get(key)
    if value is None:
        return False
    return not math.isfinite(_safe_float(value))


def _immediate_collapse_reason(update_metrics: dict[str, Any], callback: Any) -> str | None:
    if callback.invalid_action_detected:
        return callback.invalid_action_reason or "invalid_action"
    if callback.invalid_observation_detected:
        return callback.invalid_observation_reason or "invalid_observation"
    return immediate_collapse_reason(
        policy_loss_nan=_metric_is_non_finite(update_metrics, "policy_loss"),
        value_loss_nan=_metric_is_non_finite(update_metrics, "value_loss"),
        action_params_nan=_metric_is_non_finite(update_metrics, "log_std_mean"),
        permanent_line_search_failure=bool(update_metrics.get("line_search_failed", False)),
        numerically_invalid_environment_interaction=False,
    )


def _detect_numeric_issue(update_metrics: dict[str, Any], callback: Any) -> tuple[bool, str | None]:
    required_numeric_keys = ["policy_loss", "value_loss", "entropy", "log_std_mean"]
    values = []
    for key in required_numeric_keys:
        if update_metrics.get(key) is None:
            continue
        numeric = _safe_float(update_metrics.get(key), default=math.nan)
        if math.isnan(numeric):
            return True, f"non_finite_{key}"
        values.append(numeric)
    if not _all_finite(values):
        return True, "non_finite_training_metric"
    loss_threshold = 1e6
    if abs(_safe_float(update_metrics.get("policy_loss"))) > loss_threshold:
        return True, "exploding_policy_loss"
    if abs(_safe_float(update_metrics.get("value_loss"))) > loss_threshold:
        return True, "exploding_value_loss"
    if callback.invalid_action_detected:
        return True, callback.invalid_action_reason or "invalid_action"
    if callback.invalid_observation_detected:
        return True, callback.invalid_observation_reason or "invalid_observation"
    return False, None


def _blank_metrics_row() -> dict[str, Any]:
    row = {column: math.nan for column in METRICS_COLUMNS}
    row["unstable_update_flag"] = 0
    row["cumulative_unstable_updates"] = 0
    row["collapse_flag"] = 0
    row["nan_or_divergence_flag"] = 0
    return row


def _blank_update_row() -> dict[str, Any]:
    row = {column: math.nan for column in UPDATES_COLUMNS}
    row["large_step_no_drop"] = 0
    row["drop_without_large_step"] = 0
    row["unstable_update_flag"] = 0
    row["early_stopped"] = 0
    return row


def _write_updates_table(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    import csv

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=UPDATES_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _default_collapse_payload() -> dict[str, Any]:
    payload = {
        "collapse_flag": 0,
        "collapse_step": None,
        "collapse_reason": None,
        "R_init": None,
        "R_best": None,
        "collapse_threshold": None,
    }
    validate_collapse_payload(payload)
    return payload


def _save_collapse_payload(path: Path, payload: dict[str, Any]) -> None:
    validate_collapse_payload(payload)
    save_json(path, payload)


def _make_run_payload(job: TrainingJob) -> dict[str, Any]:
    payload = build_run_metadata(job.config)
    payload["config"] = job.config
    payload["config_hash"] = stable_config_hash(job.config)
    payload["python_version"] = platform.python_version()
    payload["platform"] = platform.platform()
    return payload


def _run_metadata(job: TrainingJob, run_payload: dict[str, Any]) -> dict[str, Any]:
    metadata = {
        "run_id": run_payload["run_id"],
        "config_hash": run_payload["config_hash"],
        "suite_name": job.suite_name,
        "run_tag": job.run_tag or ("default" if job.suite_name else None),
        "algorithm": job.algorithm,
        "env_key": job.env_key,
        "env_id": job.config["env"]["env_id"],
        "seed": job.seed,
        "device": job.device,
        "output_dir": str(job.output_dir),
        "created_at": utc_now_iso(),
    }
    metadata.update(_system_provenance())
    return metadata


def _cpu_description() -> str:
    processor = (platform.processor() or "").strip()
    if processor:
        return processor
    uname_processor = getattr(platform.uname(), "processor", "")
    if uname_processor:
        return uname_processor
    return platform.machine()


def _system_provenance() -> dict[str, Any]:
    provenance = {
        "hostname": platform.node() or os.environ.get("COMPUTERNAME") or os.environ.get("HOSTNAME"),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu": _cpu_description(),
    }
    try:
        import torch
    except Exception:  # pragma: no cover - defensive fallback if torch import fails at runtime
        provenance.update(
            {
                "torch_version": None,
                "cuda_available": False,
                "cuda_device_name": None,
            }
        )
        return provenance

    cuda_available = bool(torch.cuda.is_available())
    cuda_device_name = None
    if cuda_available:
        try:
            cuda_device_name = torch.cuda.get_device_name(torch.cuda.current_device())
        except Exception:  # pragma: no cover - depends on local CUDA setup
            cuda_device_name = None
    provenance.update(
        {
            "torch_version": torch.__version__,
            "cuda_available": cuda_available,
            "cuda_device_name": cuda_device_name,
        }
    )
    return provenance


def _write_run_status(path: Path, payload: dict[str, Any]) -> None:
    save_json(path, payload)


def _write_error_file(job: TrainingJob, error_message: str, traceback_text: str | None = None) -> None:
    payload = {
        "status": "failed",
        "failed_at": utc_now_iso(),
        "error_message": error_message,
    }
    if traceback_text:
        payload["traceback"] = traceback_text
    save_json(job.output_dir / "error.json", payload)
    _write_run_status(job.output_dir / "run_status.json", payload)


def _reset_run_directory(path: Path, *, preserve_names: set[str] | None = None) -> None:
    if not path.exists():
        return
    preserved = preserve_names or set()
    for child in path.iterdir():
        if child.name in preserved:
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _episodes_summary(episode_returns: list[float], episode_lengths: list[float]) -> tuple[float, float, int]:
    if not episode_returns:
        return math.nan, math.nan, 0
    return (
        float(sum(episode_returns) / len(episode_returns)),
        float(sum(episode_lengths) / len(episode_lengths)),
        len(episode_returns),
    )


def _checkpoint_metric_row(
    *,
    env_steps: int,
    wall_clock_seconds: float,
    episodes_seen: int,
    eval_summary: dict[str, float],
    train_episode_return_mean: float,
    train_episode_length_mean: float,
    update_metrics: dict[str, Any],
    cumulative_unstable_updates: int,
    collapse_flag: bool,
    nan_or_divergence_flag: bool,
) -> dict[str, Any]:
    row = _blank_metrics_row()
    row.update(
        {
            "env_steps": env_steps,
            "wall_clock_seconds": wall_clock_seconds,
            "episodes_seen": episodes_seen,
            "eval_return_mean": eval_summary["eval_return_mean"],
            "eval_return_std": eval_summary["eval_return_std"],
            "eval_return_median": eval_summary["eval_return_median"],
            "train_episode_return_mean": train_episode_return_mean,
            "train_episode_length_mean": train_episode_length_mean,
            "mean_kl_old_new": update_metrics["mean_kl_old_new"],
            "max_kl_old_new": update_metrics["max_kl_old_new"],
            "policy_ratio_mean": update_metrics["policy_ratio_mean"],
            "policy_ratio_std": update_metrics["policy_ratio_std"],
            "unstable_update_flag": int(update_metrics["unstable_update_flag"]),
            "cumulative_unstable_updates": cumulative_unstable_updates,
            "collapse_flag": int(collapse_flag),
            "nan_or_divergence_flag": int(nan_or_divergence_flag),
            "policy_loss": update_metrics["policy_loss"],
            "value_loss": update_metrics["value_loss"],
            "entropy": update_metrics["entropy"],
            "advantage_mean": update_metrics["advantage_mean"],
            "advantage_std": update_metrics["advantage_std"],
            "value_target_mean": update_metrics["value_target_mean"],
            "grad_norm_actor": update_metrics["grad_norm_actor"],
            "grad_norm_critic": update_metrics["grad_norm_critic"],
            "clip_fraction": update_metrics["clip_fraction"],
            "epochs_completed_before_early_stop": update_metrics["epochs_completed_before_early_stop"],
            "cg_iterations_used": update_metrics["cg_iterations_used"],
            "line_search_backtracks": update_metrics["line_search_backtracks"],
            "accepted_step_fraction": update_metrics["accepted_step_fraction"],
        }
    )
    return row


def _complete_update_with_eval(
    update_row: dict[str, Any],
    eval_history: list[float],
    cumulative_unstable_updates: int,
    algorithm: str,
    nominal_kl_budget: float,
) -> tuple[dict[str, Any], int]:
    if len(eval_history) < 4:
        return update_row, cumulative_unstable_updates
    decision = evaluate_unstable_update(
        StabilityInputs(
            algorithm=algorithm,
            mean_kl_old_new=_safe_float(update_row["mean_kl_old_new"]),
            nominal_kl_budget=nominal_kl_budget,
            next_eval_return=float(eval_history[-1]),
            trailing_eval_mean_prev3=float(sum(eval_history[-4:-1]) / 3.0),
            max_eval_so_far=float(max(eval_history[:-1])),
            initial_eval_mean_first3=float(sum(eval_history[:3]) / 3.0),
        )
    )
    update_row["unstable_update_flag"] = int(decision.unstable_update)
    update_row["large_step_no_drop"] = int(decision.large_step_no_drop)
    update_row["drop_without_large_step"] = int(decision.drop_without_large_step)
    if decision.unstable_update:
        cumulative_unstable_updates += 1
    return update_row, cumulative_unstable_updates


def run_training_job(
    job: TrainingJob,
    *,
    build_model: Callable[[TrainingJob, Any, str], Any],
) -> None:
    ensure_runtime_dependencies()
    from src.envs.make_env import make_env_bundle
    from src.callbacks.checkpoint_callback import CheckpointLayout, save_checkpoint_bundle
    from src.callbacks.eval_callback import evaluate_policy_checkpoint
    from src.callbacks.metrics_callback import MetricsWriters
    from src.callbacks.runtime_checks import RolloutStatsCallback

    set_global_seeds(job.seed)
    if job.algorithm in {"ppo_kl", "trpo"}:
        nominal_kl_budget = float(
            job.config["algo"]["target_kl"] if job.algorithm == "ppo_kl" else job.config["algo"]["max_kl_delta"]
        )
    else:
        nominal_kl_budget = 0.05

    if job.resume and (job.output_dir / "run_status.json").exists():
        status_payload = load_json(job.output_dir / "run_status.json")
        if status_payload.get("status") == "completed":
            return

    if job.output_dir.exists() and not job.dry_run and not job.resume:
        _reset_run_directory(job.output_dir, preserve_names={"stdout.log", "stderr.log"})
    job.output_dir.mkdir(parents=True, exist_ok=True)

    run_payload = _make_run_payload(job)
    save_json(job.output_dir / "run_config.json", run_payload)
    save_json(job.output_dir / "run_metadata.json", _run_metadata(job, run_payload))
    _save_collapse_payload(job.output_dir / job.config["logging"]["collapse_json"], _default_collapse_payload())
    _write_run_status(
        job.output_dir / "run_status.json",
        {
            "status": "running",
            "started_at": utc_now_iso(),
            "suite_name": job.suite_name,
            "run_tag": job.run_tag or ("default" if job.suite_name else None),
            "config_hash": run_payload["config_hash"],
        },
    )

    if job.dry_run:
        return

    train_env = None
    eval_env = None
    try:
        env_bundle = make_env_bundle(job.config)
        train_env = env_bundle.train_env
        eval_env = env_bundle.eval_env

        model = build_model(job, train_env, job.device)
        checkpoint_layout = CheckpointLayout(
            latest_checkpoint=job.config["logging"]["latest_checkpoint"],
            best_checkpoint=job.config["logging"]["best_checkpoint"],
            normalization_state=job.config["logging"]["normalization_state"],
            rng_state=job.config["logging"]["rng_state"],
        )
        writers = MetricsWriters(
            metrics_path=job.output_dir / job.config["logging"]["metrics_csv"],
            updates_path=job.output_dir / job.config["logging"]["updates_csv"],
        )

        start_time = time.perf_counter()
        eval_history: list[float] = []
        updates_rows: list[dict[str, Any]] = []
        cumulative_unstable_updates = 0
        best_eval_mean = -math.inf
        pending_update_index: int | None = None
        episodes_seen_total = 0
        train_returns_since_checkpoint: list[float] = []
        train_lengths_since_checkpoint: list[float] = []
        collapse_payload = _default_collapse_payload()
        collapse_tracker: CollapseTracker | None = None

        initial_eval = evaluate_policy_checkpoint(model, train_env, eval_env, job.config)
        eval_history.append(initial_eval["eval_return_mean"])
        metrics_row = _checkpoint_metric_row(
            env_steps=0,
            wall_clock_seconds=0.0,
            episodes_seen=0,
            eval_summary=initial_eval,
            train_episode_return_mean=math.nan,
            train_episode_length_mean=math.nan,
            update_metrics=_blank_metrics_row(),
            cumulative_unstable_updates=0,
            collapse_flag=False,
            nan_or_divergence_flag=False,
        )
        writers.log_checkpoint(metrics_row)
        save_checkpoint_bundle(
            model=model,
            train_env=train_env,
            run_dir=job.output_dir,
            checkpoint_layout=checkpoint_layout,
            rng_state=capture_rng_state(),
            is_best=True,
        )
        best_eval_mean = initial_eval["eval_return_mean"]
        next_eval_target = job.config["evaluation"]["checkpoint_every_env_steps"]

        while model.num_timesteps < job.config["env"]["total_timesteps"] and not collapse_payload["collapse_flag"]:
            update_start_steps = model.num_timesteps
            rollout_callback = RolloutStatsCallback(train_env.action_space)
            remaining = job.config["env"]["total_timesteps"] - model.num_timesteps
            learn_steps = min(job.config["environment"]["rollout_batch_size"], remaining)
            model.learn(
                total_timesteps=learn_steps,
                callback=rollout_callback,
                reset_num_timesteps=False,
                progress_bar=False,
            )
            update_end_steps = model.num_timesteps
            wall_clock_seconds = time.perf_counter() - start_time

            train_returns_since_checkpoint.extend(rollout_callback.episode_returns)
            train_lengths_since_checkpoint.extend(rollout_callback.episode_lengths)
            episodes_seen_total += len(rollout_callback.episode_returns)

            update_metrics = getattr(model, "last_training_info", {})
            numeric_issue, numeric_issue_reason = _detect_numeric_issue(update_metrics, rollout_callback)
            immediate_reason = _immediate_collapse_reason(update_metrics, rollout_callback)
            if numeric_issue and immediate_reason is None:
                immediate_reason = numeric_issue_reason

            update_row = _blank_update_row()
            update_row.update(
                {
                    "update_index": int(update_metrics.get("update_index", len(updates_rows) + 1)),
                    "start_env_steps": update_start_steps,
                    "end_env_steps": update_end_steps,
                    "wall_clock_seconds": wall_clock_seconds,
                    "mean_kl_old_new": update_metrics.get("mean_kl_old_new", math.nan),
                    "max_kl_old_new": update_metrics.get("max_kl_old_new", math.nan),
                    "policy_ratio_mean": update_metrics.get("policy_ratio_mean", math.nan),
                    "policy_ratio_std": update_metrics.get("policy_ratio_std", math.nan),
                    "epochs_completed": update_metrics.get("epochs_completed", math.nan),
                    "early_stopped": int(bool(update_metrics.get("early_stopped", False))),
                    "policy_loss": update_metrics.get("policy_loss", math.nan),
                    "value_loss": update_metrics.get("value_loss", math.nan),
                    "entropy": update_metrics.get("entropy", math.nan),
                }
            )
            updates_rows.append(update_row)
            pending_update_index = len(updates_rows) - 1

            if immediate_reason is not None:
                collapse_payload.update(
                    {
                        "collapse_flag": 1,
                        "collapse_step": update_end_steps,
                        "collapse_reason": immediate_reason,
                        "R_init": collapse_payload["R_init"],
                        "R_best": collapse_payload["R_best"],
                        "collapse_threshold": collapse_payload["collapse_threshold"],
                    }
                )

            while update_end_steps >= next_eval_target or (
                collapse_payload["collapse_flag"] and pending_update_index is not None
            ):
                eval_summary = evaluate_policy_checkpoint(model, train_env, eval_env, job.config)
                eval_history.append(eval_summary["eval_return_mean"])

                if len(eval_history) >= 4 and pending_update_index is not None:
                    updates_rows[pending_update_index], cumulative_unstable_updates = _complete_update_with_eval(
                        updates_rows[pending_update_index],
                        eval_history,
                        cumulative_unstable_updates,
                        job.algorithm,
                        nominal_kl_budget,
                    )
                    update_metrics["unstable_update_flag"] = updates_rows[pending_update_index]["unstable_update_flag"]

                if len(eval_history) == 4 and collapse_tracker is None:
                    r_init = float(sum(eval_history[1:4]) / 3.0)
                    collapse_tracker = CollapseTracker(initial_eval_mean=r_init, best_eval=max(eval_history[1:4]))
                    collapse_payload["R_init"] = r_init
                    collapse_payload["R_best"] = collapse_tracker.best_eval

                if collapse_tracker is not None and not collapse_payload["collapse_flag"]:
                    collapsed, collapse_threshold = collapse_tracker.update(eval_summary["eval_return_mean"])
                    collapse_payload["R_best"] = collapse_tracker.best_eval
                    collapse_payload["collapse_threshold"] = collapse_threshold
                    if collapsed:
                        collapse_payload["collapse_flag"] = 1
                        collapse_payload["collapse_step"] = update_end_steps
                        collapse_payload["collapse_reason"] = "relative_performance_collapse"

                train_return_mean, train_length_mean, _ = _episodes_summary(
                    train_returns_since_checkpoint,
                    train_lengths_since_checkpoint,
                )
                metrics_row = _checkpoint_metric_row(
                    env_steps=update_end_steps,
                    wall_clock_seconds=wall_clock_seconds,
                    episodes_seen=episodes_seen_total,
                    eval_summary=eval_summary,
                    train_episode_return_mean=train_return_mean,
                    train_episode_length_mean=train_length_mean,
                    update_metrics={
                        **_blank_metrics_row(),
                        **update_metrics,
                        "unstable_update_flag": updates_rows[pending_update_index]["unstable_update_flag"] if pending_update_index is not None else 0,
                    },
                    cumulative_unstable_updates=cumulative_unstable_updates,
                    collapse_flag=bool(collapse_payload["collapse_flag"]),
                    nan_or_divergence_flag=bool(numeric_issue),
                )
                writers.log_checkpoint(metrics_row)
                train_returns_since_checkpoint.clear()
                train_lengths_since_checkpoint.clear()

                is_best = eval_summary["eval_return_mean"] >= best_eval_mean
                if is_best:
                    best_eval_mean = eval_summary["eval_return_mean"]
                save_checkpoint_bundle(
                    model=model,
                    train_env=train_env,
                    run_dir=job.output_dir,
                    checkpoint_layout=checkpoint_layout,
                    rng_state=capture_rng_state(),
                    is_best=is_best,
                )
                _save_collapse_payload(job.output_dir / job.config["logging"]["collapse_json"], collapse_payload)
                _write_updates_table(writers.updates_path, updates_rows)
                next_eval_target += job.config["evaluation"]["checkpoint_every_env_steps"]
                if collapse_payload["collapse_flag"]:
                    break

        if not writers.updates_path.exists():
            _write_updates_table(writers.updates_path, updates_rows)
        if eval_history and updates_rows and pending_update_index is not None and model.num_timesteps > 0:
            last_metrics_path = job.output_dir / job.config["logging"]["metrics_csv"]
            line_count = len(last_metrics_path.read_text(encoding="utf-8").splitlines()) if last_metrics_path.exists() else 0
            expected_checkpoints = 1 + math.ceil(job.config["env"]["total_timesteps"] / job.config["evaluation"]["checkpoint_every_env_steps"])
            if not collapse_payload["collapse_flag"] and line_count < expected_checkpoints + 1:
                final_eval = evaluate_policy_checkpoint(model, train_env, eval_env, job.config)
                eval_history.append(final_eval["eval_return_mean"])
                updates_rows[pending_update_index], cumulative_unstable_updates = _complete_update_with_eval(
                    updates_rows[pending_update_index],
                    eval_history,
                    cumulative_unstable_updates,
                    job.algorithm,
                    nominal_kl_budget,
                )
                train_return_mean, train_length_mean, _ = _episodes_summary(
                    train_returns_since_checkpoint,
                    train_lengths_since_checkpoint,
                )
                metrics_row = _checkpoint_metric_row(
                    env_steps=model.num_timesteps,
                    wall_clock_seconds=time.perf_counter() - start_time,
                    episodes_seen=episodes_seen_total,
                    eval_summary=final_eval,
                    train_episode_return_mean=train_return_mean,
                    train_episode_length_mean=train_length_mean,
                    update_metrics={
                        **_blank_metrics_row(),
                        **update_metrics,
                        "unstable_update_flag": updates_rows[pending_update_index]["unstable_update_flag"],
                    },
                    cumulative_unstable_updates=cumulative_unstable_updates,
                    collapse_flag=bool(collapse_payload["collapse_flag"]),
                    nan_or_divergence_flag=bool(numeric_issue),
                )
                writers.log_checkpoint(metrics_row)
                _write_updates_table(writers.updates_path, updates_rows)
        _save_collapse_payload(job.output_dir / job.config["logging"]["collapse_json"], collapse_payload)
        target_timesteps = int(job.config["env"]["total_timesteps"])
        final_env_steps = int(model.num_timesteps)
        reached_target_timesteps = final_env_steps >= target_timesteps
        terminal_status = "completed" if reached_target_timesteps and not collapse_payload["collapse_flag"] else "collapsed"
        _write_run_status(
            job.output_dir / "run_status.json",
            {
                "status": terminal_status,
                "started_at": load_json(job.output_dir / "run_status.json")["started_at"],
                "completed_at": utc_now_iso(),
                "suite_name": job.suite_name,
                "run_tag": job.run_tag or ("default" if job.suite_name else None),
                "config_hash": run_payload["config_hash"],
                "target_env_steps": target_timesteps,
                "final_env_steps": final_env_steps,
                "reached_target_timesteps": reached_target_timesteps,
                "collapse_flag": int(collapse_payload["collapse_flag"]),
                "collapse_reason": collapse_payload["collapse_reason"],
            },
        )
    except Exception as exc:
        import traceback

        _write_error_file(job, str(exc), traceback.format_exc())
        raise
    finally:
        if train_env is not None:
            train_env.close()
        if eval_env is not None:
            eval_env.close()


def evaluate_saved_run(run_dir_path: Path, *, device: str = "auto") -> dict[str, Any]:
    ensure_runtime_dependencies()
    from src.envs.make_env import make_env_bundle
    from src.callbacks.eval_callback import evaluate_policy_checkpoint
    from stable_baselines3.common.vec_env import VecNormalize

    run_payload = load_json(run_dir_path / "run_config.json")
    config = run_payload["config"]
    env_bundle = make_env_bundle(config)
    train_env = env_bundle.train_env
    eval_env = VecNormalize.load(
        str(run_dir_path / config["logging"]["normalization_state"]),
        env_bundle.eval_env.venv,
    )
    eval_env.training = False
    eval_env.norm_reward = False
    algorithm = run_payload["algorithm"]
    module = get_algorithm_module(algorithm)
    model = module.load_model(run_dir_path / config["logging"]["latest_checkpoint"], eval_env, device)
    result = evaluate_policy_checkpoint(model, eval_env, eval_env, config)
    train_env.close()
    eval_env.close()
    return {
        "algorithm": algorithm,
        "env_key": run_payload["config"]["run"]["env_key"],
        "seed": run_payload["seed"],
        **result,
    }
