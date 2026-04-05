"""Data loading and aggregation helpers for benchmark analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.algos.common import ALGO_OVERRIDE_PATHS
from src.analysis.bootstrap_ci import bootstrap_mean_ci
from src.utils.paths import raw_results_root
from src.utils.serialization import load_json


MAIN_BENCHMARK_KIND = "main_benchmark"
SENSITIVITY_SWEEP_KIND = "sensitivity_sweep"


@dataclass(frozen=True)
class RunRecord:
    run_dir: Path
    suite_name: str | None
    suite_kind: str | None
    algorithm: str
    algorithm_display: str
    env_key: str
    env_id: str
    seed: int
    run_tag: str | None
    config_hash: str | None
    total_timesteps: int
    eval_interval: int
    final_env_steps: int
    auc_eval_return: float
    final_eval_return: float
    final_train_return: float
    unstable_updates_total: int
    unstable_updates_rate_per_100k: float
    collapse_flag: bool
    collapse_reason: str | None
    wall_clock_seconds_final: float
    seconds_per_10k_env_steps: float
    actor_lr: float | None
    clip_epsilon: float | None
    target_kl: float | None
    max_kl_delta: float | None
    sweep_dimension: str | None
    sweep_value: float | None


def discover_run_dirs(root: Path | None = None) -> list[Path]:
    root = root or raw_results_root()
    if not root.exists():
        return []
    return sorted(path.parent for path in root.rglob("metrics.csv"))


def _safe_mean(values: Iterable[float]) -> float:
    series = pd.Series(list(values), dtype=float).dropna()
    if series.empty:
        return float("nan")
    return float(series.mean())


def _tail_mean(series: pd.Series, n: int = 5) -> float:
    clean = series.dropna()
    if clean.empty:
        return float("nan")
    return float(clean.tail(min(n, clean.shape[0])).mean())


def _safe_auc(frame: pd.DataFrame) -> float:
    valid = frame.loc[frame["eval_return_mean"].notna() & frame["env_steps"].notna(), ["env_steps", "eval_return_mean"]]
    if valid.empty:
        return float("nan")
    if valid.shape[0] == 1:
        return float(valid["eval_return_mean"].iloc[0])
    integrator = getattr(np, "trapezoid", np.trapz)
    return float(integrator(valid["eval_return_mean"], valid["env_steps"]))


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(numeric):
        return None
    return numeric


def _numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = frame.copy()
    for column in numeric.columns:
        try:
            numeric[column] = pd.to_numeric(numeric[column])
        except (TypeError, ValueError):
            continue
    return numeric


def _get_nested(config: dict[str, Any], path: tuple[str, ...]) -> Any:
    cursor: Any = config
    for key in path:
        if not isinstance(cursor, dict) or key not in cursor:
            return None
        cursor = cursor[key]
    return cursor


def _infer_metadata_from_path(run_dir: Path, raw_root: Path) -> dict[str, Any]:
    parts = run_dir.relative_to(raw_root).parts
    if len(parts) == 3:
        algorithm, env_key, seed_dir = parts
        suite_name = None
        run_tag = None
    elif len(parts) >= 5:
        suite_name, algorithm, env_key, seed_dir, run_tag = parts[:5]
    else:
        raise ValueError(f"Unrecognized run directory layout: {run_dir}")
    return {
        "suite_name": suite_name,
        "algorithm": algorithm,
        "env_key": env_key,
        "seed": int(seed_dir.removeprefix("seed_")),
        "run_tag": run_tag if len(parts) >= 5 else None,
    }


def _suite_kind(suite_name: str | None, manifests_root: Path) -> str | None:
    if not suite_name:
        return None
    suite_path = manifests_root / suite_name / "suite.json"
    if not suite_path.exists():
        return None
    return load_json(suite_path).get("suite_kind")


def _parse_sweep_dimension(run_tag: str | None) -> str | None:
    if not run_tag or run_tag == "default":
        return None
    matches = [name for name in ALGO_OVERRIDE_PATHS if run_tag == name or run_tag.startswith(f"{name}_")]
    if not matches:
        return None
    return max(matches, key=len)


def load_run_record(
    run_dir: Path,
    *,
    manifests_root: Path | None = None,
    raw_root: Path | None = None,
) -> tuple[RunRecord, pd.DataFrame, pd.DataFrame]:
    metrics_path = run_dir / "metrics.csv"
    updates_path = run_dir / "updates.csv"
    collapse_path = run_dir / "collapse.json"
    run_config_path = run_dir / "run_config.json"

    metrics = _numeric_frame(pd.read_csv(metrics_path)).sort_values("env_steps").reset_index(drop=True)
    updates = _numeric_frame(pd.read_csv(updates_path)) if updates_path.exists() else pd.DataFrame()
    collapse_payload = load_json(collapse_path) if collapse_path.exists() else {}

    raw_root = raw_root or raw_results_root()

    if run_config_path.exists():
        run_payload = load_json(run_config_path)
        config = run_payload["config"]
        suite_name = config["run"].get("suite_name")
        algorithm = str(run_payload["algorithm"])
        algorithm_display = str(config["algo"].get("display_name", algorithm))
        env_key = str(config["run"]["env_key"])
        env_id = str(config["env"]["env_id"])
        seed = int(run_payload["seed"])
        run_tag = config["run"].get("run_tag")
        config_hash = run_payload.get("config_hash")
        total_timesteps = int(config["env"]["total_timesteps"])
        eval_interval = int(config["evaluation"]["checkpoint_every_env_steps"])
        actor_lr = _safe_float(config.get("algo", {}).get("actor_lr"))
        clip_epsilon = _safe_float(config.get("algo", {}).get("clip_epsilon"))
        target_kl = _safe_float(config.get("algo", {}).get("target_kl"))
        max_kl_delta = _safe_float(config.get("algo", {}).get("max_kl_delta"))
    else:
        inferred = _infer_metadata_from_path(run_dir, raw_root)
        suite_name = inferred["suite_name"]
        algorithm = inferred["algorithm"]
        algorithm_display = algorithm.upper()
        env_key = inferred["env_key"]
        env_id = env_key
        seed = inferred["seed"]
        run_tag = inferred["run_tag"]
        config_hash = None
        total_timesteps = int(metrics["env_steps"].max())
        eval_interval = int(metrics["env_steps"].diff().dropna().mode().iloc[0]) if metrics.shape[0] > 1 else total_timesteps
        actor_lr = None
        clip_epsilon = None
        target_kl = None
        max_kl_delta = None

    manifests_root = manifests_root or raw_root.parent / "manifests"
    suite_kind = _suite_kind(suite_name, manifests_root)
    unstable_updates_total = int(
        updates["unstable_update_flag"].fillna(0).astype(int).sum()
        if not updates.empty and "unstable_update_flag" in updates
        else metrics["cumulative_unstable_updates"].fillna(0).max()
    )
    final_env_steps = int(metrics["env_steps"].dropna().iloc[-1]) if not metrics.empty else 0
    wall_clock_seconds_final = float(metrics["wall_clock_seconds"].dropna().iloc[-1]) if not metrics.empty else float("nan")
    unstable_rate = float(unstable_updates_total * 100000.0 / final_env_steps) if final_env_steps > 0 else float("nan")
    seconds_per_10k = float(wall_clock_seconds_final * 10000.0 / final_env_steps) if final_env_steps > 0 else float("nan")
    sweep_dimension = _parse_sweep_dimension(run_tag) if suite_kind == SENSITIVITY_SWEEP_KIND else None
    sweep_value = None
    if sweep_dimension:
        override_path = ALGO_OVERRIDE_PATHS.get(sweep_dimension)
        if override_path is not None and run_config_path.exists():
            sweep_value = _safe_float(_get_nested(config, override_path))

    record = RunRecord(
        run_dir=run_dir,
        suite_name=suite_name,
        suite_kind=suite_kind,
        algorithm=algorithm,
        algorithm_display=algorithm_display,
        env_key=env_key,
        env_id=env_id,
        seed=seed,
        run_tag=run_tag,
        config_hash=config_hash,
        total_timesteps=total_timesteps,
        eval_interval=eval_interval,
        final_env_steps=final_env_steps,
        auc_eval_return=_safe_auc(metrics),
        final_eval_return=_tail_mean(metrics["eval_return_mean"]),
        final_train_return=_tail_mean(metrics["train_episode_return_mean"]),
        unstable_updates_total=unstable_updates_total,
        unstable_updates_rate_per_100k=unstable_rate,
        collapse_flag=bool(collapse_payload.get("collapse_flag", bool(metrics["collapse_flag"].fillna(0).max()))),
        collapse_reason=collapse_payload.get("collapse_reason"),
        wall_clock_seconds_final=wall_clock_seconds_final,
        seconds_per_10k_env_steps=seconds_per_10k,
        actor_lr=actor_lr,
        clip_epsilon=clip_epsilon,
        target_kl=target_kl,
        max_kl_delta=max_kl_delta,
        sweep_dimension=sweep_dimension,
        sweep_value=sweep_value,
    )
    return record, metrics, updates


def load_run_bundle(root: Path | None = None) -> tuple[list[RunRecord], pd.DataFrame, pd.DataFrame]:
    root = root or raw_results_root()
    manifests_root = root.parent / "manifests"
    records: list[RunRecord] = []
    checkpoint_frames: list[pd.DataFrame] = []
    update_frames: list[pd.DataFrame] = []
    for run_dir in discover_run_dirs(root):
        record, metrics, updates = load_run_record(run_dir, manifests_root=manifests_root, raw_root=root)
        records.append(record)

        metrics = metrics.copy()
        metrics["algorithm"] = record.algorithm
        metrics["algorithm_display"] = record.algorithm_display
        metrics["env_key"] = record.env_key
        metrics["env_id"] = record.env_id
        metrics["seed"] = record.seed
        metrics["suite_name"] = record.suite_name
        metrics["suite_kind"] = record.suite_kind
        metrics["run_tag"] = record.run_tag
        metrics["run_dir"] = str(record.run_dir)
        checkpoint_frames.append(metrics)

        if not updates.empty:
            updates = updates.copy()
            updates["algorithm"] = record.algorithm
            updates["algorithm_display"] = record.algorithm_display
            updates["env_key"] = record.env_key
            updates["env_id"] = record.env_id
            updates["seed"] = record.seed
            updates["suite_name"] = record.suite_name
            updates["suite_kind"] = record.suite_kind
            updates["run_tag"] = record.run_tag
            updates["run_dir"] = str(record.run_dir)
            update_frames.append(updates)

    run_summary = pd.DataFrame(asdict(record) for record in records)
    checkpoints = pd.concat(checkpoint_frames, ignore_index=True) if checkpoint_frames else pd.DataFrame()
    updates = pd.concat(update_frames, ignore_index=True) if update_frames else pd.DataFrame()
    return records, run_summary, checkpoints, updates


def primary_run_summary(run_summary: pd.DataFrame) -> pd.DataFrame:
    if run_summary.empty:
        return run_summary.copy()
    if (run_summary["suite_kind"] == MAIN_BENCHMARK_KIND).any():
        return run_summary.loc[run_summary["suite_kind"] == MAIN_BENCHMARK_KIND].copy()
    return run_summary.loc[run_summary["suite_kind"] != SENSITIVITY_SWEEP_KIND].copy()


def sweep_run_summary(run_summary: pd.DataFrame) -> pd.DataFrame:
    if run_summary.empty:
        return run_summary.copy()
    return run_summary.loc[run_summary["suite_kind"] == SENSITIVITY_SWEEP_KIND].copy()


def primary_checkpoints(checkpoints: pd.DataFrame) -> pd.DataFrame:
    if checkpoints.empty:
        return checkpoints.copy()
    if (checkpoints["suite_kind"] == MAIN_BENCHMARK_KIND).any():
        return checkpoints.loc[checkpoints["suite_kind"] == MAIN_BENCHMARK_KIND].copy()
    return checkpoints.loc[checkpoints["suite_kind"] != SENSITIVITY_SWEEP_KIND].copy()


def summarize_curve_metric(
    checkpoints: pd.DataFrame,
    *,
    metric: str,
) -> pd.DataFrame:
    columns = [
        "algorithm",
        "algorithm_display",
        "env_key",
        "env_id",
        "env_steps",
        "metric",
        "mean",
        "std",
        "ci_lower",
        "ci_upper",
        "n_runs",
    ]
    if checkpoints.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    group_columns = ["algorithm", "algorithm_display", "env_key", "env_id", "env_steps"]
    for keys, frame in checkpoints.groupby(group_columns, dropna=False):
        values = frame[metric].dropna().astype(float).tolist()
        if not values:
            continue
        interval = bootstrap_mean_ci(values) if len(values) > 1 else None
        algorithm, algorithm_display, env_key, env_id, env_steps = keys
        rows.append(
            {
                "algorithm": algorithm,
                "algorithm_display": algorithm_display,
                "env_key": env_key,
                "env_id": env_id,
                "env_steps": float(env_steps),
                "metric": metric,
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "ci_lower": interval.lower if interval else float(values[0]),
                "ci_upper": interval.upper if interval else float(values[0]),
                "n_runs": len(values),
            }
        )
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows).sort_values(["env_id", "algorithm_display", "env_steps"]).reset_index(drop=True)


def summarize_scalar_metric(
    run_summary: pd.DataFrame,
    *,
    value_column: str,
) -> pd.DataFrame:
    if run_summary.empty:
        return pd.DataFrame()
    if value_column == "auc_eval_return" and ("env_id" not in run_summary or "env_key" not in run_summary):
        raise ValueError("AUC summaries must retain per-environment grouping columns.")
    rows: list[dict[str, Any]] = []
    for keys, frame in run_summary.groupby(["algorithm", "algorithm_display", "env_key", "env_id"], dropna=False):
        values = frame[value_column].dropna().astype(float).tolist()
        if not values:
            continue
        interval = bootstrap_mean_ci(values) if len(values) > 1 else None
        algorithm, algorithm_display, env_key, env_id = keys
        rows.append(
            {
                "algorithm": algorithm,
                "algorithm_display": algorithm_display,
                "env_key": env_key,
                "env_id": env_id,
                "metric": value_column,
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "variance": float(np.var(values, ddof=1)) if len(values) > 1 else 0.0,
                "ci_lower": interval.lower if interval else float(values[0]),
                "ci_upper": interval.upper if interval else float(values[0]),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "n_runs": len(values),
            }
        )
    return pd.DataFrame(rows).sort_values(["env_id", "algorithm_display"]).reset_index(drop=True)


def summarize_instability(run_summary: pd.DataFrame) -> pd.DataFrame:
    if run_summary.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for keys, frame in run_summary.groupby(["algorithm", "algorithm_display", "env_key", "env_id"], dropna=False):
        algorithm, algorithm_display, env_key, env_id = keys
        unstable_totals = frame["unstable_updates_total"].astype(float).tolist()
        unstable_rates = frame["unstable_updates_rate_per_100k"].astype(float).tolist()
        collapse_flags = frame["collapse_flag"].astype(int).tolist()
        unstable_interval = bootstrap_mean_ci(unstable_rates) if len(unstable_rates) > 1 else None
        rows.append(
            {
                "algorithm": algorithm,
                "algorithm_display": algorithm_display,
                "env_key": env_key,
                "env_id": env_id,
                "unstable_updates_mean": float(np.mean(unstable_totals)),
                "unstable_updates_rate_per_100k_mean": float(np.mean(unstable_rates)),
                "unstable_updates_rate_per_100k_std": float(np.std(unstable_rates, ddof=1)) if len(unstable_rates) > 1 else 0.0,
                "unstable_updates_rate_per_100k_ci_lower": unstable_interval.lower if unstable_interval else float(unstable_rates[0]),
                "unstable_updates_rate_per_100k_ci_upper": unstable_interval.upper if unstable_interval else float(unstable_rates[0]),
                "collapse_rate": float(np.mean(collapse_flags)),
                "collapse_count": int(np.sum(collapse_flags)),
                "n_runs": int(frame.shape[0]),
            }
        )
    return pd.DataFrame(rows).sort_values(["env_id", "algorithm_display"]).reset_index(drop=True)


def summarize_sensitivity(run_summary: pd.DataFrame) -> pd.DataFrame:
    sweep_runs = sweep_run_summary(run_summary)
    if sweep_runs.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    valid = sweep_runs.loc[sweep_runs["sweep_dimension"].notna() & sweep_runs["sweep_value"].notna()].copy()
    for keys, frame in valid.groupby(
        ["algorithm", "algorithm_display", "env_key", "env_id", "sweep_dimension", "sweep_value"],
        dropna=False,
    ):
        values = frame["final_eval_return"].dropna().astype(float).tolist()
        if not values:
            continue
        interval = bootstrap_mean_ci(values) if len(values) > 1 else None
        algorithm, algorithm_display, env_key, env_id, sweep_dimension, sweep_value = keys
        rows.append(
            {
                "algorithm": algorithm,
                "algorithm_display": algorithm_display,
                "env_key": env_key,
                "env_id": env_id,
                "sweep_dimension": sweep_dimension,
                "sweep_value": float(sweep_value),
                "mean_final_eval_return": float(np.mean(values)),
                "std_final_eval_return": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "ci_lower": interval.lower if interval else float(values[0]),
                "ci_upper": interval.upper if interval else float(values[0]),
                "n_runs": len(values),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["algorithm_display", "sweep_dimension", "env_id", "sweep_value"]
    ).reset_index(drop=True)
