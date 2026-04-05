from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.pipeline import build_analysis_outputs
from src.metrics.logging_schema import COLLAPSE_FIELDS, METRICS_COLUMNS, UPDATES_COLUMNS
from src.utils.manifests import stable_config_hash
from src.utils.serialization import save_json


def _blank_row(columns: list[str]) -> dict[str, float]:
    return {column: np.nan for column in columns}


def _write_suite_metadata(results_root: Path, suite_name: str, suite_kind: str) -> None:
    save_json(
        results_root / "manifests" / suite_name / "suite.json",
        {"suite_name": suite_name, "suite_kind": suite_kind},
    )


def _write_run(
    raw_root: Path,
    *,
    suite_name: str | None,
    suite_kind: str | None,
    algorithm: str,
    env_key: str,
    seed: int,
    run_tag: str | None,
    overrides: dict[str, float] | None,
    eval_returns: list[float],
    train_returns: list[float],
    unstable_flags: list[int],
    wall_clock_scale: float,
    collapse_flag: int = 0,
) -> None:
    if suite_name and suite_kind:
        _write_suite_metadata(raw_root.parent, suite_name, suite_kind)

    env_id = {
        "hopper_v4": "Hopper-v4",
        "halfcheetah_v4": "HalfCheetah-v4",
        "walker2d_v4": "Walker2d-v4",
        "pendulum_v1": "Pendulum-v1",
    }[env_key]
    display_name = {
        "a2c": "A2C",
        "ppo_clip": "PPO-Clip",
        "ppo_kl": "PPO-KL",
        "trpo": "TRPO",
    }[algorithm]
    algo_defaults = {
        "a2c": {"actor_lr": 0.0003, "critic_lr": 0.001},
        "ppo_clip": {"actor_lr": 0.0003, "critic_lr": 0.001, "clip_epsilon": 0.20},
        "ppo_kl": {"actor_lr": 0.0003, "critic_lr": 0.001, "target_kl": 0.02},
        "trpo": {"critic_lr": 0.001, "max_kl_delta": 0.02},
    }[algorithm]
    algo_config = {**algo_defaults, **(overrides or {})}
    config = {
        "algo": {"display_name": display_name, **algo_config},
        "env": {"env_id": env_id, "total_timesteps": 30000},
        "evaluation": {"checkpoint_every_env_steps": 10000},
        "run": {
            "algorithm": algorithm,
            "env_key": env_key,
            "seed": seed,
            "suite_name": suite_name,
            "run_tag": run_tag,
        },
    }
    if suite_name:
        run_dir = raw_root / suite_name / algorithm / env_key / f"seed_{seed}" / (run_tag or "default")
    else:
        run_dir = raw_root / algorithm / env_key / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "algorithm": algorithm,
        "config": config,
        "config_hash": stable_config_hash(config),
        "seed": seed,
    }
    save_json(run_dir / "run_config.json", payload)

    steps = [0, 10000, 20000, 30000]
    metrics_rows: list[dict[str, float]] = []
    cumulative_unstable = 0
    for index, step in enumerate(steps):
        row = _blank_row(METRICS_COLUMNS)
        row.update(
            {
                "env_steps": step,
                "wall_clock_seconds": wall_clock_scale * (index + 1),
                "episodes_seen": 10 * index,
                "eval_return_mean": eval_returns[index],
                "eval_return_std": 1.0 + index,
                "eval_return_median": eval_returns[index],
                "train_episode_return_mean": train_returns[index],
                "train_episode_length_mean": 200.0,
                "mean_kl_old_new": 0.01,
                "max_kl_old_new": 0.02,
                "policy_ratio_mean": 1.0,
                "policy_ratio_std": 0.05,
                "unstable_update_flag": 0 if index == 0 else unstable_flags[index - 1],
                "collapse_flag": collapse_flag,
                "nan_or_divergence_flag": 0,
                "policy_loss": -0.1,
                "value_loss": 0.2,
                "entropy": 0.3,
                "advantage_mean": 0.0,
                "advantage_std": 1.0,
                "value_target_mean": 0.4,
                "grad_norm_actor": 0.5,
                "grad_norm_critic": 0.6,
            }
        )
        if index > 0:
            cumulative_unstable += unstable_flags[index - 1]
        row["cumulative_unstable_updates"] = cumulative_unstable
        metrics_rows.append(row)
    pd.DataFrame(metrics_rows).to_csv(run_dir / "metrics.csv", index=False)

    updates_rows: list[dict[str, float]] = []
    for index, flag in enumerate(unstable_flags, start=1):
        row = _blank_row(UPDATES_COLUMNS)
        row.update(
            {
                "update_index": index,
                "start_env_steps": (index - 1) * 10000,
                "end_env_steps": index * 10000,
                "wall_clock_seconds": wall_clock_scale * (index + 1),
                "mean_kl_old_new": 0.01,
                "max_kl_old_new": 0.02,
                "policy_ratio_mean": 1.0,
                "policy_ratio_std": 0.05,
                "large_step_no_drop": 0,
                "drop_without_large_step": 0,
                "unstable_update_flag": flag,
                "epochs_completed": 10,
                "early_stopped": 0,
                "policy_loss": -0.1,
                "value_loss": 0.2,
                "entropy": 0.3,
            }
        )
        updates_rows.append(row)
    pd.DataFrame(updates_rows).to_csv(run_dir / "updates.csv", index=False)

    collapse = {field: None for field in COLLAPSE_FIELDS}
    collapse["collapse_flag"] = collapse_flag
    collapse["collapse_reason"] = "relative_performance_collapse" if collapse_flag else None
    save_json(run_dir / "collapse.json", collapse)


def test_build_analysis_outputs_exports_report_assets(tmp_path: Path) -> None:
    raw_root = tmp_path / "results" / "raw"
    aggregated_root = tmp_path / "results" / "aggregated"

    _write_run(
        raw_root,
        suite_name="main_benchmark",
        suite_kind="main_benchmark",
        algorithm="a2c",
        env_key="hopper_v4",
        seed=0,
        run_tag="default",
        overrides=None,
        eval_returns=[10.0, 20.0, 22.0, 24.0],
        train_returns=[np.nan, 11.0, 18.0, 21.0],
        unstable_flags=[0, 1, 0],
        wall_clock_scale=12.0,
    )
    _write_run(
        raw_root,
        suite_name="main_benchmark",
        suite_kind="main_benchmark",
        algorithm="a2c",
        env_key="hopper_v4",
        seed=1,
        run_tag="default",
        overrides=None,
        eval_returns=[9.0, 18.0, 21.0, 23.0],
        train_returns=[np.nan, 10.0, 16.0, 20.0],
        unstable_flags=[0, 0, 1],
        wall_clock_scale=13.0,
    )
    _write_run(
        raw_root,
        suite_name="main_benchmark",
        suite_kind="main_benchmark",
        algorithm="ppo_clip",
        env_key="hopper_v4",
        seed=0,
        run_tag="default",
        overrides=None,
        eval_returns=[12.0, 24.0, 28.0, 30.0],
        train_returns=[np.nan, 14.0, 21.0, 26.0],
        unstable_flags=[0, 0, 0],
        wall_clock_scale=15.0,
    )
    _write_run(
        raw_root,
        suite_name="main_benchmark",
        suite_kind="main_benchmark",
        algorithm="ppo_clip",
        env_key="hopper_v4",
        seed=1,
        run_tag="default",
        overrides=None,
        eval_returns=[11.0, 23.0, 27.0, 29.0],
        train_returns=[np.nan, 13.0, 20.0, 25.0],
        unstable_flags=[0, 0, 0],
        wall_clock_scale=16.0,
    )

    _write_run(
        raw_root,
        suite_name="sweep_hopper_v4",
        suite_kind="sensitivity_sweep",
        algorithm="ppo_clip",
        env_key="hopper_v4",
        seed=0,
        run_tag="actor_lr_0p0001",
        overrides={"actor_lr": 0.0001},
        eval_returns=[10.0, 19.0, 20.0, 21.0],
        train_returns=[np.nan, 10.5, 15.0, 17.0],
        unstable_flags=[0, 0, 0],
        wall_clock_scale=14.0,
    )
    _write_run(
        raw_root,
        suite_name="sweep_hopper_v4",
        suite_kind="sensitivity_sweep",
        algorithm="ppo_clip",
        env_key="hopper_v4",
        seed=1,
        run_tag="actor_lr_0p0003",
        overrides={"actor_lr": 0.0003},
        eval_returns=[11.0, 22.0, 24.0, 26.0],
        train_returns=[np.nan, 12.0, 17.0, 19.0],
        unstable_flags=[0, 0, 0],
        wall_clock_scale=14.5,
    )

    summary = build_analysis_outputs(raw_root=raw_root, aggregated_root=aggregated_root)

    assert summary["run_count"] == 6
    assert "training_return_curves.png" in summary["figures"]
    assert "sensitivity_plots.pdf" in summary["figures"]

    final_eval = pd.read_csv(aggregated_root / "tables" / "final_evaluation_return.csv")
    assert set(final_eval["algorithm"]) == {"a2c", "ppo_clip"}
    assert set(final_eval["env_key"]) == {"hopper_v4"}

    sensitivity = pd.read_csv(aggregated_root / "tables" / "sensitivity_summary.csv")
    assert set(sensitivity["sweep_dimension"]) == {"actor_lr"}

    for expected in [
        aggregated_root / "tables" / "final_evaluation_return.tex",
        aggregated_root / "tables" / "instability_frequency.tex",
        aggregated_root / "figures" / "training_return_curves.png",
        aggregated_root / "figures" / "wall_clock_comparison.pdf",
        aggregated_root / "summaries" / "analysis_summary.json",
    ]:
        assert expected.exists()


def test_build_analysis_outputs_rejects_empty_raw_root(tmp_path: Path) -> None:
    raw_root = tmp_path / "results" / "raw"
    aggregated_root = tmp_path / "results" / "aggregated"
    raw_root.mkdir(parents=True)

    try:
        build_analysis_outputs(raw_root=raw_root, aggregated_root=aggregated_root)
    except RuntimeError as exc:
        assert "No completed runs" in str(exc)
    else:
        raise AssertionError("expected build_analysis_outputs to reject an empty raw root")


def test_summarize_curve_metric_tolerates_all_nan_metric_values() -> None:
    from src.analysis.aggregate import summarize_curve_metric

    checkpoints = pd.DataFrame(
        [
            {
                "algorithm": "a2c",
                "algorithm_display": "A2C",
                "env_key": "pendulum_v1",
                "env_id": "Pendulum-v1",
                "env_steps": 0,
                "train_episode_return_mean": np.nan,
            },
            {
                "algorithm": "a2c",
                "algorithm_display": "A2C",
                "env_key": "pendulum_v1",
                "env_id": "Pendulum-v1",
                "env_steps": 10000,
                "train_episode_return_mean": np.nan,
            },
        ]
    )

    summary = summarize_curve_metric(checkpoints, metric="train_episode_return_mean")

    assert summary.empty
    assert list(summary.columns) == [
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
