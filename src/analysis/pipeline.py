"""End-to-end analysis pipeline for tables, summaries, and figures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.analysis.aggregate import (
    load_run_bundle,
    primary_checkpoints,
    primary_run_summary,
    summarize_curve_metric,
    summarize_instability,
    summarize_scalar_metric,
    summarize_sensitivity,
)
from src.analysis.plots import plot_sensitivity, plot_training_curves, plot_wall_clock
from src.utils.paths import (
    aggregated_figures_dir,
    aggregated_results_root,
    aggregated_summaries_dir,
    aggregated_tables_dir,
    raw_results_root,
)


def _ensure_dirs(root: Path | None = None) -> dict[str, Path]:
    base = root or aggregated_results_root()
    figures = base / "figures"
    tables = base / "tables"
    summaries = base / "summaries"
    for path in (base, figures, tables, summaries):
        path.mkdir(parents=True, exist_ok=True)
    return {"root": base, "figures": figures, "tables": tables, "summaries": summaries}


def _format_interval(mean: float, lower: float, upper: float, digits: int = 1) -> str:
    return f"{mean:.{digits}f} [{lower:.{digits}f}, {upper:.{digits}f}]"


def _format_scalar(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False)


def _write_latex(frame: pd.DataFrame, path: Path) -> None:
    latex = frame.to_latex(index=False, escape=False)
    path.write_text(latex, encoding="utf-8")


def _latex_pivot(
    frame: pd.DataFrame,
    *,
    value_builder,
    value_name: str,
) -> pd.DataFrame:
    formatted = frame.copy()
    formatted[value_name] = formatted.apply(value_builder, axis=1)
    pivot = (
        formatted.pivot(index="algorithm_display", columns="env_id", values=value_name)
        .reset_index()
        .rename_axis(columns=None)
        .sort_values("algorithm_display")
    )
    return pivot


def build_analysis_outputs(
    *,
    raw_root: Path | None = None,
    aggregated_root: Path | None = None,
) -> dict[str, Any]:
    raw_root = raw_root or raw_results_root()
    outputs = _ensure_dirs(aggregated_root)
    records, run_summary, checkpoints, updates = load_run_bundle(raw_root)
    if run_summary.empty:
        raise RuntimeError(f"No completed runs with metrics were found under {raw_root}.")

    primary_runs = primary_run_summary(run_summary)
    primary_ckpts = primary_checkpoints(checkpoints)
    curves = summarize_curve_metric(primary_ckpts, metric="train_episode_return_mean")
    if curves.empty:
        curves = summarize_curve_metric(primary_ckpts, metric="eval_return_mean")

    final_eval = summarize_scalar_metric(primary_runs, value_column="final_eval_return")
    variance = summarize_scalar_metric(primary_runs, value_column="final_eval_return")
    wall_clock = summarize_scalar_metric(primary_runs, value_column="wall_clock_seconds_final")
    instability = summarize_instability(primary_runs)
    sensitivity = summarize_sensitivity(run_summary)

    _write_csv(run_summary, outputs["tables"] / "run_summary.csv")
    _write_csv(primary_runs, outputs["tables"] / "primary_run_summary.csv")
    _write_csv(checkpoints, outputs["tables"] / "checkpoint_metrics.csv")
    _write_csv(updates, outputs["tables"] / "update_metrics.csv")
    _write_csv(curves, outputs["tables"] / "curve_summary.csv")
    _write_csv(final_eval, outputs["tables"] / "final_evaluation_return.csv")
    _write_csv(variance[["algorithm", "algorithm_display", "env_key", "env_id", "std", "variance", "min", "max", "n_runs"]], outputs["tables"] / "final_evaluation_variance.csv")
    _write_csv(wall_clock, outputs["tables"] / "wall_clock_comparison.csv")
    _write_csv(instability, outputs["tables"] / "instability_frequency.csv")
    _write_csv(sensitivity, outputs["tables"] / "sensitivity_summary.csv")

    final_eval_latex = _latex_pivot(
        final_eval,
        value_builder=lambda row: _format_interval(row["mean"], row["ci_lower"], row["ci_upper"], digits=1),
        value_name="final_eval_summary",
    )
    variance_latex = _latex_pivot(
        variance,
        value_builder=lambda row: f"{_format_scalar(row['std'], digits=1)} / {_format_scalar(row['variance'], digits=1)}",
        value_name="std_var_summary",
    )
    wall_clock_latex = _latex_pivot(
        wall_clock,
        value_builder=lambda row: _format_interval(row["mean"], row["ci_lower"], row["ci_upper"], digits=1),
        value_name="wall_clock_summary",
    )
    instability_latex = _latex_pivot(
        instability,
        value_builder=lambda row: (
            f"{_format_scalar(row['unstable_updates_rate_per_100k_mean'], digits=2)} "
            f"[{_format_scalar(row['unstable_updates_rate_per_100k_ci_lower'], digits=2)}, "
            f"{_format_scalar(row['unstable_updates_rate_per_100k_ci_upper'], digits=2)}], "
            f"collapse={_format_scalar(row['collapse_rate'], digits=2)}"
        ),
        value_name="instability_summary",
    )

    _write_latex(final_eval_latex, outputs["tables"] / "final_evaluation_return.tex")
    _write_latex(variance_latex, outputs["tables"] / "final_evaluation_variance.tex")
    _write_latex(wall_clock_latex, outputs["tables"] / "wall_clock_comparison.tex")
    _write_latex(instability_latex, outputs["tables"] / "instability_frequency.tex")

    plot_training_curves(curves, outputs["figures"])
    plot_wall_clock(wall_clock, outputs["figures"])
    plot_sensitivity(sensitivity, outputs["figures"])

    summary_payload = {
        "run_count": int(run_summary.shape[0]),
        "primary_run_count": int(primary_runs.shape[0]),
        "sweep_run_count": int(sensitivity["n_runs"].sum()) if not sensitivity.empty else 0,
        "algorithms": sorted(run_summary["algorithm"].dropna().unique().tolist()),
        "environments": sorted(run_summary["env_key"].dropna().unique().tolist()),
        "tables": sorted(path.name for path in outputs["tables"].iterdir() if path.is_file()),
        "figures": sorted(path.name for path in outputs["figures"].iterdir() if path.is_file()),
    }
    (outputs["summaries"] / "analysis_summary.json").write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary_payload


def default_output_dirs() -> dict[str, Path]:
    return {
        "tables": aggregated_tables_dir(),
        "figures": aggregated_figures_dir(),
        "summaries": aggregated_summaries_dir(),
    }
