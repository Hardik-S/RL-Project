"""Figure construction for the benchmark analysis outputs."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PNG_DPI = 240


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    _ensure_dir(output_dir)
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}.png", dpi=PNG_DPI, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def configure_matplotlib() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "font.size": 10,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _color_map(labels: Iterable[str]) -> dict[str, tuple[float, float, float, float]]:
    unique = list(dict.fromkeys(labels))
    cmap = plt.get_cmap("tab10")
    return {label: cmap(index % cmap.N) for index, label in enumerate(unique)}


def plot_training_curves(curves: pd.DataFrame, output_dir: Path) -> None:
    if curves.empty:
        return
    configure_matplotlib()
    envs = list(dict.fromkeys(curves["env_id"].tolist()))
    algorithms = list(dict.fromkeys(curves["algorithm_display"].tolist()))
    colors = _color_map(algorithms)
    ncols = 2 if len(envs) > 1 else 1
    nrows = math.ceil(len(envs) / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6.5 * ncols, 3.8 * nrows), squeeze=False, sharex=False)

    for axis, env_id in zip(axes.flat, envs):
        env_frame = curves.loc[curves["env_id"] == env_id]
        for algorithm in algorithms:
            algo_frame = env_frame.loc[env_frame["algorithm_display"] == algorithm].sort_values("env_steps")
            if algo_frame.empty:
                continue
            axis.plot(
                algo_frame["env_steps"],
                algo_frame["mean"],
                label=algorithm,
                color=colors[algorithm],
                linewidth=2.0,
            )
            axis.fill_between(
                algo_frame["env_steps"],
                algo_frame["ci_lower"],
                algo_frame["ci_upper"],
                color=colors[algorithm],
                alpha=0.18,
                linewidth=0,
            )
        axis.set_title(env_id)
        axis.set_xlabel("Environment steps")
        axis.set_ylabel("Mean training return")

    for axis in axes.flat[len(envs) :]:
        axis.axis("off")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncols=min(len(labels), 4), frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Training-return curves with bootstrap uncertainty bands", y=1.05, fontsize=13, fontweight="bold")
    _save_figure(fig, output_dir, "training_return_curves")


def plot_wall_clock(wall_clock: pd.DataFrame, output_dir: Path) -> None:
    if wall_clock.empty:
        return
    configure_matplotlib()
    envs = list(dict.fromkeys(wall_clock["env_id"].tolist()))
    algorithms = list(dict.fromkeys(wall_clock["algorithm_display"].tolist()))
    colors = _color_map(algorithms)
    ncols = 2 if len(envs) > 1 else 1
    nrows = math.ceil(len(envs) / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6.5 * ncols, 3.9 * nrows), squeeze=False)

    for axis, env_id in zip(axes.flat, envs):
        env_frame = wall_clock.loc[wall_clock["env_id"] == env_id].sort_values("algorithm_display")
        x = np.arange(env_frame.shape[0])
        means = env_frame["mean"].to_numpy(dtype=float)
        lower = env_frame["mean"].to_numpy(dtype=float) - env_frame["ci_lower"].to_numpy(dtype=float)
        upper = env_frame["ci_upper"].to_numpy(dtype=float) - env_frame["mean"].to_numpy(dtype=float)
        axis.bar(
            x,
            means,
            color=[colors[label] for label in env_frame["algorithm_display"]],
            yerr=np.vstack([lower, upper]),
            capsize=4,
            alpha=0.9,
        )
        axis.set_title(env_id)
        axis.set_ylabel("Wall-clock seconds")
        axis.set_xticks(x, env_frame["algorithm_display"], rotation=20, ha="right")

    for axis in axes.flat[len(envs) :]:
        axis.axis("off")

    fig.suptitle("Wall-clock comparison by environment", y=1.02, fontsize=13, fontweight="bold")
    _save_figure(fig, output_dir, "wall_clock_comparison")


def plot_sensitivity(sensitivity: pd.DataFrame, output_dir: Path) -> None:
    if sensitivity.empty:
        return
    configure_matplotlib()
    panels = (
        sensitivity[["algorithm_display", "sweep_dimension"]]
        .drop_duplicates()
        .sort_values(["algorithm_display", "sweep_dimension"])
        .to_records(index=False)
    )
    envs = list(dict.fromkeys(sensitivity["env_id"].tolist()))
    colors = _color_map(envs)
    n_panels = len(panels)
    ncols = 2 if n_panels > 1 else 1
    nrows = math.ceil(n_panels / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6.5 * ncols, 3.9 * nrows), squeeze=False)

    for axis, (algorithm_display, sweep_dimension) in zip(axes.flat, panels):
        panel = sensitivity.loc[
            (sensitivity["algorithm_display"] == algorithm_display)
            & (sensitivity["sweep_dimension"] == sweep_dimension)
        ].sort_values(["env_id", "sweep_value"])
        for env_id in envs:
            env_frame = panel.loc[panel["env_id"] == env_id]
            if env_frame.empty:
                continue
            x = env_frame["sweep_value"].to_numpy(dtype=float)
            axis.plot(
                x,
                env_frame["mean_final_eval_return"],
                marker="o",
                linewidth=2.0,
                color=colors[env_id],
                label=env_id,
            )
            axis.fill_between(
                x,
                env_frame["ci_lower"],
                env_frame["ci_upper"],
                color=colors[env_id],
                alpha=0.18,
                linewidth=0,
            )
        axis.set_title(f"{algorithm_display}: {sweep_dimension}")
        axis.set_xlabel(sweep_dimension.replace("_", " "))
        axis.set_ylabel("Final evaluation return")

    for axis in axes.flat[n_panels:]:
        axis.axis("off")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncols=min(len(labels), 3), frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Sensitivity sweeps", y=1.04, fontsize=13, fontweight="bold")
    _save_figure(fig, output_dir, "sensitivity_plots")
