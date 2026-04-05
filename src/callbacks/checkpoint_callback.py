"""Checkpoint saving helpers."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.utils.serialization import save_pickle


@dataclass(frozen=True)
class CheckpointLayout:
    latest_checkpoint: str = "latest.pt"
    best_checkpoint: str = "best_by_eval_mean.pt"
    normalization_state: str = "vecnormalize.pkl"
    rng_state: str = "rng_state.pkl"

    def files(self, run_dir: Path) -> dict[str, Path]:
        return {
            "latest": run_dir / self.latest_checkpoint,
            "best": run_dir / self.best_checkpoint,
            "vecnormalize": run_dir / self.normalization_state,
            "rng_state": run_dir / self.rng_state,
        }


def save_checkpoint_bundle(
    *,
    model: Any,
    train_env: Any,
    run_dir: Path,
    checkpoint_layout: CheckpointLayout,
    rng_state: dict[str, Any],
    is_best: bool,
) -> None:
    files = checkpoint_layout.files(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    model.save(files["latest"])
    train_env.save(str(files["vecnormalize"]))
    save_pickle(files["rng_state"], rng_state)
    if is_best:
        shutil.copyfile(files["latest"], files["best"])
