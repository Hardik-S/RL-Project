import random
from pathlib import Path

import numpy as np

from src.callbacks.checkpoint_callback import CheckpointLayout
from src.utils.seeding import capture_rng_state, restore_rng_state, set_global_seeds


def test_checkpoint_layout_uses_frozen_names(tmp_path: Path) -> None:
    layout = CheckpointLayout()
    files = layout.files(tmp_path)
    assert files["latest"] == tmp_path / "latest.pt"
    assert files["best"] == tmp_path / "best_by_eval_mean.pt"
    assert files["vecnormalize"] == tmp_path / "vecnormalize.pkl"
    assert files["rng_state"] == tmp_path / "rng_state.pkl"


def test_rng_state_round_trip_restores_random_streams() -> None:
    set_global_seeds(123)
    state = capture_rng_state()
    python_first = random.random()
    numpy_first = float(np.random.rand())
    restore_rng_state(state)
    assert random.random() == python_first
    assert float(np.random.rand()) == numpy_first
