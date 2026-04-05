from pathlib import Path

from src.envs.normalization import (
    FrozenObservationStats,
    ObservationNormalizationConfig,
    build_vecnormalize_kwargs,
    load_frozen_stats,
    save_frozen_stats,
)


def test_reward_normalization_remains_disabled() -> None:
    kwargs = build_vecnormalize_kwargs(ObservationNormalizationConfig())
    assert kwargs["norm_obs"] is True
    assert kwargs["norm_reward"] is False
    assert kwargs["clip_obs"] == 10.0


def test_frozen_stats_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "vecnormalize.json"
    stats = FrozenObservationStats(
        mean=[0.0, 1.0],
        var=[1.0, 2.0],
        count=3.0,
        clip_obs=10.0,
        epsilon=1e-8,
    )
    save_frozen_stats(path, stats)
    loaded = load_frozen_stats(path)
    assert loaded == stats
