from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.algos.common import _immediate_collapse_reason, _reset_run_directory
from src.callbacks.runtime_checks import RolloutStatsCallback
from src.utils.serialization import save_json
from src.utils.suite_runner import build_run_spec, is_completed_run, run_suite, suite_command


def test_rollout_callback_does_not_reject_finite_out_of_bounds_actions() -> None:
    action_space = SimpleNamespace(low=np.array([-2.0], dtype=np.float32), high=np.array([2.0], dtype=np.float32))
    callback = RolloutStatsCallback(action_space)
    callback.locals = {
        "actions": np.array([[3.5]], dtype=np.float32),
        "infos": [],
        "rewards": np.array([1.0], dtype=np.float32),
        "new_obs": np.array([[0.0]], dtype=np.float32),
    }

    assert callback._on_step() is True
    assert callback.invalid_action_detected is False
    assert callback.invalid_action_reason is None


def test_immediate_collapse_reason_prefers_callback_failure_over_missing_training_metrics() -> None:
    callback = SimpleNamespace(
        invalid_action_detected=True,
        invalid_action_reason="non_finite_action",
        invalid_observation_detected=False,
        invalid_observation_reason=None,
    )

    assert _immediate_collapse_reason({}, callback) == "non_finite_action"


def test_immediate_collapse_reason_does_not_infer_policy_nan_from_missing_metrics() -> None:
    callback = SimpleNamespace(
        invalid_action_detected=False,
        invalid_action_reason=None,
        invalid_observation_detected=False,
        invalid_observation_reason=None,
    )

    assert _immediate_collapse_reason({}, callback) is None


def test_immediate_collapse_reason_does_not_infer_entropy_failure_when_metric_is_missing() -> None:
    callback = SimpleNamespace(
        invalid_action_detected=False,
        invalid_action_reason=None,
        invalid_observation_detected=False,
        invalid_observation_reason=None,
    )

    assert _immediate_collapse_reason({"entropy": None}, callback) is None


def test_is_completed_run_rejects_early_collapsed_runs(tmp_path: Path) -> None:
    for artifact in (
        "run_config.json",
        "run_metadata.json",
        "run_status.json",
        "metrics.csv",
        "updates.csv",
        "collapse.json",
        "latest.pt",
        "best_by_eval_mean.pt",
        "vecnormalize.pkl",
        "rng_state.pkl",
    ):
        path = tmp_path / artifact
        if path.suffix:
            path.write_text("placeholder", encoding="utf-8")

    save_json(
        tmp_path / "run_config.json",
        {"config": {"env": {"total_timesteps": 1000}}},
    )
    save_json(
        tmp_path / "run_metadata.json",
        {"config_hash": "abc"},
    )
    save_json(
        tmp_path / "run_status.json",
        {
            "status": "completed",
            "target_env_steps": 1000,
            "final_env_steps": 320,
            "collapse_flag": 1,
        },
    )
    save_json(
        tmp_path / "collapse.json",
        {
            "collapse_flag": 1,
            "collapse_reason": "non_finite_action",
        },
    )

    assert is_completed_run(tmp_path) is False


def test_suite_command_omits_resume_flag_when_resume_disabled() -> None:
    spec = build_run_spec(
        suite_name="main_benchmark",
        suite_kind="main_benchmark",
        algorithm="a2c",
        env_key="pendulum_v1",
        seed=0,
    )

    assert "--resume" in suite_command(spec, device="cpu", resume=True)
    assert "--resume" not in suite_command(spec, device="cpu", resume=False)


def test_run_suite_does_not_skip_invalid_run_even_if_manifest_lists_it_completed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "results" / "raw" / "main_benchmark" / "a2c" / "pendulum_v1" / "seed_0" / "default"
    run_dir.mkdir(parents=True, exist_ok=True)

    for artifact in (
        "run_config.json",
        "run_metadata.json",
        "run_status.json",
        "metrics.csv",
        "updates.csv",
        "collapse.json",
        "latest.pt",
        "best_by_eval_mean.pt",
        "vecnormalize.pkl",
        "rng_state.pkl",
    ):
        path = run_dir / artifact
        if path.suffix:
            path.write_text("placeholder", encoding="utf-8")

    save_json(run_dir / "run_config.json", {"config": {"env": {"total_timesteps": 1000}}})
    save_json(run_dir / "run_metadata.json", {"config_hash": "abc"})
    save_json(
        run_dir / "run_status.json",
        {"status": "completed", "target_env_steps": 1000, "final_env_steps": 320, "collapse_flag": 1},
    )
    save_json(run_dir / "collapse.json", {"collapse_flag": 1, "collapse_reason": "policy_loss_nan"})

    manifest_root = tmp_path / "results" / "manifests" / "main_benchmark"
    manifest_root.mkdir(parents=True, exist_ok=True)
    (manifest_root / "completed_runs.jsonl").write_text(
        '{"run_key":"a2c:pendulum_v1:seed_0:default"}\n',
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "src.utils.suite_runner.manifest_paths",
        lambda suite_name: {
            "root": manifest_root,
            "completed": manifest_root / "completed_runs.jsonl",
            "failed": manifest_root / "failed_runs.jsonl",
            "planned": manifest_root / "planned_runs.jsonl",
            "suite": manifest_root / "suite.json",
        },
    )

    def fake_create_training_job(*args, **kwargs):
        return SimpleNamespace(output_dir=run_dir, config={"env": {"total_timesteps": 1000}})

    monkeypatch.setattr("src.utils.suite_runner.create_training_job", fake_create_training_job)

    run_calls: list[list[str]] = []

    def fake_subprocess_run(command, stdout=None, stderr=None, check=False):
        run_calls.append(command)
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr("src.utils.suite_runner.subprocess.run", fake_subprocess_run)

    spec = build_run_spec(
        suite_name="main_benchmark",
        suite_kind="main_benchmark",
        algorithm="a2c",
        env_key="pendulum_v1",
        seed=0,
    )
    run_suite(
        suite_name="main_benchmark",
        suite_kind="main_benchmark",
        specs=[spec],
        device="cpu",
        resume=True,
    )

    assert len(run_calls) == 1


def test_reset_run_directory_preserves_active_suite_logs(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    nested_dir = run_dir / "tensorboard"
    nested_dir.mkdir(parents=True)
    preserved_stdout = run_dir / "stdout.log"
    preserved_stderr = run_dir / "stderr.log"
    preserved_stdout.write_text("keep", encoding="utf-8")
    preserved_stderr.write_text("keep", encoding="utf-8")
    (run_dir / "metrics.csv").write_text("remove", encoding="utf-8")
    (nested_dir / "events.out.tfevents").write_text("remove", encoding="utf-8")

    _reset_run_directory(run_dir, preserve_names={"stdout.log", "stderr.log"})

    assert preserved_stdout.exists()
    assert preserved_stderr.exists()
    assert not (run_dir / "metrics.csv").exists()
    assert not nested_dir.exists()
