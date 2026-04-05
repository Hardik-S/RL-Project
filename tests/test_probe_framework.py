from pathlib import Path

from src.utils.probe_framework import (
    default_probe_cases,
    get_probe_case,
    next_pending_probe,
    save_probe_result,
    summarize_probe_run,
)
from src.utils.serialization import save_json


def test_probe_ids_are_unique() -> None:
    probe_ids = [case.probe_id for case in default_probe_cases()]
    assert len(probe_ids) == len(set(probe_ids))


def test_summarize_probe_run_reports_completed_status(tmp_path: Path) -> None:
    case = get_probe_case("a2c_pendulum_s0")
    run_dir = tmp_path / "results" / "raw" / case.suite_name / case.algorithm / case.env_key / "seed_0" / "default"
    run_dir.mkdir(parents=True)

    (run_dir / "metrics.csv").write_text("env_steps,entropy,nan_or_divergence_flag\n100,0.1,0\n", encoding="utf-8")
    (run_dir / "updates.csv").write_text("unstable_update_flag\n0\n", encoding="utf-8")
    (run_dir / "latest.pt").write_text("x", encoding="utf-8")
    (run_dir / "best_by_eval_mean.pt").write_text("x", encoding="utf-8")
    (run_dir / "vecnormalize.pkl").write_text("x", encoding="utf-8")
    (run_dir / "rng_state.pkl").write_text("x", encoding="utf-8")
    save_json(run_dir / "run_metadata.json", {"config_hash": "abc"})
    save_json(run_dir / "run_config.json", {"config": {"env": {"total_timesteps": 100}}})
    save_json(
        run_dir / "run_status.json",
        {
            "status": "completed",
            "target_env_steps": 100,
            "final_env_steps": 100,
            "collapse_flag": 0,
        },
    )
    save_json(run_dir / "collapse.json", {"collapse_flag": 0, "collapse_reason": None})

    summary = summarize_probe_run(case, run_path=run_dir)

    assert summary["completed_run"] is True
    assert summary["metrics"]["rows"] == 1
    assert summary["updates"]["unstable_updates_total"] == 0


def test_next_pending_probe_skips_completed_results(tmp_path: Path, monkeypatch) -> None:
    state_path = tmp_path / "probe_state.json"
    monkeypatch.setattr("src.utils.probe_framework.probe_state_path", lambda: state_path)

    first = default_probe_cases()[0]
    second = default_probe_cases()[1]
    save_probe_result(first.probe_id, {"completed_run": True})

    assert next_pending_probe().probe_id == second.probe_id
