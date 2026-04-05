"""One-at-a-time validation probe planning and result summaries."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.paths import manifests_root, run_dir, slugify
from src.utils.serialization import load_json, save_json
from src.utils.suite_runner import is_completed_run


@dataclass(frozen=True)
class ProbeCase:
    probe_id: str
    algorithm: str
    env_key: str
    seed: int
    description: str
    rationale: str

    @property
    def suite_name(self) -> str:
        return f"probe_{self.probe_id}"

    @property
    def run_tag(self) -> str:
        return "default"

    @property
    def output_dir(self) -> Path:
        return run_dir(
            self.algorithm,
            self.env_key,
            self.seed,
            suite_name=self.suite_name,
            run_tag=self.run_tag,
        )


def default_probe_cases() -> list[ProbeCase]:
    return [
        ProbeCase(
            probe_id="a2c_pendulum_s0",
            algorithm="a2c",
            env_key="pendulum_v1",
            seed=0,
            description="Baseline sanity check on the fastest environment.",
            rationale="Confirms the common harness and A2C path can complete a full Pendulum run.",
        ),
        ProbeCase(
            probe_id="ppo_clip_pendulum_s0",
            algorithm="ppo_clip",
            env_key="pendulum_v1",
            seed=0,
            description="Pendulum check for the clipped PPO branch.",
            rationale="Verifies PPO-Clip does not reproduce the old universal collapse pattern.",
        ),
        ProbeCase(
            probe_id="ppo_kl_pendulum_s0",
            algorithm="ppo_kl",
            env_key="pendulum_v1",
            seed=0,
            description="Pendulum check for the target-KL PPO branch.",
            rationale="Validates the PPO-KL instrumentation and immediate-collapse labeling path.",
        ),
        ProbeCase(
            probe_id="trpo_pendulum_s0",
            algorithm="trpo",
            env_key="pendulum_v1",
            seed=0,
            description="Pendulum check for TRPO.",
            rationale="Exercises the trust-region code path and entropy fallback instrumentation.",
        ),
        ProbeCase(
            probe_id="a2c_hopper_s0",
            algorithm="a2c",
            env_key="hopper_v4",
            seed=0,
            description="First locomotion probe on the baseline method.",
            rationale="Checks whether the old near-immediate Hopper failures were harness artifacts.",
        ),
        ProbeCase(
            probe_id="ppo_clip_hopper_s0",
            algorithm="ppo_clip",
            env_key="hopper_v4",
            seed=0,
            description="Locomotion probe for PPO-Clip.",
            rationale="Tests whether PPO-Clip remains stable on a harder environment after the harness fixes.",
        ),
        ProbeCase(
            probe_id="ppo_kl_hopper_s0",
            algorithm="ppo_kl",
            env_key="hopper_v4",
            seed=0,
            description="Locomotion probe for PPO-KL.",
            rationale="Tests whether the KL-controlled PPO path shows real instability or clean completion on Hopper.",
        ),
        ProbeCase(
            probe_id="trpo_hopper_s0",
            algorithm="trpo",
            env_key="hopper_v4",
            seed=0,
            description="Locomotion probe for TRPO.",
            rationale="Checks the trust-region path on the first environment that used to fail at very low step counts.",
        ),
    ]


def probe_case_map() -> dict[str, ProbeCase]:
    return {case.probe_id: case for case in default_probe_cases()}


def get_probe_case(probe_id: str) -> ProbeCase:
    try:
        return probe_case_map()[probe_id]
    except KeyError as exc:
        known = ", ".join(case.probe_id for case in default_probe_cases())
        raise KeyError(f"Unknown probe_id '{probe_id}'. Known probe ids: {known}") from exc


def probe_manifest_root() -> Path:
    return manifests_root() / "validation_probes"


def probe_state_path() -> Path:
    return probe_manifest_root() / "probe_state.json"


def load_probe_state() -> dict[str, Any]:
    path = probe_state_path()
    if not path.exists():
        return {"results": {}}
    return load_json(path)


def save_probe_result(probe_id: str, payload: dict[str, Any]) -> None:
    state = load_probe_state()
    results = dict(state.get("results", {}))
    results[probe_id] = payload
    save_json(probe_state_path(), {"results": results})


def summarize_probe_run(case: ProbeCase, *, run_path: Path | None = None) -> dict[str, Any]:
    run_path = run_path or case.output_dir
    summary: dict[str, Any] = {
        "probe_id": case.probe_id,
        "algorithm": case.algorithm,
        "env_key": case.env_key,
        "seed": case.seed,
        "suite_name": case.suite_name,
        "run_dir": str(run_path),
        "artifacts_present": sorted(path.name for path in run_path.iterdir()) if run_path.exists() else [],
        "completed_run": False,
        "run_status": None,
        "collapse": None,
        "metrics": None,
    }
    if not run_path.exists():
        return summary

    run_status_path = run_path / "run_status.json"
    collapse_path = run_path / "collapse.json"
    metrics_path = run_path / "metrics.csv"
    updates_path = run_path / "updates.csv"

    if run_status_path.exists():
        summary["run_status"] = load_json(run_status_path)
    if collapse_path.exists():
        summary["collapse"] = load_json(collapse_path)
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        metric_summary = {
            "rows": int(metrics.shape[0]),
        }
        if not metrics.empty:
            final_row = metrics.iloc[-1].to_dict()
            metric_summary["final_row"] = final_row
        if "entropy" in metrics:
            metric_summary["entropy_nan_rows"] = int(metrics["entropy"].isna().sum())
        if "nan_or_divergence_flag" in metrics:
            metric_summary["nan_or_divergence_flag_max"] = int(metrics["nan_or_divergence_flag"].fillna(0).max())
        summary["metrics"] = metric_summary
    if updates_path.exists():
        updates = pd.read_csv(updates_path)
        update_summary = {"rows": int(updates.shape[0])}
        if "unstable_update_flag" in updates:
            update_summary["unstable_updates_total"] = int(updates["unstable_update_flag"].fillna(0).sum())
        summary["updates"] = update_summary

    summary["completed_run"] = is_completed_run(run_path)
    return summary


def next_pending_probe() -> ProbeCase | None:
    state = load_probe_state().get("results", {})
    for case in default_probe_cases():
        result = state.get(case.probe_id)
        if not result or not result.get("completed_run", False):
            return case
    return None


def probe_listing() -> list[dict[str, Any]]:
    state = load_probe_state().get("results", {})
    listing: list[dict[str, Any]] = []
    for case in default_probe_cases():
        result = state.get(case.probe_id, {})
        listing.append(
            {
                "probe_id": case.probe_id,
                "algorithm": case.algorithm,
                "env_key": case.env_key,
                "seed": case.seed,
                "suite_name": case.suite_name,
                "description": case.description,
                "status": "completed" if result.get("completed_run", False) else "pending",
            }
        )
    return listing


def state_slug_for_probe(probe_id: str) -> str:
    return slugify(probe_id)
