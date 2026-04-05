"""Resumable local suite runner for smoke tests, main benchmarks, and sweeps."""

from __future__ import annotations

import json
import subprocess
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.algos.common import create_training_job
from src.utils.manifests import (
    append_jsonl,
    load_suite_metadata,
    manifest_paths,
    run_key,
    save_suite_metadata,
    stable_config_hash,
    utc_now_iso,
    write_planned_manifest,
)
from src.utils.paths import variant_tag
from src.utils.serialization import load_json, save_json

REQUIRED_RUN_ARTIFACTS = [
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
]


@dataclass(frozen=True)
class SuiteRunSpec:
    suite_name: str
    suite_kind: str
    algorithm: str
    env_key: str
    seed: int
    run_tag: str
    cli_overrides: list[str]
    override_values: dict[str, Any]


def build_run_spec(
    *,
    suite_name: str,
    suite_kind: str,
    algorithm: str,
    env_key: str,
    seed: int,
    tag_name: str | None = None,
    tag_value: Any | None = None,
    cli_overrides: list[str] | None = None,
    override_values: dict[str, Any] | None = None,
) -> SuiteRunSpec:
    return SuiteRunSpec(
        suite_name=suite_name,
        suite_kind=suite_kind,
        algorithm=algorithm,
        env_key=env_key,
        seed=seed,
        run_tag=variant_tag(tag_name, tag_value),
        cli_overrides=cli_overrides or [],
        override_values=override_values or {},
    )


def _run_status_payload(run_dir: Path) -> dict[str, Any]:
    status_path = run_dir / "run_status.json"
    if not status_path.exists():
        return {}
    return load_json(status_path)


def is_completed_run(run_dir: Path) -> bool:
    status = _run_status_payload(run_dir)
    if status.get("status") != "completed":
        return False
    if not all((run_dir / artifact).exists() for artifact in REQUIRED_RUN_ARTIFACTS):
        return False
    target_env_steps = status.get("target_env_steps")
    final_env_steps = status.get("final_env_steps")
    if target_env_steps is None or final_env_steps is None:
        run_config_path = run_dir / "run_config.json"
        if not run_config_path.exists():
            return False
        run_config = load_json(run_config_path)
        target_env_steps = run_config["config"]["env"]["total_timesteps"]
    if int(final_env_steps) < int(target_env_steps):
        return False
    if int(status.get("collapse_flag", 0)):
        return False
    collapse_path = run_dir / "collapse.json"
    if collapse_path.exists() and int(load_json(collapse_path).get("collapse_flag", 0)):
        return False
    return True


def _failure_metadata(run_dir: Path) -> dict[str, Any]:
    failure_path = run_dir / "error.json"
    if not failure_path.exists():
        return {}
    return load_json(failure_path)


def planned_manifest_row(spec: SuiteRunSpec, *, device: str) -> dict[str, Any]:
    job = create_training_job(
        spec.algorithm,
        spec.env_key,
        spec.seed,
        device=device,
        dry_run=True,
        overrides=spec.override_values,
        suite_name=spec.suite_name,
        run_tag=spec.run_tag,
    )
    return {
        "suite_name": spec.suite_name,
        "suite_kind": spec.suite_kind,
        "run_key": run_key(
            algorithm=spec.algorithm,
            env_key=spec.env_key,
            seed=spec.seed,
            run_tag=spec.run_tag,
        ),
        "algorithm": spec.algorithm,
        "env_key": spec.env_key,
        "seed": spec.seed,
        "run_tag": spec.run_tag,
        "output_dir": str(job.output_dir),
        "config_hash": stable_config_hash(job.config),
        "override_values": spec.override_values,
    }


def suite_command(spec: SuiteRunSpec, *, device: str, resume: bool) -> list[str]:
    command = [
        sys.executable,
        "scripts/train.py",
        "--algo",
        spec.algorithm,
        "--env",
        spec.env_key,
        "--seed",
        str(spec.seed),
        "--suite-name",
        spec.suite_name,
        "--run-tag",
        spec.run_tag,
        "--device",
        device,
    ]
    if resume:
        command.append("--resume")
    command.extend(spec.cli_overrides)
    return command


def _append_completed_manifest(spec: SuiteRunSpec, run_dir: Path, *, device: str, command: list[str]) -> None:
    paths = manifest_paths(spec.suite_name)
    run_metadata = load_json(run_dir / "run_metadata.json")
    run_status = load_json(run_dir / "run_status.json")
    append_jsonl(
        paths["completed"],
        {
            "timestamp": utc_now_iso(),
            "suite_name": spec.suite_name,
            "suite_kind": spec.suite_kind,
            "run_key": run_key(
                algorithm=spec.algorithm,
                env_key=spec.env_key,
                seed=spec.seed,
                run_tag=spec.run_tag,
            ),
            "algorithm": spec.algorithm,
            "env_key": spec.env_key,
            "seed": spec.seed,
            "run_tag": spec.run_tag,
            "device": device,
            "output_dir": str(run_dir),
            "stdout_log": str(run_dir / "stdout.log"),
            "stderr_log": str(run_dir / "stderr.log"),
            "config_hash": run_metadata["config_hash"],
            "command": command,
            "completed_at": run_status.get("completed_at"),
        },
    )


def _append_failed_manifest(
    spec: SuiteRunSpec,
    run_dir: Path,
    *,
    device: str,
    command: list[str],
    return_code: int | None,
    error_message: str,
    traceback_text: str | None = None,
) -> None:
    paths = manifest_paths(spec.suite_name)
    failure = {
        "timestamp": utc_now_iso(),
        "suite_name": spec.suite_name,
        "suite_kind": spec.suite_kind,
        "run_key": run_key(
            algorithm=spec.algorithm,
            env_key=spec.env_key,
            seed=spec.seed,
            run_tag=spec.run_tag,
        ),
        "algorithm": spec.algorithm,
        "env_key": spec.env_key,
        "seed": spec.seed,
        "run_tag": spec.run_tag,
        "device": device,
        "output_dir": str(run_dir),
        "stdout_log": str(run_dir / "stdout.log"),
        "stderr_log": str(run_dir / "stderr.log"),
        "command": command,
        "return_code": return_code,
        "error_message": error_message,
    }
    if traceback_text:
        failure["traceback"] = traceback_text
    append_jsonl(paths["failed"], failure)


def run_suite(
    *,
    suite_name: str,
    suite_kind: str,
    specs: list[SuiteRunSpec],
    device: str = "auto",
    dry_run: bool = False,
    resume: bool = True,
    stop_on_error: bool = False,
) -> list[list[str]]:
    paths = manifest_paths(suite_name)
    prior_suite_metadata = load_suite_metadata(suite_name)
    save_suite_metadata(
        suite_name,
        {
            "suite_name": suite_name,
            "suite_kind": suite_kind,
            "created_at": prior_suite_metadata.get("created_at", utc_now_iso()),
            "last_invoked_at": utc_now_iso(),
            "device": device,
            "resume": resume,
            "planned_run_count": len(specs),
        },
    )
    write_planned_manifest(
        suite_name,
        [planned_manifest_row(spec, device=device) for spec in specs],
    )

    commands = [suite_command(spec, device=device, resume=resume) for spec in specs]
    if dry_run:
        return commands

    for spec, command in zip(specs, commands):
        job = create_training_job(
            spec.algorithm,
            spec.env_key,
            spec.seed,
            device=device,
            dry_run=False,
            resume=resume,
            overrides=spec.override_values,
            suite_name=spec.suite_name,
            run_tag=spec.run_tag,
        )
        run_dir = job.output_dir
        if resume and is_completed_run(run_dir):
            continue

        run_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = run_dir / "stdout.log"
        stderr_path = run_dir / "stderr.log"
        try:
            with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open("w", encoding="utf-8") as stderr_handle:
                completed = subprocess.run(command, stdout=stdout_handle, stderr=stderr_handle, check=False)
            if completed.returncode == 0 and is_completed_run(run_dir):
                _append_completed_manifest(spec, run_dir, device=device, command=command)
                continue

            failure = _failure_metadata(run_dir)
            error_message = failure.get("error_message") or f"training command exited with code {completed.returncode}"
            _append_failed_manifest(
                spec,
                run_dir,
                device=device,
                command=command,
                return_code=completed.returncode,
                error_message=error_message,
            )
            if stop_on_error:
                raise RuntimeError(error_message)
        except Exception as exc:
            traceback_text = traceback.format_exc()
            error_payload = {
                "status": "failed",
                "failed_at": utc_now_iso(),
                "error_message": str(exc),
                "traceback": traceback_text,
            }
            save_json(run_dir / "error.json", error_payload)
            save_json(run_dir / "run_status.json", error_payload)
            _append_failed_manifest(
                spec,
                run_dir,
                device=device,
                command=command,
                return_code=None,
                error_message=str(exc),
                traceback_text=traceback_text,
            )
            if stop_on_error:
                raise
    return commands
