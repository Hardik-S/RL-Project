"""Manifest and suite-run bookkeeping helpers."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from src.utils.paths import suite_manifest_dir
from src.utils.serialization import load_json, save_json

COMPLETED_MANIFEST = "completed_runs.jsonl"
FAILED_MANIFEST = "failed_runs.jsonl"
PLANNED_MANIFEST = "planned_runs.jsonl"
SUITE_METADATA = "suite.json"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def stable_config_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def manifest_paths(suite_name: str) -> dict[str, Path]:
    root = suite_manifest_dir(suite_name)
    root.mkdir(parents=True, exist_ok=True)
    return {
        "root": root,
        "completed": root / COMPLETED_MANIFEST,
        "failed": root / FAILED_MANIFEST,
        "planned": root / PLANNED_MANIFEST,
        "suite": root / SUITE_METADATA,
    }


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_suite_metadata(suite_name: str) -> dict[str, Any]:
    paths = manifest_paths(suite_name)
    if not paths["suite"].exists():
        return {}
    return load_json(paths["suite"])


def save_suite_metadata(suite_name: str, payload: dict[str, Any]) -> None:
    paths = manifest_paths(suite_name)
    save_json(paths["suite"], payload)


def write_planned_manifest(suite_name: str, rows: Iterable[dict[str, Any]]) -> None:
    paths = manifest_paths(suite_name)
    rows = list(rows)
    with paths["planned"].open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def run_key(*, algorithm: str, env_key: str, seed: int, run_tag: str) -> str:
    return f"{algorithm}:{env_key}:seed_{seed}:{run_tag}"
