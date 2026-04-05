"""Canonical output path helpers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def results_root() -> Path:
    return repo_root() / "results"


def raw_results_root() -> Path:
    return results_root() / "raw"


def manifests_root() -> Path:
    return results_root() / "manifests"


def aggregated_results_root() -> Path:
    return results_root() / "aggregated"


def slugify(value: str) -> str:
    slug = value.strip().lower().replace(".", "p")
    slug = re.sub(r"[^a-z0-9_+=-]+", "_", slug)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "default"


def format_override_value(value: Any) -> str:
    if isinstance(value, float):
        return format(value, "g")
    return str(value)


def variant_tag(name: str | None = None, value: Any | None = None) -> str:
    if not name:
        return "default"
    if value is None:
        return slugify(name)
    return slugify(f"{name}_{format_override_value(value)}")


def suite_raw_root(suite_name: str) -> Path:
    return raw_results_root() / slugify(suite_name)


def suite_manifest_dir(suite_name: str) -> Path:
    return manifests_root() / slugify(suite_name)


def run_dir(
    algorithm: str,
    env_key: str,
    seed: int,
    *,
    suite_name: str | None = None,
    run_tag: str | None = None,
) -> Path:
    base = raw_results_root() if suite_name is None else suite_raw_root(suite_name)
    path = base / algorithm / env_key / f"seed_{seed}"
    if suite_name is not None:
        path = path / (run_tag or "default")
    return path


def aggregated_tables_dir() -> Path:
    return aggregated_results_root() / "tables"


def aggregated_figures_dir() -> Path:
    return aggregated_results_root() / "figures"


def aggregated_summaries_dir() -> Path:
    return aggregated_results_root() / "summaries"
