from pathlib import Path

import pytest

from src.metrics.logging_schema import (
    COLLAPSE_FIELDS,
    METRICS_COLUMNS,
    UPDATES_COLUMNS,
    append_metrics_row,
    append_updates_row,
    validate_collapse_payload,
)


def _row(columns: list[str], value: float = 0.0) -> dict[str, float]:
    return {column: value for column in columns}


def test_metrics_and_updates_headers_are_written(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.csv"
    updates_path = tmp_path / "updates.csv"
    append_metrics_row(metrics_path, _row(METRICS_COLUMNS))
    append_updates_row(updates_path, _row(UPDATES_COLUMNS))
    assert metrics_path.read_text(encoding="utf-8").splitlines()[0].split(",") == METRICS_COLUMNS
    assert updates_path.read_text(encoding="utf-8").splitlines()[0].split(",") == UPDATES_COLUMNS


def test_collapse_payload_requires_all_fields() -> None:
    valid = {field: 0 for field in COLLAPSE_FIELDS}
    validate_collapse_payload(valid)
    with pytest.raises(ValueError):
        validate_collapse_payload({"collapse_flag": 0})
