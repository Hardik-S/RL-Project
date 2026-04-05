"""CSV-writing helpers for benchmark callbacks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.metrics.logging_schema import append_metrics_row, append_updates_row


@dataclass(frozen=True)
class MetricsWriters:
    metrics_path: Path
    updates_path: Path

    def log_checkpoint(self, row: dict[str, Any]) -> None:
        append_metrics_row(self.metrics_path, row)

    def log_update(self, row: dict[str, Any]) -> None:
        append_updates_row(self.updates_path, row)
