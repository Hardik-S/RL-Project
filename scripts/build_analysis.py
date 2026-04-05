"""Build tables and figures from completed raw benchmark logs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.pipeline import build_analysis_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build aggregated analysis tables and figures.")
    parser.add_argument("--raw-root", type=Path, help="Optional override for the raw results root.")
    parser.add_argument("--aggregated-root", type=Path, help="Optional override for the aggregated output root.")
    args = parser.parse_args()

    summary = build_analysis_outputs(raw_root=args.raw_root, aggregated_root=args.aggregated_root)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
