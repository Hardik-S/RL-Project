"""Evaluate a saved run directory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.algos.common import evaluate_saved_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved run directory.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"run directory does not exist: {run_dir}")
    result = evaluate_saved_run(run_dir, device=args.device)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
