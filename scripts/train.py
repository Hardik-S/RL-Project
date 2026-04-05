"""Train a single benchmark run."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.algos.common import (
    ALGO_MODULES,
    common_arg_parser,
    create_training_job,
    job_to_json,
    overrides_from_args,
)


def main() -> None:
    parser = common_arg_parser("Train one algorithm/environment/seed run.")
    args = parser.parse_args()
    job = create_training_job(
        args.algo,
        args.env_key,
        args.seed,
        suite_name=args.suite_name,
        run_tag=args.run_tag,
        device=args.device,
        dry_run=args.dry_run,
        resume=args.resume,
        overrides=overrides_from_args(args),
    )

    if args.dry_run:
        print(job_to_json(job))
        return

    module = importlib.import_module(ALGO_MODULES[job.algorithm])
    module.run(job)


if __name__ == "__main__":
    main()
