"""Launch or print the resumable main benchmark suite."""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.algos.common import load_run_config
from src.utils.suite_runner import build_run_spec, run_suite


def build_specs(suite_name: str) -> list:
    base = load_run_config("a2c", "pendulum_v1", 0)
    algorithms = base["benchmark"]["algorithms"]
    environments = base["benchmark"]["environments"]
    seeds = base["benchmark"]["main_seeds"]
    specs = []
    for algorithm in algorithms:
        for env_key in environments:
            for seed in seeds:
                specs.append(
                    build_run_spec(
                        suite_name=suite_name,
                        suite_kind="main_benchmark",
                        algorithm=algorithm,
                        env_key=env_key,
                        seed=seed,
                    )
                )
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen main benchmark matrix.")
    parser.add_argument("--suite-name", default="main_benchmark")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()

    commands = run_suite(
        suite_name=args.suite_name,
        suite_kind="main_benchmark",
        specs=build_specs(args.suite_name),
        device=args.device,
        dry_run=args.dry_run,
        resume=not args.no_resume,
        stop_on_error=args.stop_on_error,
    )
    if args.dry_run:
        for command in commands:
            print(shlex.join(command))


if __name__ == "__main__":
    main()
