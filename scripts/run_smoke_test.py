"""Launch or print a small resumable smoke-test suite."""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.suite_runner import build_run_spec, run_suite


def build_smoke_specs(suite_name: str, total_timesteps: int, eval_every: int, eval_episodes: int) -> list:
    base_overrides = {
        "total_timesteps": total_timesteps,
        "eval_every": eval_every,
        "eval_episodes": eval_episodes,
    }
    cli_overrides = [
        "--total-timesteps",
        str(total_timesteps),
        "--eval-every",
        str(eval_every),
        "--eval-episodes",
        str(eval_episodes),
    ]
    return [
        build_run_spec(suite_name=suite_name, suite_kind="smoke_test", algorithm="a2c", env_key="pendulum_v1", seed=0, cli_overrides=cli_overrides, override_values=base_overrides),
        build_run_spec(suite_name=suite_name, suite_kind="smoke_test", algorithm="ppo_clip", env_key="pendulum_v1", seed=0, cli_overrides=cli_overrides, override_values=base_overrides),
        build_run_spec(suite_name=suite_name, suite_kind="smoke_test", algorithm="ppo_kl", env_key="pendulum_v1", seed=0, cli_overrides=cli_overrides, override_values=base_overrides),
        build_run_spec(suite_name=suite_name, suite_kind="smoke_test", algorithm="trpo", env_key="pendulum_v1", seed=0, cli_overrides=cli_overrides, override_values=base_overrides),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a resumable Pendulum smoke suite.")
    parser.add_argument("--suite-name", default="smoke_pendulum")
    parser.add_argument("--total-timesteps", type=int, default=20_000)
    parser.add_argument("--eval-every", type=int, default=5_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()
    commands = run_suite(
        suite_name=args.suite_name,
        suite_kind="smoke_test",
        specs=build_smoke_specs(args.suite_name, args.total_timesteps, args.eval_every, args.eval_episodes),
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
