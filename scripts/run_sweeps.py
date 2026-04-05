"""Launch or print resumable one-factor-at-a-time sweeps."""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.algos.common import load_run_config
from src.utils.suite_runner import build_run_spec, run_suite


def build_sweep_specs(env_key: str, suite_name: str) -> list:
    base = load_run_config("a2c", env_key, 0)
    seeds = base["benchmark"]["sweep_seeds"]
    sweeps = base["sweeps"]
    specs = []

    for seed in seeds:
        for actor_lr in sweeps["a2c"]["actor_lr"]:
            specs.append(
                build_run_spec(
                    suite_name=suite_name,
                    suite_kind="sensitivity_sweep",
                    algorithm="a2c",
                    env_key=env_key,
                    seed=seed,
                    tag_name="actor_lr",
                    tag_value=actor_lr,
                    cli_overrides=["--actor-lr", str(actor_lr)],
                    override_values={"actor_lr": actor_lr},
                )
            )
        for actor_lr in sweeps["ppo_clip"]["actor_lr"]:
            specs.append(
                build_run_spec(
                    suite_name=suite_name,
                    suite_kind="sensitivity_sweep",
                    algorithm="ppo_clip",
                    env_key=env_key,
                    seed=seed,
                    tag_name="actor_lr",
                    tag_value=actor_lr,
                    cli_overrides=["--actor-lr", str(actor_lr)],
                    override_values={"actor_lr": actor_lr},
                )
            )
        for clip_epsilon in sweeps["ppo_clip"]["clip_epsilon"]:
            specs.append(
                build_run_spec(
                    suite_name=suite_name,
                    suite_kind="sensitivity_sweep",
                    algorithm="ppo_clip",
                    env_key=env_key,
                    seed=seed,
                    tag_name="clip_epsilon",
                    tag_value=clip_epsilon,
                    cli_overrides=["--clip-epsilon", str(clip_epsilon)],
                    override_values={"clip_epsilon": clip_epsilon},
                )
            )
        for actor_lr in sweeps["ppo_kl"]["actor_lr"]:
            specs.append(
                build_run_spec(
                    suite_name=suite_name,
                    suite_kind="sensitivity_sweep",
                    algorithm="ppo_kl",
                    env_key=env_key,
                    seed=seed,
                    tag_name="actor_lr",
                    tag_value=actor_lr,
                    cli_overrides=["--actor-lr", str(actor_lr)],
                    override_values={"actor_lr": actor_lr},
                )
            )
        for target_kl in sweeps["ppo_kl"]["target_kl"]:
            specs.append(
                build_run_spec(
                    suite_name=suite_name,
                    suite_kind="sensitivity_sweep",
                    algorithm="ppo_kl",
                    env_key=env_key,
                    seed=seed,
                    tag_name="target_kl",
                    tag_value=target_kl,
                    cli_overrides=["--target-kl", str(target_kl)],
                    override_values={"target_kl": target_kl},
                )
            )
        for max_kl_delta in sweeps["trpo"]["max_kl_delta"]:
            specs.append(
                build_run_spec(
                    suite_name=suite_name,
                    suite_kind="sensitivity_sweep",
                    algorithm="trpo",
                    env_key=env_key,
                    seed=seed,
                    tag_name="max_kl_delta",
                    tag_value=max_kl_delta,
                    cli_overrides=["--max-kl-delta", str(max_kl_delta)],
                    override_values={"max_kl_delta": max_kl_delta},
                )
            )
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen sensitivity sweeps.")
    parser.add_argument("--env", dest="env_key", required=True, choices=["hopper_v4", "halfcheetah_v4"])
    parser.add_argument("--suite-name")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()
    suite_name = args.suite_name or f"sweep_{args.env_key}"

    commands = run_suite(
        suite_name=suite_name,
        suite_kind="sensitivity_sweep",
        specs=build_sweep_specs(args.env_key, suite_name),
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
