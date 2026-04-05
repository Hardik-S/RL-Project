"""Run exactly one validation probe and stop for human review."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.probe_framework import (
    default_probe_cases,
    get_probe_case,
    load_probe_state,
    next_pending_probe,
    probe_listing,
    save_probe_result,
    summarize_probe_run,
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def list_probes() -> None:
    payload = {
        "probes": probe_listing(),
        "next_pending_probe": next_pending_probe().probe_id if next_pending_probe() else None,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def run_probe(probe_id: str, *, device: str, resume: bool) -> None:
    case = get_probe_case(probe_id)
    command = [
        sys.executable,
        "scripts/train.py",
        "--algo",
        case.algorithm,
        "--env",
        case.env_key,
        "--seed",
        str(case.seed),
        "--suite-name",
        case.suite_name,
        "--run-tag",
        case.run_tag,
        "--device",
        device,
    ]
    if resume:
        command.append("--resume")

    completed = subprocess.run(command, check=False)
    summary = summarize_probe_run(case)
    summary["executed_at"] = utc_now_iso()
    summary["command"] = command
    summary["return_code"] = completed.returncode
    save_probe_result(probe_id, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(
        "\nStop here. Review the summary and ask the user for confirmation before executing the next probe."
    )


def show_probe(probe_id: str) -> None:
    case = get_probe_case(probe_id)
    state = load_probe_state().get("results", {})
    payload = {
        "probe": {
            "probe_id": case.probe_id,
            "algorithm": case.algorithm,
            "env_key": case.env_key,
            "seed": case.seed,
            "suite_name": case.suite_name,
            "description": case.description,
            "rationale": case.rationale,
        },
        "saved_result": state.get(probe_id),
        "current_run_summary": summarize_probe_run(case),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one benchmark validation probe at a time. This script never loops over multiple probes."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="List the ordered probe plan and current state.")
    group.add_argument("--next", action="store_true", help="Print the next recommended pending probe.")
    group.add_argument("--run", dest="run_probe_id", help="Execute exactly one probe by probe_id.")
    group.add_argument("--show", dest="show_probe_id", help="Show saved and current summary for one probe.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true", help="Resume an existing probe run instead of replacing it.")
    args = parser.parse_args()

    if args.list:
        list_probes()
        return
    if args.next:
        case = next_pending_probe()
        payload = None
        if case is not None:
            payload = {
                "probe_id": case.probe_id,
                "algorithm": case.algorithm,
                "env_key": case.env_key,
                "seed": case.seed,
                "suite_name": case.suite_name,
                "description": case.description,
                "rationale": case.rationale,
            }
        print(json.dumps({"next_pending_probe": payload}, indent=2, sort_keys=True))
        return
    if args.run_probe_id:
        run_probe(args.run_probe_id, device=args.device, resume=args.resume)
        return
    if args.show_probe_id:
        show_probe(args.show_probe_id)
        return

    raise RuntimeError("No action selected.")


if __name__ == "__main__":
    main()

