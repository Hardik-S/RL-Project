# Validation Probe Framework

This framework is for the next agent to validate the benchmark one run at a time.

## Safety rule

Do not launch `run_main_benchmark.py` from this workflow.

Use exactly one probe per command invocation, then stop and ask the user whether to continue.

The runner in this framework does not loop across multiple tests.

## Entry point

Use:

```bash
python scripts/run_validation_probe.py --list
python scripts/run_validation_probe.py --next
python scripts/run_validation_probe.py --run <probe_id>
python scripts/run_validation_probe.py --show <probe_id>
```

Optional:

```bash
python scripts/run_validation_probe.py --run <probe_id> --resume
```

Default behavior without `--resume` is to rerun that single probe fresh.

## Ordered probe plan

1. `a2c_pendulum_s0`
2. `ppo_clip_pendulum_s0`
3. `ppo_kl_pendulum_s0`
4. `trpo_pendulum_s0`
5. `a2c_hopper_s0`
6. `ppo_clip_hopper_s0`
7. `ppo_kl_hopper_s0`
8. `trpo_hopper_s0`

Rationale:

- Pendulum first to validate the harness on the fastest environment.
- Hopper next because it previously showed early false failures and is the first meaningful locomotion stress test.
- Only after the Hopper probes are reviewed should the next agent consider Walker2d or HalfCheetah.

## What each run writes

Each probe uses its own suite name:

- `probe_a2c_pendulum_s0`
- `probe_ppo_clip_pendulum_s0`
- `probe_ppo_kl_pendulum_s0`
- `probe_trpo_pendulum_s0`
- `probe_a2c_hopper_s0`
- `probe_ppo_clip_hopper_s0`
- `probe_ppo_kl_hopper_s0`
- `probe_trpo_hopper_s0`

Run outputs go under:

```text
results/raw/<suite_name>/<algorithm>/<env_key>/seed_<seed>/default/
```

State is recorded in:

```text
results/manifests/validation_probes/probe_state.json
```

## Required operator behavior for the next agent

After every `--run` command:

1. Read the JSON summary printed by the script.
2. Inspect at least:
   - `completed_run`
   - `run_status`
   - `collapse`
   - final metrics row
3. Report the result to the user.
4. Ask whether to proceed to the next probe.
5. Do not run another probe until the user explicitly confirms.

## Minimum interpretation rules

- If `completed_run = true`, `collapse_flag = 0`, and `reached_target_timesteps = true`, treat that probe as evidence against the old universal-collapse conclusion.
- If a probe fails, inspect its `stderr.log` before making any algorithm-instability claim.
- Do not generalize from a single probe to the whole benchmark.
- Do not rebuild aggregated analysis from probe suites.

## Suggested next-agent prompt

Continue from `RL-Project/09_validation_probe_framework.md`.

Use `python scripts/run_validation_probe.py --next` to choose the next single probe, run exactly one probe, inspect its JSON summary plus raw artifacts, and then stop for user confirmation before running anything else.

