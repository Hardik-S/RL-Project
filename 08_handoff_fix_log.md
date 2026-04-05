# Fix Log and Handoff

Date: 2026-04-03

## Scope completed

This pass implemented the P0/P1 benchmark-harness fixes identified after the invalid `main_benchmark` run.

Files changed:

- `src/callbacks/runtime_checks.py`
- `src/algos/common.py`
- `src/utils/suite_runner.py`
- `src/algos/_sb3_helpers.py`
- `src/algos/trpo_runner.py`
- `tests/test_runtime_and_suite_status.py`

## What was fixed

### 1. False rollout stop on raw SB3 actions

In `src/callbacks/runtime_checks.py`, the callback no longer aborts training when finite raw `actions` exceed env bounds.

Reason:
- SB3 callback-visible actions can be raw policy samples rather than env-side clipped actions.
- The old bound check produced false `out_of_bounds_action` failures.

Current behavior:
- The callback still stops on non-finite actions, rewards, or observations.
- It no longer treats merely out-of-range finite actions as invalid.

### 2. Collapse-reason attribution

In `src/algos/common.py`:

- Added `_metric_is_non_finite(...)`
- Added `_immediate_collapse_reason(...)`

Reason:
- Empty `last_training_info` was being interpreted as `policy_loss_nan`.
- That mislabeled pre-training or interrupted rollouts as optimizer collapse.

Current behavior:
- Callback failures are reported directly.
- Missing metrics do not automatically become `policy_loss_nan`.
- Final run status now includes:
  - `target_env_steps`
  - `final_env_steps`
  - `reached_target_timesteps`
  - `collapse_reason`
- Terminal status is now:
  - `completed` if target timesteps were reached and `collapse_flag == 0`
  - `collapsed` otherwise

### 3. Suite completion semantics

In `src/utils/suite_runner.py`, `is_completed_run(...)` is stricter.

A run now counts as completed only if:
- `run_status.json` says `completed`
- required artifacts exist
- `final_env_steps >= target_env_steps`
- `collapse_flag == 0`
- `collapse.json` does not report collapse

Reason:
- Previously, early-stop collapsed runs could still be counted as completed.

### 4. TRPO entropy instrumentation

In `src/algos/_sb3_helpers.py` and `src/algos/trpo_runner.py`:

- Added `policy_entropy_mean` to rollout diagnostics
- TRPO now falls back to that value if `train/entropy_loss` is missing from logger output

Reason:
- TRPO Pendulum smoke initially collapsed with `non_finite_entropy`
- This was instrumentation failure, not proven optimizer instability

## Tests added

Added `tests/test_runtime_and_suite_status.py` covering:

- finite out-of-range raw actions do not trigger invalid-action failure
- callback failure reason is preferred over missing training metrics
- missing metrics do not infer `policy_loss_nan`
- missing entropy metric does not infer entropy failure
- early collapsed runs are not counted as completed

## Validation performed

### Pytest

Executed:

```bash
python -m pytest tests\test_runtime_and_suite_status.py tests\test_analysis_pipeline.py tests\test_paths_and_manifests.py
```

Result:
- passed

### Targeted real runs

Executed:

```bash
python scripts/train.py --algo ppo_kl --env pendulum_v1 --seed 0 --suite-name smoke_single_fixcheck --run-tag ppo_kl --total-timesteps 20000 --eval-every 5000 --eval-episodes 5 --device cpu --resume
python scripts/run_smoke_test.py --suite-name smoke_pendulum_fixcheck --total-timesteps 20000 --eval-every 5000 --eval-episodes 5 --device cpu
python scripts/train.py --algo trpo --env pendulum_v1 --seed 0 --suite-name smoke_trpo_fixcheck --run-tag default --total-timesteps 20000 --eval-every 5000 --eval-episodes 5 --device cpu --resume
python scripts/run_smoke_test.py --suite-name smoke_pendulum_fixcheck --total-timesteps 20000 --eval-every 5000 --eval-episodes 5 --device cpu
python scripts/train.py --algo ppo_kl --env hopper_v4 --seed 0 --suite-name smoke_hopper_fixcheck --run-tag ppo_kl --total-timesteps 10000 --eval-every 5000 --eval-episodes 3 --device cpu --resume
```

Observed results:

- `smoke_single_fixcheck/ppo_kl/pendulum_v1`: completed, reached budget, no collapse
- `smoke_pendulum_fixcheck`: all four algorithms completed and reached budget
  - `a2c`
  - `ppo_clip`
  - `ppo_kl`
  - `trpo`
- `smoke_hopper_fixcheck/ppo_kl/hopper_v4`: completed, reached budget, no collapse

## Current interpretation

The original universal `policy_loss_nan` benchmark failure was primarily harness/instrumentation error.

At this point:
- the false positive runtime guard is fixed
- mislabeling of interrupted runs is fixed
- false suite completion is fixed
- TRPO missing-entropy instrumentation is fixed

This does **not** yet prove the full benchmark is scientifically valid.
It does mean the next rerun should measure actual training behavior rather than harness artifacts.

## Next agent instructions

Priority order:

1. Rerun the full main benchmark:

```bash
python scripts/run_main_benchmark.py --suite-name main_benchmark --device auto
```

2. Inspect status before trusting outputs:
- verify all expected runs exist
- verify how many are `completed` vs `collapsed`
- verify how many reached target timesteps
- check whether any collapse reasons remain and whether they cluster by algorithm/env

3. Rebuild analysis only after the rerun completes:

```bash
python scripts/build_analysis.py
```

4. If the rerun is clean enough to interpret, update audit notes and result summaries.

5. If real collapse remains:
- identify whether it is algorithm-specific or environment-specific
- inspect the first failing run's `run_status.json`, `collapse.json`, `metrics.csv`, and `updates.csv`
- do not assume it is another harness bug without reproducing it on a minimal run

## Important cautions for next agent

- Do not reuse the old invalidity conclusion from the earlier 80/80 failed benchmark as if it still reflects current code.
- Recompute all benchmark claims from the rerun.
- Trust `run_status.json` plus `collapse.json`, not shell return codes alone.
- The smoke runner only covers Pendulum. For locomotion smoke, use direct `scripts/train.py` runs.
- There is still a spec/implementation question around the repo's claimed tanh-squashed action contract. That was not resolved in this pass because it was not required to clear the confirmed harness failures.

## Suggested handoff prompt

Use this prompt for the next agent:

> Continue from `RL-Project/08_handoff_fix_log.md`. The harness fixes for false action-bound failures, collapse mislabeling, suite completion semantics, and TRPO entropy instrumentation are already implemented and verified by smoke runs. Your task is to rerun `main_benchmark`, inspect the new run statuses and collapse reasons, rebuild analysis, and determine whether any remaining failures are real algorithm instability or new instrumentation issues. Do not rely on the earlier 80/80 invalid benchmark conclusion without recomputing from the rerun.
