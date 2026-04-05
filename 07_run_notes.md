# 07 Run Notes

## Commands executed

1. `python scripts/train.py --algo ppo_kl --env pendulum_v1 --seed 0 --suite-name smoke_single --run-tag ppo_kl --total-timesteps 20000 --eval-every 5000 --eval-episodes 5 --device cpu --resume`
2. `python -m pip install --upgrade pip`
3. `python -m pip install -r requirements.txt`
4. `python scripts/train.py --algo ppo_kl --env pendulum_v1 --seed 0 --suite-name smoke_single --run-tag ppo_kl --total-timesteps 20000 --eval-every 5000 --eval-episodes 5 --device cpu --resume`
5. `python scripts/run_smoke_test.py --suite-name smoke_pendulum --total-timesteps 20000 --eval-every 5000 --eval-episodes 5 --device cpu`
6. `python scripts/run_main_benchmark.py --suite-name main_benchmark --device auto`
7. `python scripts/build_analysis.py`
8. `python -m pytest tests/test_analysis_pipeline.py`
9. `python scripts/build_analysis.py`

## Which runs completed

Smoke runs:

- `smoke_single`: completed for `ppo_kl` on `pendulum_v1`, seed `0`
- `smoke_pendulum`: completed for `a2c`, `ppo_clip`, `ppo_kl`, and `trpo` on `pendulum_v1`

Main benchmark:

- All 80 expected algorithm-environment-seed folders were created
- All 80 runs wrote the required raw files
- `results/manifests/main_benchmark/suite.json` reports `planned_run_count = 80`

## Which runs failed or were missing

Main benchmark budget-complete runs:

- `0 / 80`

Main benchmark missing runs:

- `0 / 80`

Main benchmark failed shell launches:

- `0 / 80`

Main benchmark `partial_or_corrupted` runs:

- `80 / 80`

Classification rule used for the primary benchmark:

- `completed`: reached target environment steps and did not set `nan_or_divergence_flag`
- `failed`: run status not marked completed, or critical files missing
- `missing`: expected run folder absent
- `partial_or_corrupted`: raw files exist but the run terminated before its target budget or set divergence/collapse indicators

Outcome:

- every main-benchmark run is `partial_or_corrupted`
- every main-benchmark run has `collapse_flag = 1`
- every main-benchmark run has `nan_or_divergence_flag = 1`
- all recorded collapse reasons are `policy_loss_nan`

## Any reruns, resumes, or recovered outputs

- The first documented smoke command failed immediately because dependencies were not installed
- After `pip install -r requirements.txt`, the same smoke command was rerun successfully
- No pre-existing benchmark outputs were found anywhere else in the broader `Project` workspace
- No results were recovered from outside `RL-Project`
- The analysis pipeline initially failed on an empty-curve bug and was rerun after a local fix in `src/analysis/aggregate.py`

## Any anomalies that matter for interpretation

- The documented main benchmark command returned successfully at the shell level, but all 80 runs terminated far short of their target step budgets
- Example collapse signatures:
  - `pendulum_v1` runs stopped at `288` to `1056` steps instead of `100000`
  - `hopper_v4`, `walker2d_v4`, and `halfcheetah_v4` runs mostly stopped at `8` or `16` steps instead of `300000`
- The raw logs show a universal collapse reason of `policy_loss_nan`
- The generated tables and figures therefore summarize an all-collapse failure state, not a valid completed benchmark
- `run_status.json` reports `status = completed` for these runs, so shell completion must not be treated as scientific completion

## Whether the sensitivity sweep is complete, partial, or not yet run

- Not yet run

Reason:

- The main benchmark is not scientifically valid yet
- Running the sweep grid before fixing the all-run `policy_loss_nan` failure mode would likely only generate more collapsed runs

## How strong are these results?

- This is not a complete main benchmark
- This is stronger than smoke-test-only evidence because the full main matrix was launched and audited
- It is still not strong enough for honest results interpretation because `0 / 80` main runs reached their intended budgets
- Current evidence should be treated as benchmark-execution and failure-diagnosis evidence, not as interpretable algorithm-comparison evidence

## What is inside RL-Project vs outside it?

Inside `RL-Project`:

- source code, configs, scripts, tests, and docs
- lightweight manifests in `results/manifests/`
- tracked report-facing artifacts in `report_assets/main_benchmark/`
- this note and the provenance note

Inside `RL-Project` but ignored:

- raw run dumps in `results/raw/`
- generated analysis outputs in `results/aggregated/`
- checkpoints, tensorboard logs, and other execution byproducts under ignored result paths

Outside `RL-Project`:

- the original outer workspace copy of the project files
- no recovered benchmark outputs were found outside the repo

## .gitignore policy applied before execution

The repo now ignores:

- `results/raw/`
- `results/aggregated/`
- recovered result folders
- checkpoints and serialized model files
- tensorboard directories
- logs, caches, and temp exports
- local virtual environments
- notebook checkpoints
- machine-specific scratch artifacts

This keeps the repo limited to commit-worthy assets while still allowing the benchmark tooling to write repo-local execution directories when needed.
