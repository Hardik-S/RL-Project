# Artifact Provenance

Canonical repo root: `RL-Project`.

## Source sync

The benchmark source, configs, scripts, tests, and project docs originally existed in the outer workspace at:

- `Project/configs/`
- `Project/scripts/`
- `Project/src/`
- `Project/tests/`
- `Project/README.md`
- `Project/pyproject.toml`
- `Project/requirements.txt`
- `Project/md files/*.md`

Those commit-worthy assets were copied into `RL-Project` so the git repo now contains the benchmark code and documentation.

## Results recovery audit

Searched locations:

- inside `RL-Project`
- elsewhere in the broader `Project` workspace

Searched for:

- `metrics.csv`
- `updates.csv`
- `collapse.json`
- `run_config.json`
- aggregated CSV/TeX outputs
- PNG/PDF figures
- summary markdown files

Recovery result:

- No pre-existing benchmark result artifacts were found anywhere in the broader `Project` workspace before execution.
- No results were recovered from outside `RL-Project`.

## Storage decisions

The benchmark scripts write to `results/` under the repo root, so raw and aggregated execution artifacts were allowed to live inside `RL-Project` only because the tooling expects repo-local paths.

To avoid polluting version control:

- bulky local execution artifacts remain in ignored paths under `results/raw/` and `results/aggregated/`
- compact report-facing copies were placed under `report_assets/main_benchmark/`
- lightweight manifests remain under `results/manifests/`

## What was copied into tracked report assets

Copied from ignored execution outputs into tracked paths:

- final figures from `results/aggregated/figures/`
- compact summary tables from `results/aggregated/tables/`
- analysis summary JSON from `results/aggregated/summaries/`
- run audit tables generated from `results/raw/main_benchmark/`

These tracked copies exist so the benchmark state can be audited from the canonical repo without committing raw dumps, checkpoints, tensorboard logs, or duplicate bulky result trees.
