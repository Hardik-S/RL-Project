# RL Update Stability

Controlled benchmark for update-stability behavior in continuous-control RL.  
Algorithms: `a2c`, `ppo_clip`, `ppo_kl`, `trpo`.  
Environments: `Pendulum-v1`, `Hopper-v4`, `Walker2d-v4`, `HalfCheetah-v4`.

## Submission-Facing Files

- `docs/final_report_submission.pdf`: canonical final report artifact for submission.
- `docs/final_report_submission.tex`: canonical final report source.
- `docs/final_report_draft.md`: earlier Markdown draft kept for provenance.
- `docs/FINAL_SUBMISSION_CHECKLIST.md`: last-pass submission checklist.
- `docs/REPRODUCIBILITY.md`: exact reproduction commands.
- `docs/artifact_provenance.md`: what is authoritative vs legacy.

## Setup

1. Create and activate a Python `3.10+` virtual environment.
2. Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

3. Run tests:

```bash
python -m pytest tests
```

## Reproduction Commands

Run from repository root.

Single-run smoke test (one algorithm, Pendulum):

```bash
python scripts/train.py --algo ppo_kl --env pendulum_v1 --seed 0 --suite-name smoke_single --run-tag ppo_kl --total-timesteps 20000 --eval-every 5000 --eval-episodes 5 --device cpu --resume
```

All-algorithm Pendulum smoke test:

```bash
python scripts/run_smoke_test.py --suite-name smoke_pendulum --total-timesteps 20000 --eval-every 5000 --eval-episodes 5 --device cpu
```

Full benchmark:

```bash
python scripts/run_main_benchmark.py --suite-name main_benchmark --device auto
```

Sensitivity sweep:

```bash
python scripts/run_sweeps.py --env hopper_v4 --suite-name sweep_hopper_v4 --device auto
```

Build aggregated analysis tables/figures:

```bash
python scripts/build_analysis.py
```

## Results Layout

Raw runs:

```text
results/raw/{suite_name}/{algorithm}/{env_key}/seed_{seed}/{run_tag}/
```

Aggregated outputs:

- `results/aggregated/tables/`
- `results/aggregated/figures/`
- `results/aggregated/summaries/`

## Notes

- Wall-clock comparisons are meaningful only when runs are produced on the same hardware/software stack.
- AUC is interpreted per environment; do not pool raw AUC across environments.
- MuJoCo-backed environments require a working MuJoCo setup compatible with `gymnasium[mujoco]`.
- Treat per-run logs as the source of truth when they disagree with aggregate exports.
- The tracked files under `report_assets/main_benchmark/` are legacy copies from an earlier aggregation pass and are not authoritative for the final submission unless revalidated against raw logs.
