# REPRODUCIBILITY

This file defines the exact command lines used to reproduce smoke runs, the full benchmark, and sensitivity sweeps.

## 1. Environment

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Optional verification:

```bash
python -m pytest tests
```

## 2. Exact Run Commands

Run all commands from repository root.

Single-run smoke test:

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

## 3. Aggregate Analysis

```bash
python scripts/build_analysis.py
```

Equivalent alias:

```bash
python scripts/summarize_results.py
```

## 4. Output Locations

- Raw run logs and checkpoints: `results/raw/...`
- Suite manifests: `results/manifests/...`
- Aggregated tables/figures: `results/aggregated/...`

## 5. Reproducibility Notes

- Use the same hardware/software stack when comparing wall-clock results.
- Keep default seed sets from configs unless intentionally running a sweep/ablation.
- Interpret AUC per environment only; do not compare pooled AUC across environments.
