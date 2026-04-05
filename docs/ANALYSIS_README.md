# Analysis Regeneration

This repository now includes a log-driven analysis pipeline for the benchmark defined in [03_experiment_protocol.md](C:\Users\hshre\OneDrive\Documents\42 - Agents\Codex\4453\Project\03_experiment_protocol.md).

## What It Builds

Running the analysis step over `results/raw/` produces:

- checkpoint-level and run-level CSV summaries
- LaTeX tables for final return, variability, wall-clock, and instability
- publication-ready PNG and PDF figures for return curves, wall-clock, and sensitivity sweeps
- a JSON summary of generated outputs

The pipeline is designed to support a short report, so it focuses on a small number of stronger outputs instead of generating many overlapping plots.

## Expected Raw Inputs

Each completed run should contain the benchmark artifacts already defined by the training code:

- `run_config.json`
- `metrics.csv`
- `updates.csv`
- `collapse.json`

The analysis loader also reads suite metadata from `results/manifests/<suite_name>/suite.json` when runs belong to the main benchmark or sensitivity sweeps.

## Regenerate Outputs

From the repository root:

```bash
python scripts/build_analysis.py
```

Backward-compatible alias:

```bash
python scripts/summarize_results.py
```

Optional overrides:

```bash
python scripts/build_analysis.py --raw-root path/to/raw --aggregated-root path/to/aggregated
```

## Output Layout

Outputs are written under `results/aggregated/`:

- `tables/`
- `figures/`
- `summaries/`

The most report-ready files are:

- `results/aggregated/figures/training_return_curves.{png,pdf}`
- `results/aggregated/figures/wall_clock_comparison.{png,pdf}`
- `results/aggregated/figures/sensitivity_plots.{png,pdf}`
- `results/aggregated/tables/final_evaluation_return.tex`
- `results/aggregated/tables/final_evaluation_variance.tex`
- `results/aggregated/tables/wall_clock_comparison.tex`
- `results/aggregated/tables/instability_frequency.tex`

## Notes

- The curve figure uses `train_episode_return_mean` when available and falls back to `eval_return_mean` if training-return checkpoints are missing.
- Main-benchmark figures prefer runs tagged as `main_benchmark`. If no such suite exists yet, the pipeline falls back to all non-sweep runs.
- Sensitivity plots are inferred from sweep `run_tag` values and the stored resolved config, so algorithm names and paths are not hardcoded in the plotting layer.
