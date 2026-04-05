## 1. Target repo tree

```text
rl-update-stability/
├─ README.md
├─ pyproject.toml
├─ requirements.txt
├─ configs/
│  ├─ base.yaml
│  ├─ envs/
│  │  ├─ pendulum_v1.yaml
│  │  ├─ hopper_v4.yaml
│  │  ├─ walker2d_v4.yaml
│  │  └─ halfcheetah_v4.yaml
│  └─ algos/
│     ├─ a2c.yaml
│     ├─ ppo_clip.yaml
│     ├─ ppo_kl.yaml
│     └─ trpo.yaml
├─ src/
│  ├─ __init__.py
│  ├─ envs/
│  │  ├─ make_env.py
│  │  ├─ wrappers.py
│  │  └─ normalization.py
│  ├─ policies/
│  │  ├─ actor_critic.py
│  │  └─ init.py
│  ├─ algos/
│  │  ├─ common.py
│  │  ├─ a2c_runner.py
│  │  ├─ ppo_clip_runner.py
│  │  ├─ ppo_kl_runner.py
│  │  └─ trpo_runner.py
│  ├─ callbacks/
│  │  ├─ eval_callback.py
│  │  ├─ checkpoint_callback.py
│  │  └─ metrics_callback.py
│  ├─ metrics/
│  │  ├─ stability.py
│  │  ├─ collapse.py
│  │  └─ logging_schema.py
│  ├─ analysis/
│  │  ├─ aggregate.py
│  │  ├─ bootstrap_ci.py
│  │  └─ plots.py
│  └─ utils/
│     ├─ seeding.py
│     ├─ paths.py
│     └─ serialization.py
├─ scripts/
│  ├─ train.py
│  ├─ evaluate.py
│  ├─ run_main_benchmark.py
│  ├─ run_sweeps.py
│  └─ summarize_results.py
├─ results/
│  ├─ raw/
│  │  └─ {algorithm}/{env_id}/seed_{seed}/
│  │     ├─ run_config.json
│  │     ├─ metrics.csv
│  │     ├─ updates.csv
│  │     ├─ collapse.json
│  │     ├─ latest.pt
│  │     ├─ best_by_eval_mean.pt
│  │     ├─ vecnormalize.pkl
│  │     └─ rng_state.pkl
│  └─ aggregated/
│     ├─ tables/
│     ├─ figures/
│     └─ summaries/
├─ notebooks/
│  ├─ debug_pendulum.ipynb
│  └─ analysis.ipynb
└─ tests/
   ├─ test_env_shapes.py
   ├─ test_normalization.py
   ├─ test_logging_schema.py
   ├─ test_eval_determinism.py
   └─ test_checkpoint_resume.py
```

This tree is the right starting point because the algorithm spec already froze a modular repo with shared env/policy/callback/metrics layers, raw vs aggregated results separation, and explicit test coverage.  

## 2. One-paragraph purpose for each top-level folder

**configs/**
Holds frozen experiment configuration, split into shared defaults, per-environment overrides, and per-algorithm overrides. This keeps controlled variables explicit and prevents hard-coded hyperparameters from drifting across runners.  

**src/**
Contains all importable project logic. Responsibilities must stay separated: env creation and normalization, policy definitions, algorithm runners, callbacks, metric definitions, postprocessing, and utilities. This is the core benchmark code, not ad hoc scripts. 

**scripts/**
Contains thin CLI entrypoints only. Each script should parse config, call library code in `src/`, and write outputs into the frozen results schema. Script logic should stay orchestration-only. 

**results/**
Contains all run outputs. `raw/` stores per-run artifacts exactly as produced by training. `aggregated/` stores cross-seed tables, plots, and summary artifacts derived from `raw/`. Nothing in `results/` should be manually edited.  

**notebooks/**
Contains optional debugging and exploratory analysis notebooks. These are for inspection only, not for core training or official aggregation. Any result needed for the report must also be reproducible from a script. 

**tests/**
Contains smoke tests for environment wiring, normalization, logging schema, deterministic evaluation behavior, and checkpoint resume. Since the project’s conclusions depend on strict parity and reproducibility, these tests are not optional.  

## 3. Required Python modules/files

These files should exist and have non-overlapping responsibilities.

**Root**

* `README.md`: setup, benchmark overview, exact CLI usage, expected outputs, and reproduction instructions.
* `pyproject.toml`: package metadata, entrypoints, tool config.
* `requirements.txt`: pinned runtime dependencies. The frozen stack is Python, PyTorch, Gymnasium, MuJoCo, SB3, and SB3-Contrib. 

**src/envs/**

* `make_env.py`: create train/eval envs for `Pendulum-v1`, `Hopper-v4`, `Walker2d-v4`, `HalfCheetah-v4`; handle vectorization and seeding.
* `wrappers.py`: action rescaling, time-limit handling, eval wrapper composition.
* `normalization.py`: observation normalization with saved running stats and eval-time frozen stats. Observation normalization must be on for all methods; reward normalization must remain off. 

**src/policies/**

* `actor_critic.py`: separate actor and critic MLPs, 64x64 tanh, Gaussian actor, learned log-std, orthogonal init hooks.
* `init.py`: frozen initialization policy, including output gains and initial log-std. 

**src/algos/**

* `common.py`: rollout schema, shared batch utilities, GAE, optimizer helpers, checkpoint step helpers.
* `a2c_runner.py`: A2C training loop using shared rollout/eval/logging interfaces.
* `ppo_clip_runner.py`: PPO-Clip runner with clipped ratio surrogate.
* `ppo_kl_runner.py`: custom PPO-KL runner with plain ratio surrogate plus target-KL early stopping, not clip+KL hybrid.
* `trpo_runner.py`: TRPO runner built on the library-backed trust-region stack. PPO-KL is the main required custom algo file.  

**src/callbacks/**

* `eval_callback.py`: step-0 and every-10k-step eval over 10 deterministic episodes.
* `checkpoint_callback.py`: write `latest.pt`, `best_by_eval_mean.pt`, `vecnormalize.pkl`, `rng_state.pkl`.
* `metrics_callback.py`: emit checkpoint rows and update-level rows.  

**src/metrics/**

* `stability.py`: exact unstable-update rule.
* `collapse.py`: exact collapse-event rule.
* `logging_schema.py`: central field definitions and row validators for `metrics.csv`, `updates.csv`, and `collapse.json`. 

**src/analysis/**

* `aggregate.py`: load raw runs, compute per-env and per-method summaries.
* `bootstrap_ci.py`: bootstrap CIs for AUC, final return, unstable-update rate, and wall-clock.
* `plots.py`: learning curves, instability/collapse plots, sensitivity plots, wall-clock vs return scatter. 

**src/utils/**

* `seeding.py`: Python/NumPy/PyTorch/env seed handling.
* `paths.py`: canonical output paths.
* `serialization.py`: run config save/load, RNG state capture, normalization stat save/load. 

## 4. Required config files

* `configs/base.yaml`: shared benchmark constants: `gamma=0.99`, `gae_lambda=0.95`, rollout batch `2048`, eval every `10000` steps, eval episodes `10`, observation normalization on, reward normalization off, advantage normalization on, actor/critic architecture, seeds, checkpoint cadence. 
* `configs/envs/pendulum_v1.yaml`: `Pendulum-v1`, total steps `100000`.
* `configs/envs/hopper_v4.yaml`: `Hopper-v4`, total steps `300000`.
* `configs/envs/walker2d_v4.yaml`: `Walker2d-v4`, total steps `300000`.
* `configs/envs/halfcheetah_v4.yaml`: `HalfCheetah-v4`, total steps `300000`. 
* `configs/algos/a2c.yaml`: actor LR `3e-4`, critic LR `1e-3`.
* `configs/algos/ppo_clip.yaml`: actor LR `3e-4`, critic LR `1e-3`, epochs `10`, minibatch `256`, clip epsilon `0.20`.
* `configs/algos/ppo_kl.yaml`: actor LR `3e-4`, critic LR `1e-3`, max policy epochs `10`, minibatch `256`, target KL `0.02`, early-stop enabled.
* `configs/algos/trpo.yaml`: max KL delta `0.02`, critic LR `1e-3`, critic epochs `10`, critic minibatch `256`, CG steps `10`, damping `0.1`, line-search steps `10`, backtrack coeff `0.8`. 

## 5. CLI commands that the repo should expose

Use these commands as the public CLI surface:

* `python scripts/train.py --algo {a2c|ppo_clip|ppo_kl|trpo} --env {pendulum_v1|hopper_v4|walker2d_v4|halfcheetah_v4} --seed N`
* `python scripts/evaluate.py --run-dir results/raw/{algorithm}/{env_id}/seed_{seed}`
* `python scripts/run_main_benchmark.py`
* `python scripts/run_sweeps.py --env {hopper_v4|halfcheetah_v4}`
* `python scripts/summarize_results.py`

Expected behavior:

* `train.py` runs exactly one training job and writes one raw run folder.
* `evaluate.py` re-runs deterministic evaluation with frozen normalization stats.
* `run_main_benchmark.py` launches the main 4 algorithms × 4 envs × 5 seeds matrix.
* `run_sweeps.py` launches one-factor-at-a-time sweeps only on Hopper and HalfCheetah.
* `summarize_results.py` builds aggregated tables and figures from raw runs. This matches the frozen main benchmark, sweep plan, and result schema.  

## 6. Minimal smoke-test matrix

The smoke phase should be small, fast, and debugging-oriented.

| Algo     | Env         | Seeds | Purpose                            |
| -------- | ----------- | ----: | ---------------------------------- |
| A2C      | Pendulum-v1 |     1 | End-to-end sanity check            |
| PPO-Clip | Pendulum-v1 |     1 | Ratio/clipping/logging sanity      |
| PPO-KL   | Pendulum-v1 |     1 | Early-stop logic sanity            |
| TRPO     | Pendulum-v1 |     1 | Trust-region plumbing sanity       |
| PPO-Clip | Hopper-v4   |     1 | First MuJoCo sanity check          |
| PPO-KL   | Hopper-v4   |     1 | KL stats and event detection check |

Smoke-test acceptance:

* training starts and reaches at least one checkpoint,
* deterministic eval runs,
* all required raw artifacts are written,
* no schema violations,
* resume from checkpoint works on at least one run. Pendulum is explicitly the debugging env; Hopper is the first instability-sensitive locomotion check.  

## 7. Full benchmark matrix

**Main benchmark**

* Algorithms: `A2C`, `PPO-Clip`, `PPO-KL`, `TRPO`
* Environments: `Pendulum-v1`, `Hopper-v4`, `Walker2d-v4`, `HalfCheetah-v4`
* Seeds: `5` per algorithm-env pair
  Total main runs = `4 × 4 × 5 = 80`.  

**Sensitivity sweeps**

* Sweep envs only: `Hopper-v4`, `HalfCheetah-v4`
* Seeds: `3` per sweep setting
* A2C actor LR: `{1e-4, 3e-4, 1e-3}`
* PPO-Clip actor LR: `{1e-4, 3e-4, 1e-3}`
* PPO-Clip epsilon: `{0.10, 0.20, 0.30}`
* PPO-KL actor LR: `{1e-4, 3e-4, 1e-3}`
* PPO-KL target KL: `{0.01, 0.02, 0.05}`
* TRPO delta: `{0.01, 0.02, 0.05}`
  This is one-factor-at-a-time only, not a Cartesian grid.  

## 8. Required logging artifacts

Per run, the raw run folder must contain:

* `run_config.json`
* `metrics.csv`
* `updates.csv`
* `collapse.json`
* `latest.pt`
* `best_by_eval_mean.pt`
* `vecnormalize.pkl`
* `rng_state.pkl` 

Required `metrics.csv` fields include environment steps, wall-clock time, eval return stats, train episode stats, KL stats, policy-ratio stats, instability/collapse flags, NaN/divergence flag, policy/value/entropy stats, advantage stats, gradient norms, and algorithm-specific fields like `clip_fraction`, `epochs_completed_before_early_stop`, `cg_iterations_used`, `line_search_backtracks`, and `accepted_step_fraction`. 

Required `updates.csv` fields include `update_index`, step range, wall-clock, KL stats, ratio stats, `large_step_no_drop`, `drop_without_large_step`, `unstable_update_flag`, `epochs_completed`, `early_stopped`, `policy_loss`, `value_loss`, and `entropy`. 

Required `collapse.json` fields include `collapse_flag`, `collapse_step`, `collapse_reason`, `R_init`, `R_best`, and `collapse_threshold`. 

## 9. Required output plots and tables

**Plots**

* Mean learning curves with seed bands for each environment
* Instability-event rate by method
* Collapse-rate bar chart
* Wall-clock vs return scatter
* Sensitivity plots for A2C LR, PPO-Clip LR/epsilon, PPO-KL LR/target-KL, and TRPO delta. These were already identified as the strong final figures. 

**Tables**

* Final return table by algorithm × environment
* AUC table by algorithm × environment
* Seed variance / CI table
* Wall-clock summary table
* Unstable-update count/rate table
* Collapse probability table
* Sweep summary tables for Hopper and HalfCheetah. These match the protocol’s effect-size-first evaluation plan and the MVP/final deliverables.  

## 10. Acceptance criteria for each major file

**README.md**
Passes if a coding agent can install dependencies, run one smoke-test command, run the full main benchmark, and regenerate summary outputs without guessing missing steps.

**configs/base.yaml**
Passes if every shared fairness-critical setting is defined once here and not duplicated inconsistently elsewhere.

**configs/envs/*.yaml**
Passes if each file defines exactly one env ID and one total-step budget, with no algorithm logic.

**configs/algos/*.yaml**
Passes if each file contains only that algorithm’s tunable settings and default values frozen by the protocol.

**src/envs/make_env.py**
Passes if it creates correct train and eval envs for all four frozen environments, with correct vectorization and seeding.

**src/envs/normalization.py**
Passes if observation normalization updates only during training and reuses frozen stats during evaluation; reward normalization must remain disabled. 

**src/policies/actor_critic.py**
Passes if all algorithms use the same 64x64 tanh separate actor/critic architecture with the frozen Gaussian action parameterization. 

**src/algos/a2c_runner.py**
Passes if it can train end to end and emit all required artifacts with one policy update path consistent with the shared batch and eval schema.

**src/algos/ppo_clip_runner.py**
Passes if the policy objective is clipped PPO and `clip_fraction` is logged at checkpoint time. 

**src/algos/ppo_kl_runner.py**
Passes only if it uses the plain ratio surrogate, no clip term, target-KL early stopping, and logs completed policy epochs. It fails if it silently uses stock clip PPO with a KL stop fuse.  

**src/algos/trpo_runner.py**
Passes if it performs trust-region updates with the frozen delta/defaults and logs CG/line-search diagnostics.

**src/metrics/stability.py**
Passes if unstable-update detection matches the frozen two-part rule: KL condition plus next-eval performance drop.  

**src/metrics/collapse.py**
Passes if collapse detection matches the frozen relative-to-own-progress rule and handles NaN/numeric failure as immediate collapse. 

**src/metrics/logging_schema.py**
Passes if it validates every required field and blocks partial or malformed rows.

**scripts/run_main_benchmark.py**
Passes if it launches exactly the main 80-run matrix and writes outputs under the canonical path scheme.

**scripts/run_sweeps.py**
Passes if it launches only the declared one-factor-at-a-time sweeps on Hopper and HalfCheetah, with 3 seeds per setting.

**scripts/summarize_results.py**
Passes if it can regenerate all required tables and figures from raw artifacts alone.

**tests/**
Passes if all five tests run successfully in CI or local smoke mode and catch schema/resume/eval-determinism failures. 

## 11. What Codex must not invent

* New algorithms beyond A2C, PPO-Clip, PPO-KL, and TRPO
* New environments beyond `Pendulum-v1`, `Hopper-v4`, `Walker2d-v4`, `HalfCheetah-v4`
* Different network sizes, activations, or shared-trunk variants
* Reward normalization, reward scaling, or reward clipping
* Extra PPO safeguards that blur PPO-Clip and PPO-KL
* A different PPO-KL definition
* A different eval cadence, eval episode count, or step budget
* Different seeds or sweep grids than the frozen plan
* Missing raw artifacts or renamed schema fields
* Hidden implementation asymmetries between algorithms
* Ad hoc notebook-only analysis as official output
* Selective reporting of only best seeds or only successful runs
  These prohibitions are necessary because the benchmark’s core threat is unfair attribution from implementation drift and hidden confounds.  
