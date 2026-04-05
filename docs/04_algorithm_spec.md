1. Which Python RL stack should be used and why

Use this stack:

Python 3.10+
PyTorch
Gymnasium for environments
MuJoCo for Hopper/Walker2d/HalfCheetah
Stable-Baselines3 (SB3) for the core on-policy infrastructure
SB3-Contrib for TRPO
TensorBoard + CSV/JSON logs for experiment tracking

Why this stack:

The project already locked a Gymnasium/MuJoCo benchmark with Pendulum-v1, Hopper-v4, Walker2d-v4, and HalfCheetah-v4.
SB3 is positioned as a set of reliable PyTorch RL implementations, and its current docs show maintained A2C/PPO support.
SB3-Contrib currently exposes TRPO, including KL-related parameters and the trust-region machinery you need, which removes the highest-risk implementation burden in the project.
SB3 already provides the exact utilities this benchmark needs for fairness: vectorized environments, normalization wrappers, evaluation helpers, callbacks, and logging hooks.
2. Whether to implement from scratch or build on a reliable library

Build on a reliable library. Do not implement the full stack from scratch.

Reason:

The charter identifies TRPO implementation complexity and MuJoCo/tooling friction as major risks.
The literature package also flags that implementation details can dominate PPO vs TRPO results, so using mature, widely used code for the hard parts is more defensible than a student-built second-order optimizer.
The project is explicitly compute-aware and reliability-focused, not novelty-seeking.

So the correct choice is:

Not from scratch
Not purely off-the-shelf either
Hybrid library-backed benchmark
3. The exact definition of “PPO-KL” for this project

For this project, PPO-KL means:

A PPO-style on-policy actor-critic method that uses the same rollout buffer, GAE estimator, actor/critic architecture, minibatching, and multi-epoch optimization structure as PPO-Clip, but replaces the clipped surrogate with the plain importance-ratio surrogate and uses a target-KL early-stop rule to halt policy epochs once the mean KL between old and new policy exceeds the target.

That is:

same actor-critic backbone as PPO-Clip
same gamma = 0.99, gae_lambda = 0.95
same batch size, minibatch size, and max policy epochs
same observation normalization and advantage normalization
no ratio clipping in the policy objective
no adaptive KL penalty coefficient
policy epoch early stopping based on mean KL threshold

This matches the charter’s already-locked recommendation that PPO-KL be instantiated as target-KL early stopping for feasibility and fairness.

4. Whether PPO-KL should be implemented as:
- KL penalty
- target-KL early stopping
- adaptive KL penalty
- another variant

Implement PPO-KL as:

target-KL early stopping

Not as:

fixed KL penalty
adaptive KL penalty
clip + target-KL hybrid
any other PPO variant

This is already the protocol’s frozen choice.

5. The exact pros/cons of each choice in this project context
A. Fixed KL penalty

Pros

Closer to one canonical PPO-KL form from the PPO paper family.
Gives a smooth optimization objective rather than a hard stop.

Cons

Adds an extra coefficient that must be tuned fairly.
Penalty strength is environment-sensitive.
A weak penalty can fail to control large steps; a strong one can freeze learning.
Creates another hidden asymmetry versus PPO-Clip and TRPO.
Bad fit for a course project because it expands the sweep surface.
B. Adaptive KL penalty

Pros

Also canonical in PPO-family discussions.
More principled than a fixed penalty because it tries to track a target KL.

Cons

Requires a controller for the penalty coefficient.
Adds algorithmic state and more moving parts.
Harder to explain cleanly in an 8-page report.
Greater risk that results reflect the adaptation schedule rather than the underlying update-control idea.
Higher implementation asymmetry relative to PPO-Clip.
C. Target-KL early stopping

Pros

Very simple.
Easy to explain and defend.
Keeps PPO-KL close to PPO-Clip in every respect except the actual update-control mechanism.
Low engineering cost.
Directly aligns with the charter’s “freeze one concrete PPO-KL definition and do not change it” principle.
Lets you log a clean metric: epochs_completed_before_early_stop.

Cons

It is not the classic adaptive-KL-penalty variant from the PPO paper.
It controls KL only coarsely, at the epoch level rather than continuously in the loss.
If the target is too loose, it behaves almost like unconstrained PPO.
D. Clip + target-KL hybrid

Pros

Common in practical codebases because it adds a safety fuse.

Cons

Unacceptable for this benchmark.
The protocol explicitly warns: do not give PPO-KL both clip control and KL control unless equivalent extra safeguards are added elsewhere.
It would blur the distinction between PPO-Clip and PPO-KL.
6. The recommended final choice

Final choice: PPO-KL = target-KL early stopping, with no clip term and no KL-penalty coefficient.

Exact default:

max policy epochs: 10
target mean KL: 0.02
stop remaining policy epochs for that update once measured mean KL exceeds 0.02
7. The exact observation normalization policy

Frozen policy:

Use running observation normalization for all four algorithms.
Compute running mean/std on training environments only.
Normalize each observation as (obs - mean) / sqrt(var + eps).
Clip normalized observations to [-10, 10].
Save normalization statistics with each checkpoint.
During evaluation:
use the saved training normalization statistics
do not update them

This matches the protocol’s shared preprocessing rule: observation normalization on, shared across methods, with reward normalization off. It is also consistent with SB3 guidance that input normalization is often essential for successful RL training and that VecNormalize is the standard wrapper for this.

8. The exact reward scaling policy

Frozen policy:

No reward normalization
No reward scaling
No reward clipping

Use raw environment rewards for:

rollout returns
GAE
critic targets
evaluation returns

Reason:

The protocol explicitly turns reward normalization off to avoid changing the meaning of collapse across methods.
Because the benchmark’s core outputs include collapse frequency and unstable-update frequency, introducing reward scaling would add another hidden degree of freedom.
9. The exact evaluation policy

Frozen evaluation policy:

Evaluate at step 0 and then every 10,000 environment steps
Run 10 evaluation episodes per checkpoint
Use deterministic evaluation
for Gaussian policies: use actor mean action
Use a separate evaluation env
Apply the training observation-normalization stats
Do not update normalization stats during eval
Log training and evaluation separately
Report:
eval_return_mean
eval_return_std
eval_return_median

This exactly follows the experiment protocol. It is also consistent with SB3’s evaluation helper guidance: periodic evaluation over multiple episodes, with deterministic=True for stochastic policies.

10. The exact checkpointing/logging schema

Use a checkpoint every evaluation point, meaning every 10,000 env steps plus step 0 metadata.

At each checkpoint, save:

A. Model artifacts
actor parameters
critic parameters
optimizer state for actor, if applicable
optimizer state for critic
normalization statistics
RNG states:
Python
NumPy
PyTorch
env seed info
B. Structured metrics snapshot

Save one row in metrics.csv and one JSON checkpoint summary.

C. Event logs

Append update-level events to updates.csv, including unstable-update diagnostics and KL statistics.

D. Best-model checkpoint

Also keep:

best_by_eval_mean.pt
latest.pt

The protocol already requires logging return, wall-clock, policy KL, failure events, and algorithm-specific diagnostics. SB3 callbacks and logger hooks are adequate for periodic saving and custom scalar logging, but the benchmark-specific event files must be added by custom code.

11. The exact fields each run must save

Each run must save these fields.

run_config.json
run_id
algorithm
env_id
seed
total_timesteps
n_envs
steps_per_env
rollout_batch_size
gamma
gae_lambda
obs_norm_enabled
obs_clip
reward_norm_enabled
adv_norm_enabled
policy_arch
value_arch
activation
init_scheme
algorithm-specific hyperparameters:
A2C: actor LR, critic LR
PPO-Clip: actor LR, critic LR, clip epsilon, epochs, minibatch size
PPO-KL: actor LR, critic LR, target KL, max epochs, minibatch size
TRPO: max KL delta, CG steps, damping, line-search settings, critic LR
metrics.csv

At every checkpoint:

env_steps
wall_clock_seconds
episodes_seen
eval_return_mean
eval_return_std
eval_return_median
train_episode_return_mean
train_episode_length_mean
mean_kl_old_new
max_kl_old_new
policy_ratio_mean
policy_ratio_std
unstable_update_flag
cumulative_unstable_updates
collapse_flag
nan_or_divergence_flag
policy_loss
value_loss
entropy
advantage_mean
advantage_std
value_target_mean
grad_norm_actor
grad_norm_critic
PPO-Clip only: clip_fraction
PPO-KL only: epochs_completed_before_early_stop
TRPO only: cg_iterations_used, line_search_backtracks, accepted_step_fraction
updates.csv

Per training update:

update_index
start_env_steps
end_env_steps
wall_clock_seconds
mean_kl_old_new
max_kl_old_new
policy_ratio_mean
policy_ratio_std
large_step_no_drop
drop_without_large_step
unstable_update_flag
epochs_completed
early_stopped
policy_loss
value_loss
entropy
collapse.json
collapse_flag
collapse_step
collapse_reason
R_init
R_best
collapse_threshold
binary artifacts
latest.pt
best_by_eval_mean.pt
vecnormalize.pkl
rng_state.pkl
12. The exact folder structure the repo should use

Use this repo structure:

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
│  │  ├─ a2c_runner.py
│  │  ├─ ppo_clip_runner.py
│  │  ├─ ppo_kl_runner.py
│  │  ├─ trpo_runner.py
│  │  └─ common.py
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
13. Implementation decisions frozen
Final stack
Gymnasium + MuJoCo + PyTorch + SB3 + SB3-Contrib
Final implementation mode
Library-backed benchmark, not full from-scratch
Use library implementations where they reduce risk.
Add custom benchmark code where the project requires exact parity or custom metrics.
What should be library-backed
env creation
vectorized envs
observation normalization
A2C baseline
PPO-Clip baseline
TRPO baseline
evaluation helper patterns
callback/logging infrastructure
What custom code still needs to be written
PPO-KL runner
same PPO training scaffold
no clip objective
target-KL early stopping
custom logging of completed epochs
Unified benchmark harness
same config system
same seeds
same checkpoint cadence
same path structure
Custom stability metrics
unstable-update detection
collapse detection
event counters
Custom serialization
exact run config dump
normalization-stat save/load
RNG state capture
Custom aggregation scripts
AUC
bootstrap CIs
Wilson intervals for collapse
cross-seed tables and plots
One important constraint

Do not use SB3’s stock PPO class for both PPO-Clip and PPO-KL without modification, because the project’s fairness rules already forbid giving PPO-KL both clip control and KL control simultaneously.