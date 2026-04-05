Benchmark objective
Measure whether explicit policy-step control improves update stability, collapse resistance, and robustness to update-size choices in continuous control, relative to a standard actor-critic baseline, under a strictly matched setup. The benchmark is therefore not a “who gets the highest score at any cost” exercise; it is a compute-aware test of whether TRPO, PPO-Clip, and PPO-KL are more stable and more robust than A2C when architecture, environment set, and evaluation protocol are held fixed.
Algorithms to compare
Use exactly four algorithms:
A2C: synchronous on-policy actor-critic baseline with advantage-based policy updates.
PPO-Clip: PPO with clipped probability-ratio surrogate.
PPO-KL: same actor-critic backbone, but use KL-based control via a target-KL early-stop rule during policy epochs. This is the concrete choice because it is fair, simple, and already recommended in the charter.
TRPO: KL-constrained trust-region update with separate critic fitting.

Why this choice: the proposal fixes these four algorithms, and the charter already resolves the ambiguous PPO-KL variant in the most feasible way for a one-student project.

Environments to compare
Use exactly four environments:
Pendulum-v1
Hopper-v4
Walker2d-v4
HalfCheetah-v4

Role of each environment:

Pendulum-v1: debugging and fast sanity-check environment.
Hopper-v4: instability-sensitive locomotion task.
Walker2d-v4: harder biped control.
HalfCheetah-v4: standard higher-dimensional locomotion benchmark.

Why this choice: it matches the proposal and the charter’s concrete environment lock, while staying feasible for the course timeline.

Shared architecture specification
Use the same policy/value design for all four methods.

Policy network

Separate MLP actor
Hidden sizes: [64, 64]
Activation: tanh
Output: Gaussian mean vector
Log standard deviation: one learned, state-independent parameter vector per action dimension
Action distribution: diagonal Gaussian, tanh-squashed to action bounds

Value network

Separate MLP critic
Hidden sizes: [64, 64]
Activation: tanh
Output: scalar value

Initialization

Orthogonal initialization for all linear layers
Actor hidden gain: sqrt(2)
Actor output gain: 0.01
Critic output gain: 1.0
Initial log std: -0.5

Why this choice: two separate 64x64 tanh MLPs are strong enough for the chosen benchmark but much cheaper and cleaner than larger architectures. Separate actor and critic networks are preferred over a shared trunk because the project is about policy-update stability, and this avoids representation-sharing confounds while still satisfying the “same policy network architecture / same value network architecture” requirement.

Shared preprocessing and normalization policy
Use one preprocessing policy for all methods and environments:
Observation normalization: on, with running mean/std from training data only
Normalized observations clipped to [-10, 10]
Reward clipping: off
Reward normalization: off
Advantage normalization: on, per update batch
Return targets for value learning: unnormalized discounted returns / GAE returns
Episode truncation handling: bootstrap with value on time-limit truncation; no bootstrap on true terminal states
Same action rescaling and same tanh-squash correction for all methods

Why this choice: observation normalization materially improves stability at low engineering cost; reward normalization is deliberately turned off because it can change the effective meaning of collapse and make cross-method comparisons harder to interpret. Advantage normalization stays on because the charter explicitly wants matched estimator-side controls wherever possible.

Shared rollout / batch / evaluation policy
Use one on-policy data-collection and evaluation protocol across all methods.

Shared rollout settings

Discount factor gamma = 0.99
GAE lambda = 0.95
Total rollout batch per update: 2048 environment steps
Vectorized environments: 8
Steps per environment per update: 256
Same seed list per matched comparison
Same checkpoint cadence for every algorithm

Algorithm-specific optimization within the same batch

A2C
1 policy update per rollout batch
1 critic update pass per rollout batch
Actor LR 3e-4
Critic LR 1e-3
PPO-Clip
Policy epochs 10
Critic epochs 10
Minibatch size 256
Actor LR 3e-4
Critic LR 1e-3
Clip epsilon 0.20
PPO-KL
Policy epochs 10 max
Critic epochs 10
Minibatch size 256
Actor LR 3e-4
Critic LR 1e-3
Target KL 0.02
Early-stop remaining policy epochs when mean KL on the current update exceeds 0.02
TRPO
1 trust-region policy update per rollout batch
Critic epochs 10
Critic minibatch size 256
Critic LR 1e-3
Max KL delta = 0.02
Conjugate-gradient steps 10
Damping 0.1
Backtracking line-search steps 10
Backtrack coefficient 0.8

Evaluation settings

Deterministic evaluation policy: use actor mean action
Evaluate every 10,000 environment steps
10 evaluation episodes per checkpoint
Also evaluate once at step 0
Log training and evaluation separately

Why this choice: 2048 steps/update is large enough for stable on-policy estimates but still affordable for TRPO; 10,000-step checkpoints give enough temporal resolution to detect unstable updates and collapses without drowning the project in logs. The PPO-KL choice is intentionally simple and consistent with the charter.

Training budget per environment
Use these fixed budgets:
Environment	Total training steps per run
Pendulum-v1	100,000
Hopper-v4	300,000
Walker2d-v4	300,000
HalfCheetah-v4	300,000

Why this choice: the proposal wants matched budgets and a compute-aware course project. These budgets are large enough to show stability patterns, but small enough that default comparisons plus sensitivity sweeps remain tractable, including TRPO. The project is about robustness trends, not squeezing out asymptotic leaderboard returns.

Number of seeds and justification
Use:
5 seeds for the primary matched benchmark on all four environments
3 seeds for the secondary hyperparameter sensitivity sweeps

Justification:

The charter already locks 5 seeds for the strong final experiment.
The literature package explicitly treats seed variance as central, not optional.
Reducing sweeps to 3 seeds is the compute-aware compromise that preserves robustness checking without exploding run count.

Why this choice: five seeds is the minimum acceptable standard for the main claim here; using fewer on the main comparison would weaken the whole project. Three seeds is acceptable for secondary sensitivity maps because those runs are interpretive support, not the primary result.

Metrics to log every training checkpoint
At every evaluation checkpoint, log the following:

Core progress metrics

env_steps
wall_clock_seconds
episodes_seen
eval_return_mean
eval_return_std
eval_return_median
train_episode_return_mean over episodes completed since last checkpoint
train_episode_length_mean

Stability metrics

mean_kl_old_new on the most recent update batch
max_kl_old_new on the most recent update batch
policy_ratio_mean
policy_ratio_std
unstable_update_flag
cumulative_unstable_updates
collapse_flag
nan_or_divergence_flag

Optimization metrics

policy_loss
value_loss
entropy
advantage_mean
advantage_std
value_target_mean
grad_norm_actor where applicable
grad_norm_critic

Algorithm-specific metrics

PPO-Clip: clip_fraction
PPO-KL: epochs_completed_before_early_stop
TRPO: cg_iterations_used, line_search_backtracks, accepted_step_fraction

Why this choice: the proposal and charter make unstable updates, collapse, variance, and wall-clock first-class outcomes. Logging only return would miss the point of the benchmark.

Exact collapse-event definition
A run is marked as having a collapse event if:
Define
R_init = mean of the first 3 evaluation checkpoints after step 0
R_best = best evaluation return achieved so far
A run becomes collapse-eligible only after it has once exceeded
R_init + 0.60 * (R_best - R_init)
After becoming collapse-eligible, mark a collapse if the run stays below
R_init + 0.25 * (R_best - R_init)
for 5 consecutive evaluation checkpoints
Also mark immediate collapse if any of the following occurs:
NaN in policy loss
NaN in value loss
NaN in action distribution parameters
line search fails permanently and no valid policy step can be taken
environment interaction becomes numerically invalid

Why this choice: it is exactly the charter’s relative-to-own-progress definition, which is better than a fixed reward threshold because reward scales differ sharply across environments.

Exact unstable-update definition
Treat instability as an update-level event, not a whole-run label.

For update block u, mark unstable_update = 1 iff both conditions hold:

KL condition

For PPO-KL and TRPO:
mean_kl_old_new > 2 * nominal_target_kl
For A2C and PPO-Clip:
mean_kl_old_new > 0.05

Performance condition

Let M3 = trailing mean of the previous 3 evaluation returns
Let Range_so_far = max_eval_so_far - mean(first 3 eval returns)
Require
next_eval_return < M3 - 0.20 * Range_so_far

Additional rule:

If the KL condition fires but the performance condition does not, do not count it as unstable; log it as large_step_no_drop = 1
If the performance condition fires but KL does not, do not count it as unstable; log it as drop_without_large_step = 1

Why this choice: the charter already recommends this two-part definition, and the literature framing says the project should capture large policy movement plus immediate performance damage, not one without the other.

Hyperparameter sensitivity plan
Do not run a full Cartesian grid. Use one-factor-at-a-time sweeps after the default matched benchmark.

Primary benchmark defaults

A2C actor LR: 3e-4
PPO-Clip actor LR: 3e-4, clip epsilon 0.20
PPO-KL actor LR: 3e-4, target KL 0.02
TRPO max KL delta: 0.02

Sensitivity sweep environments

Hopper-v4
HalfCheetah-v4

Sensitivity sweep design

A2C: actor LR in {1e-4, 3e-4, 1e-3}
PPO-Clip:
actor LR in {1e-4, 3e-4, 1e-3} with epsilon fixed at 0.20
epsilon in {0.10, 0.20, 0.30} with actor LR fixed at 3e-4
PPO-KL:
actor LR in {1e-4, 3e-4, 1e-3} with target KL fixed at 0.02
target KL in {0.01, 0.02, 0.05} with actor LR fixed at 3e-4
TRPO:
max KL delta in {0.01, 0.02, 0.05}

Do not sweep

hidden sizes
gamma
GAE lambda
critic LR
number of policy epochs
minibatch size

Why this choice: the proposal and charter explicitly say to vary learning rate and trust-region-related hyperparameters, but the project should remain feasible. This plan isolates the relevant aggressiveness knobs without creating an unfinishable study.

Wall-clock measurement protocol
Wall-clock must be measured in a way that is actually comparable.

Rules

Use the same physical machine for all final runs
Same Python version, same library versions, same MuJoCo/Gymnasium stack
Same number of environment workers: 8
Same CPU/GPU availability for all methods
Same logging backend and checkpoint cadence
Start timer immediately before the first training environment step
Stop timer after the final evaluation checkpoint is completed
Exclude one-time setup outside the run:
package install
model code import
environment asset download
Include:
rollout collection
policy/value optimization
KL computation
line search / conjugate-gradient work
periodic evaluation

Report

total wall-clock seconds
environment steps per second
seconds per 10,000 environment steps
median wall-clock across seeds
mean wall-clock across seeds

Why this choice: TRPO is expected to be slower per update, and the proposal explicitly says wall-clock belongs in the benchmark. Measuring only sample efficiency would systematically hide one of the project’s central tradeoffs.

Statistical comparison plan
Use a small-sample, effect-size-first plan.

Primary per-environment outcomes

AUC of evaluation-return curve up to the fixed budget
mean of the last 5 evaluation checkpoints
unstable-update rate per 100,000 steps
collapse probability across seeds
total wall-clock time

Inference

Report mean ± 95% bootstrap CI over seeds for:
AUC
final performance
unstable-update rate
wall-clock
For pairwise comparisons, compare each constrained method against A2C
Report bootstrap CI of the mean difference
For collapse probability, report:
proportion collapsed
Wilson 95% CI
difference in proportions vs A2C

Decision rule

Call a result “better” only if:
the method improves either AUC or final return, and
it does not worsen both collapse rate and unstable-update rate

No p-value chasing

Do not center the report on null-hypothesis significance testing
Center it on effect size, interval estimates, and consistency across environments

Why this choice: the literature package explicitly highlights seed variance, confidence intervals, and reproducibility. With 5 seeds, effect sizes and bootstrap intervals are more defensible than pretending precision you do not have.

Threats to validity
Implementation parity risk: small code-level differences can dominate PPO vs TRPO outcomes.
PPO-KL definition ambiguity: this is why the protocol locks one concrete PPO-KL implementation at the start.
Low sweep seed count: 3-seed sensitivity runs are useful but weaker than the 5-seed primary benchmark.
Budget truncation: a compute-aware budget may understate late asymptotic performance.
Environment-version dependence: results are only valid for the exact locked environment versions.
Estimator coupling: advantage normalization and GAE affect stability, so hidden differences there would contaminate the comparison.
Wall-clock noise: background machine load can distort timing if hardware is not controlled.
Action-distribution choice: tanh-squashed Gaussian is locked for fairness, but another policy parameterization could change results.
Small benchmark scope: four environments are enough for a course project, but not enough to claim universal superiority.

Why this choice: these are the exact risks most emphasized by the charter and literature package, especially implementation fairness and seed sensitivity.

A “Fairness Checklist” table
Item	Locked choice	Why it is locked
Algorithm roster	A2C, PPO-Clip, PPO-KL, TRPO	Proposal-defined comparison
Environment roster	Pendulum-v1, Hopper-v4, Walker2d-v4, HalfCheetah-v4	Proposal-defined benchmark
Policy architecture	Separate actor MLP, 64x64 tanh	Avoid capacity confound
Value architecture	Separate critic MLP, 64x64 tanh	Avoid critic-capacity confound
Action distribution	Same tanh-squashed diagonal Gaussian	Same policy family for all
Observation normalization	On for all methods	Cheap stability improvement, fair if shared
Reward normalization	Off for all methods	Avoid changing collapse semantics
Advantage normalization	On for all methods	Match estimator-side stabilization
Discount / GAE	gamma 0.99, lambda 0.95	Same return estimator family
Rollout batch	2048 steps/update for all	Same on-policy data budget per update
Eval cadence	Every 10,000 steps, 10 episodes	Same temporal resolution
Main seeds	5	Minimum defensible main comparison
Sweep seeds	3	Compute-aware secondary analysis
Main training budgets	100k / 300k / 300k / 300k	Matched, feasible budgets
Checkpoint metrics	Same full logging schema	Stability requires more than return
PPO-KL variant	Target-KL early stop	Charter-resolved, feasible
TRPO constraint	max KL delta = 0.02 default	Same trust-region interpretation
Shared hardware	Same machine and worker count	Wall-clock fairness
Hidden estimator controls	Same critic loss family and preprocessing	Isolate update-control effects
Reporting	Mean curves, intervals, collapse/instability counts	Avoid single-seed storytelling
A “Do not compare these unfairly” warning list
Do not compare a method’s best seed against another method’s mean.
Do not compare runs with different environment versions.
Do not compare methods trained for different total environment steps.
Do not compare methods with different rollout batch sizes and then attribute differences to trust regions.
Do not give PPO-KL both clip control and KL control unless you also give an equivalent extra safeguard to the others.
Do not let TRPO use a larger or cleaner critic-training budget than the other methods.
Do not change observation normalization, advantage normalization, or GAE for one method only.
Do not report only return and ignore collapse, unstable updates, or wall-clock.
Do not delete crashed runs from averages. Count them.
Do not interpret 3-seed sensitivity sweeps as equally strong evidence as the 5-seed main benchmark.
Do not call PPO or TRPO “more stable” if they gain return only by taking much longer wall-clock time while not improving collapse or instability.
Do not change the PPO-KL definition mid-project.