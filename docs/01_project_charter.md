One-sentence project thesis
This project will test whether explicitly limiting policy-update size, through trust-region or proximal constraints, yields more stable and more robust learning than a standard actor-critic baseline when all methods are run under a matched continuous-control setup.
Research question
Under the same policy/value architecture, training budget, and evaluation protocol, do TRPO and PPO-style constrained updates reduce instability and collapse, while preserving competitive return, relative to A2C in continuous-control environments? The proposal already defines the comparison as an empirical study of update-size sensitivity and robustness, not a search for a novel algorithm.
Primary hypothesis
Methods that explicitly constrain policy movement in policy space, rather than relying mainly on a raw learning-rate choice, will show lower instability frequency and lower collapse frequency than A2C, especially when update-size hyperparameters are made more aggressive. This follows the slide framing that policy-gradient updates are sensitive to step size, that small steps are slow but reliable, and that trust regions are introduced because small changes in policy tend to induce smoother value changes than small changes in parameter space.
Secondary hypotheses
PPO with clipping will usually recover much of the stability benefit of trust-region ideas while being simpler and cheaper to optimize than TRPO.
TRPO will likely have the cleanest protection against overly large policy steps, but it may pay for that with higher implementation complexity and wall-clock cost.
A2C will be the most sensitive baseline to actor step-size choice, because the slides explicitly frame ordinary policy-gradient style updates as hard to tune and potentially unreliable when the step is large.
Across environments, the ranking on raw return may vary, but the ranking on robustness should favor TRPO and the PPO variants over A2C. This is directly aligned with the proposal’s emphasis on robustness to update-size choices rather than only best-case score.
Fixed algorithm set
Use exactly these four algorithms from the proposal: A2C, PPO with clipping, PPO with KL-based control, and TRPO.
Use a shared stochastic Gaussian policy parameterization for all four methods, since the course slides frame continuous-action policy optimization using parameterized stochastic policies and actor-critic style updates.
Recommended concrete definitions, because the proposal is under-specified here:
A2C: shared actor-critic backbone with value baseline and advantage-based policy update.
PPO-Clip: clipped probability-ratio surrogate exactly in the spirit of the lecture derivation.
PPO-KL: the proposal names “PPO with KL-based control,” but the uploaded slides only derive clip-PPO, not a specific PPO-KL variant. Recommended implementation: PPO with the same actor-critic backbone and a target-KL control rule that stops policy epochs early once mean KL exceeds a target. This keeps the course’s KL/trust-region framing while staying feasible for one student.
TRPO: explicit KL-constrained policy update with a shared critic/value baseline.
Fixed environment set
Use the four environments named in the proposal: Pendulum, Hopper, Walker2d, and HalfCheetah.
Recommended concrete instantiation, because the proposal is under-specified on exact versions: Pendulum-v1, Hopper-v4, Walker2d-v4, and HalfCheetah-v4, all from one consistent Gymnasium/MuJoCo stack.
Recommended role of each environment: Pendulum for quick debugging and sanity checks, Hopper as an instability-sensitive locomotion task, Walker2d as a harder biped task, and HalfCheetah as a standard high-dimensional locomotion benchmark. This stays within the proposal while keeping the experiment portfolio manageable.
Fixed evaluation metrics
Use the proposal’s core metrics as the fixed reporting set: average return over training, variance across random seeds, wall-clock cost, and frequency of unstable updates or performance collapse.
Recommended concrete metric definitions:
Learning curve metric: mean evaluation return at fixed environment-step checkpoints
Sample-efficiency metric: area under the evaluation-return curve up to the fixed step budget
End-performance metric: mean of the last 10 evaluation checkpoints
Seed-robustness metric: standard deviation across seeds at each checkpoint plus final standard deviation
Compute metric: wall-clock seconds to reach the full budget and, if a threshold is reached, time-to-threshold
Stability metric: unstable-update rate as defined in section 10
Collapse metric: collapse rate as defined in section 11
This adds clarity without changing the proposal’s intended evaluation emphasis.
Controlled variables that must stay matched
These must remain fixed across algorithms unless the algorithm mathematically requires otherwise:
same environment versions and wrappers
same observation preprocessing and reward preprocessing choices
same policy network architecture
same value network architecture
same activation functions and initialization scheme
same action distribution family
same rollout budget measured in environment steps
same evaluation frequency and number of evaluation episodes
same random seeds for the matched comparison
same logging of policy KL, return, runtime, and failure events
same hardware for wall-clock comparisons
This is strongly supported by the proposal’s “matched setup,” “same network architecture,” “similar training budgets,” and “same evaluation protocol wherever possible.”
Recommended extra matching choice, because the proposal is under-specified: use the same critic loss form, discount factor, and advantage-normalization rule wherever the code path allows, so that the comparison isolates update control rather than unrelated estimator differences. This is consistent with the course emphasis on baseline and TD variance reduction in actor-critic methods.
Independent variables to sweep
The proposal explicitly says to vary learning rate and trust-region-related hyperparameters.
Recommended sweep set:
Shared actor learning-rate / step-size scale: low, medium, high
A2C: actor learning rate only
PPO-Clip: actor learning rate and clip parameter epsilon
PPO-KL: actor learning rate and target KL threshold
TRPO: trust-region radius delta
Recommended concrete three-level grids for a one-student project:
actor LR: {1e-4, 3e-4, 1e-3}
PPO clip epsilon: {0.10, 0.20, 0.30}
PPO target KL: {0.01, 0.02, 0.05}
TRPO delta: {0.01, 0.02, 0.05}
Recommendation: do not full-grid every variable on every environment. Use one-factor-at-a-time robustness sweeps after the matched default comparison. That preserves feasibility.
Recommended definition of unstable update
Recommended operational definition: an update block is unstable if both of the following happen:
the mean KL between the pre-update and post-update policy, measured on the rollout batch, is abnormally large for that method, and
the next evaluation return drops sharply relative to the run’s recent trajectory.
Recommended concrete rule: label update block 
u
u unstable if
KL condition: mean KL exceeds 2 × the algorithm’s nominal target/budget, or exceeds 0.05 for methods without an explicit KL target, and
performance condition: the next evaluation return is at least 20% of the run’s achieved improvement range below the trailing 3-checkpoint mean.
This definition is not given in the proposal, so this is a recommendation. It is motivated by the proposal’s concern with unstable updates and by the slide framing that KL-constrained methods are meant to control policy movement.
Recommended definition of performance collapse
Recommended operational definition: a run has collapsed if, after previously making substantial progress, it falls back near its early-training level and stays there.
Recommended concrete rule: let R_init be the mean of the first 3 evaluation checkpoints and R_best the best checkpoint seen so far. Mark collapse if, after the run has once exceeded R_init + 0.60*(R_best - R_init), it later stays below R_init + 0.25*(R_best - R_init) for 5 consecutive evaluations.
Also count immediate numeric failure as collapse: NaN losses, NaN action statistics, or unrecoverable divergence.
This is a recommendation because the proposal names collapse but does not define it. The relative-to-own-progress rule is better than a fixed return threshold because the environments have different reward scales.
Biggest implementation risks
The largest risk is PPO-KL ambiguity. The proposal names it, but the uploaded slides only directly derive clip-PPO. You need to lock one concrete PPO-KL definition at the start and never change it mid-project.
The next major risk is TRPO implementation complexity, especially the constrained update, conjugate-gradient solve, and line search. The slides themselves frame TRPO as conceptually and computationally challenging because of the constraint.
A third risk is unfair comparison through hidden estimator differences. If advantage estimation, normalization, reward scaling, or batch construction differ across methods, the project stops being about update stability. This matters because the course framing distinguishes baseline and TD-based variance reduction from pure Monte Carlo estimation.
A fourth risk is wall-clock unfairness. TRPO may be slower per update, so you must report both sample-based and time-based outcomes. The proposal explicitly asks for wall-clock cost.
A fifth risk is small-seed overinterpretation. The proposal references variance across seeds, so seed count is not optional.
A sixth risk is MuJoCo/tooling friction, which can derail a one-student project if environment setup is postponed.
Minimum viable experiment plan
Implement all four algorithms in one shared codebase with one shared actor/critic backbone and one shared logging/evaluation pipeline.
Run the full four-way comparison first on Pendulum and Hopper only, with 3 seeds each, at one default hyperparameter setting per algorithm. This gives an easy environment and a more instability-sensitive locomotion environment while staying feasible. The proposal already positions Pendulum/Hopper inside the chosen task family.
Then run one small robustness sweep: vary actor LR across low/medium/high for all four methods on Hopper only.
Deliverables from the MVP:
learning curves
final return table
seed variance table
wall-clock table
unstable-update counts
collapse counts
If time gets tight, this is enough for a valid report because it still directly answers the proposal’s central robustness question.
Strong final experiment plan
Stage 1: matched default comparison on all four environments with 5 seeds per algorithm.
Stage 2: robustness sweeps on two representative environments only, recommended as Hopper and HalfCheetah.
Recommended sweep design for feasibility:
A2C: 3 actor LRs
PPO-Clip: 3 actor LRs, then 3 clip epsilons at the default LR
PPO-KL: 3 actor LRs, then 3 target-KL values at the default LR
TRPO: 3 trust-region deltas
This is intentionally not a full Cartesian grid. It isolates the effect of update aggressiveness without exploding run count.
Stage 3: ablation-style summary focused on the actual thesis: which method degrades least when updates are loosened?
Strong final figures:
mean learning curves with seed bands
instability-event rate by method
collapse-rate bar chart
wall-clock vs return scatter
sensitivity plots for LR / clip / target-KL / delta
This plan remains feasible for one student while staying faithful to the proposal’s matched-comparison and sensitivity-analysis structure.
A week-by-week execution plan
Week 1: finalize exact algorithm definitions, environment versions, logging schema, and codebase structure; get Pendulum running end to end. The key output is one reproducible training script with evaluation checkpoints and KL logging.
Week 2: implement and validate A2C and PPO-Clip; verify that returns improve on Pendulum and that logged KL / ratio statistics look sensible. This follows the actor-critic and PPO lecture framing.
Week 3: implement PPO-KL and TRPO; run small sanity checks on Pendulum and Hopper; lock the instability and collapse definitions before large runs.
Week 4: run the minimum viable comparison on Pendulum and Hopper with 3 seeds; debug failures; decide default hyperparameters for the strong final run.
Week 5: run the full matched comparison on all four environments with 5 seeds; start plotting return, variance, wall-clock, and event counts.
Week 6: run the robustness sweeps on the two representative environments; draft the report around the already-fixed charter; write the conclusion around stability/robustness, not just best score.
If you have less time than 6 weeks, compress Weeks 4 to 6 and protect the MVP first.
Non-negotiables
Do not change algorithm definitions mid-project.
Do not let network architecture differ across methods.
Do not compare different environment versions under the same environment name.
Do not report only the best seed. Report all seeds and summary statistics.
Do not let evaluation frequency vary by algorithm.
Do not tune one method far more heavily than the others outside the declared sweep budget.
Do not hide crashed or collapsed runs. Count them.
Do not compare only return; always include wall-clock and instability/collapse metrics.
Do not mix different preprocessing, normalization, or reward-scaling schemes across methods unless mathematically unavoidable.
Do not add more environments or more algorithms until the four proposed methods run correctly on Pendulum and Hopper. This is how the project stays feasible for one student.