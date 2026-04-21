# Empirical Evaluation of Update Stability in Continuous-Control Policy Optimization

Canonical final submission note: `docs/final_report_submission.tex` and
`docs/final_report_submission.pdf` are the authoritative final report source and
artifact. This Markdown file is retained as an earlier draft/provenance copy.

## Abstract
Policy-gradient methods are attractive for continuous control, but they are also sensitive to update size. This project studies that sensitivity through a controlled benchmark rather than through a new optimizer. Four closely related on-policy methods were compared under a locked implementation and evaluation protocol: A2C, PPO-Clip, PPO-KL, and TRPO. The benchmark used Pendulum-v1, Hopper-v4, Walker2d-v4, and HalfCheetah-v4, with matched architecture, preprocessing, rollout size, evaluation cadence, and seed budget. On the main locomotion tasks, the step-control methods outperformed A2C under the default configuration. TRPO achieved the highest mean final return on all four environments and zero collapses across 20 runs, although the Hopper-v4 variance was large enough that fine-grained ranking there should be interpreted cautiously. PPO-Clip was the strongest first-order baseline. PPO-KL was functionally distinct because target-KL early stopping was active, but it was less reliable on Walker2d-v4. All quantitative claims in the report were recomputed from per-run `metrics.csv`, `updates.csv`, and `run_status.json` files rather than from known-bad aggregate exports.

## 1. Introduction
Policy-gradient reinforcement learning is appealing because it optimizes a stochastic policy directly and handles continuous action spaces naturally. Its practical weakness is that performance is tightly coupled to update size. If policy steps are too conservative, learning is slow; if they are too aggressive, training can degrade abruptly or collapse. That tension appears in foundational policy-gradient work and remains central in modern actor-critic systems [1,2].

TRPO and PPO are best understood as responses to that step-size problem. TRPO enforces an explicit KL trust region [4]. PPO replaces second-order trust-region machinery with first-order approximations, most commonly ratio clipping or KL-based control [6]. At the same time, the deep RL evaluation literature warns that seed choice, implementation details, and protocol drift can dominate named algorithm differences [7,8]. For a course project, that means a fair comparison is itself a technical contribution.

This project therefore does not propose a new optimizer. Instead, it develops a matched empirical benchmark designed to answer one narrow question cleanly: under shared architecture, preprocessing, rollout budget, and evaluation cadence, does explicit policy-step control improve default-setting performance and robustness relative to a standard actor-critic baseline in continuous control?

## 2. Techniques to Tackle the Problem
### 2.1 Related work and framing
Prior work gives three relevant ideas. First, REINFORCE and actor-critic methods make clear that policy-gradient learning is sensitive to gradient variance and update scale [1,2]. Second, TRPO and PPO are explicit attempts to limit destructive policy movement while retaining useful policy improvement [4,6]. Third, evaluation-focused studies show that implementation parity and reporting discipline are necessary if algorithm comparisons are to mean much at all [7,8]. The benchmark in this project follows that third lesson directly.

### 2.2 Benchmark design
The project freezes a single comparison protocol across all four methods. Every algorithm uses separate actor and critic MLPs with hidden sizes `[64, 64]`, `tanh` activations, a tanh-squashed diagonal-Gaussian policy, orthogonal initialization, actor hidden gain `sqrt(2)`, actor output gain `0.01`, critic output gain `1.0`, and initial log standard deviation `-0.5`. Observation normalization is enabled for all methods, reward normalization is disabled, advantage normalization is enabled, and GAE uses `gamma = 0.99` and `lambda = 0.95`. Value targets bootstrap on time-limit truncation but not on true terminal transitions. All methods use the same rollout batch of 2048 environment steps per update, including TRPO. Evaluation is run at step 0 and then every 10,000 environment steps for 10 episodes with `deterministic=True`, which for this Gaussian policy means using the mean action under the tanh-squashed policy. The same tanh-squash correction is used across methods when logging policy statistics.

The benchmark logs both return and stability outcomes. Run-level robustness is measured by collapse incidence under the pre-registered relative-performance rule. Update-level behavior is measured with an unstable-update diagnostic based on a KL condition plus an immediate post-update performance drop. That diagnostic is asymmetric by design: TRPO and PPO-KL are flagged when `mean_kl_old_new > 2 x 0.02 = 0.04`, whereas A2C and PPO-Clip use a fixed threshold of `0.05`. That asymmetry must be disclosed when interpreting the metric.

### 2.3 Compared methods
A2C is the baseline without explicit trust-region or proximal step control beyond ordinary learning-rate choice. PPO-Clip uses the clipped probability-ratio surrogate. PPO-KL uses the same rollout, minibatching, actor-critic architecture, and multi-epoch scaffold as PPO-Clip, but its policy loss is the unclipped importance-ratio surrogate `-(A * ratio)`, and policy epochs stop early once the mean KL exceeds the target. Because SB3 requires a `clip_range` argument, the implementation passes a non-operative bound (`1e9`) while relying on target-KL early stopping as the actual control mechanism. TRPO uses an explicit KL constraint with `max_kl_delta = 0.02` and a second-order trust-region step.

This roster is intentionally narrow. The methods are close enough that a matched comparison is meaningful, but different enough that update-control choices can still be observed.

## 3. Evaluation
All quantitative values in this section were recomputed directly from per-run `metrics.csv`, `updates.csv`, and `run_status.json` files because the aggregate analysis pipeline produced inconsistent outputs.

**Table 1. Mean final evaluation return by environment (mean +/- sd across 5 seeds), with collapsed runs in parentheses.**

| Environment | A2C | PPO-Clip | PPO-KL | TRPO |
|---|---:|---:|---:|---:|
| HalfCheetah-v4 | -1.4 +/- 1.7 (3/5) | 2243.6 +/- 506.4 (0/5) | 1852.3 +/- 450.9 (0/5) | **3578.7 +/- 1053.2 (0/5)** |
| Hopper-v4 | 243.1 +/- 44.3 (0/5) | 560.7 +/- 33.0 (0/5) | 808.0 +/- 83.0 (0/5) | **1912.1 +/- 1215.8 (0/5)** |
| Pendulum-v1 | -1185.7 +/- 116.7 (0/5) | -1206.7 +/- 120.3 (1/5) | -1110.3 +/- 137.8 (0/5) | **-782.0 +/- 118.9 (0/5)** |
| Walker2d-v4 | 321.6 +/- 17.1 (0/5) | 675.1 +/- 84.6 (0/5) | 504.9 +/- 290.0 (2/5) | **3256.2 +/- 576.8 (0/5)** |

Final return here means the last logged evaluation checkpoint for each run. Collapsed runs remain in the averages at their last logged evaluation return; none were removed from summary statistics.

Across the 80 planned runs, 74 reached the target budget and 6 terminated early under the collapse rule. Collapse counts were A2C `3`, PPO-Clip `1`, PPO-KL `2`, and TRPO `0`. Wall-clock cost preserved the expected tradeoff but did not reverse the ranking: A2C averaged `132.3 s`, PPO-KL `167.0 s`, TRPO `249.5 s`, and PPO-Clip `295.5 s`. These timings are hardware-dependent, but they still matter for an honest comparison.

### 3.1 Main findings
The main locomotion tasks support a narrow but defensible conclusion: explicit step control improved default-setting performance and run-level robustness relative to A2C. HalfCheetah-v4 and Walker2d-v4 carry most of that evidence. On HalfCheetah-v4, A2C averaged `-1.4` with `3/5` collapses, PPO-Clip and PPO-KL improved to `2243.6` and `1852.3` with zero collapses, and TRPO reached `3578.7`. On Walker2d-v4, TRPO separated even more strongly at `3256.2`, PPO-Clip reached `675.1`, PPO-KL reached `504.9` with `2/5` collapses, and A2C remained low at `321.6`.

Hopper-v4 points in the same direction but with much wider uncertainty. TRPO had the highest mean final return (`1912.1`), but the standard deviation (`1215.8`) is large enough that the TRPO versus PPO-KL margin there should not be described as statistically clean. Pendulum-v1 shows the same qualitative ordering, but it used a smaller 100k-step budget and mainly serves as a sanity-check environment rather than the main evidence.

TRPO therefore has the strongest overall empirical profile in this benchmark, but that claim should be stated precisely: it had the highest mean final return on all four environments and zero collapses across 20 runs, while Hopper-v4 still carries wide uncertainty. PPO-Clip was the strongest first-order baseline. It consistently improved on A2C in the locomotion suite and suffered only one collapse in the benchmark.

PPO-KL behaved differently from PPO-Clip in implementation terms: on the MuJoCo tasks, policy optimization stopped after only about `2.7-3.1` epochs on average rather than the nominal maximum of `10`. That shows the early-stop rule was active. It does not, by itself, prove that KL-based stopping caused the observed return differences rather than simply co-occurring with them, so the claim should remain narrow.

### 3.2 Robustness and metric disagreement
For this paper's main thesis about robustness, collapse incidence is the primary measure. It is a run-level failure outcome and directly reflects whether training remained usable. By that metric the ranking is clean: A2C was worst, PPO-Clip and PPO-KL were intermediate, and TRPO was best.

The update-level unstable-update metric tells a less tidy story. TRPO logged nonzero unstable-update counts, concentrated on Hopper-v4 and Walker2d-v4, while the other methods did not. That diagnostic should not be over-interpreted. First, it is measuring something local rather than a run-level outcome. Second, it uses a tighter KL threshold for TRPO and PPO-KL (`0.04`) than for A2C and PPO-Clip (`0.05`). The correct conclusion is therefore not that TRPO was globally unstable, but that the local instability diagnostic and the run-level collapse metric capture different phenomena. For the main benchmark claim, collapse incidence is the more decision-relevant robustness measure.

### 3.3 Threats to validity
The largest limitation is the missing hyperparameter-sensitivity sweep analysis. The benchmark therefore supports a default-setting claim much more strongly than a robustness-to-hyperparameter-variation claim. The second limitation is the aggregate analysis bug: report claims had to be rebuilt from raw per-run artifacts. Third, five seeds per condition are reasonable for a one-student project but still produce wide uncertainty on noisy tasks such as Hopper-v4. Fourth, the benchmark is intentionally small. Four environments are enough for a disciplined course comparison, but not enough to justify universal claims about policy optimization. Finally, the results remain implementation-dependent. The project intentionally uses a library-backed, fairness-controlled setup, and that is the right engineering choice here, but it limits how abstractly the ranking should be interpreted.

## 4. Conclusion
This project asked whether explicit control of policy-update size improves continuous-control training behavior under a matched benchmark. Under the locked default configuration, the answer is yes relative to A2C on the main locomotion tasks. TRPO produced the strongest overall empirical profile, combining the highest mean final return with zero collapses, although Hopper-v4 variance means that claim should be stated with caution rather than as a blanket dominance result. PPO-Clip was the strongest simpler first-order baseline. PPO-KL was active and competitive, but less reliable on Walker2d-v4. The most defensible summary is therefore modest: in this benchmark, stronger step control helped, and explicit trust-region updates worked best. The strongest next step would be to repair the aggregate pipeline fully and run the missing sensitivity sweeps.

## References
[1] Ronald J. Williams. Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning. *Machine Learning*, 8(3-4):229-256, 1992.

[2] Volodymyr Mnih, Adria Puigdomenech Badia, Mehdi Mirza, Alex Graves, Timothy P. Lillicrap, Tim Harley, David Silver, and Koray Kavukcuoglu. Asynchronous Methods for Deep Reinforcement Learning. In *Proceedings of the 33rd International Conference on Machine Learning*, 2016.

[3] Sham Kakade and John Langford. Approximately Optimal Approximate Reinforcement Learning. In *Proceedings of the Nineteenth International Conference on Machine Learning*, 2002.

[4] John Schulman, Sergey Levine, Pieter Abbeel, Michael I. Jordan, and Philipp Moritz. Trust Region Policy Optimization. In *Proceedings of the 32nd International Conference on Machine Learning*, 2015.

[5] John Schulman, Philipp Moritz, Sergey Levine, Michael I. Jordan, and Pieter Abbeel. High-Dimensional Continuous Control Using Generalized Advantage Estimation. arXiv:1506.02438, 2015.

[6] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal Policy Optimization Algorithms. arXiv:1707.06347, 2017.

[7] Peter Henderson, Riashat Islam, Philip Bachman, Joelle Pineau, Doina Precup, and David Meger. Deep Reinforcement Learning That Matters. In *Proceedings of the AAAI Conference on Artificial Intelligence*, 2018.

[8] Logan Engstrom, Andrew Ilyas, Shibani Santurkar, Dimitris Tsipras, Firdaus Janoos, Larry Rudolph, and Aleksander Madry. Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO. In *International Conference on Learning Representations*, 2020.
