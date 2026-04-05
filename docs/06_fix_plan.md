# 06 Fix Plan

This plan is based on the Sonnet audit, [03_experiment_protocol.md](C:\Users\hshre\OneDrive\Documents\42%20-%20Agents\Codex\4453\Project\03_experiment_protocol.md), [04_algorithm_spec.md](C:\Users\hshre\OneDrive\Documents\42%20-%20Agents\Codex\4453\Project\04_algorithm_spec.md), [05_build_manifest.md](C:\Users\hshre\OneDrive\Documents\42%20-%20Agents\Codex\4453\Project\05_build_manifest.md), [ANALYSIS_README.md](C:\Users\hshre\OneDrive\Documents\42%20-%20Agents\Codex\4453\Project\ANALYSIS_README.md), and the current `src/` and `tests/` snapshot. No standalone file literally named "current repo snapshot summary" exists in the workspace, so the live code plus the build manifest were treated as the snapshot source of truth.

## 1. Which criticisms are valid and must be fixed now

- [Engineering] PPO-KL still carries a finite `clip_range=10.0` in [src/algos/ppo_kl_runner.py](C:\Users\hshre\OneDrive\Documents\42%20-%20Agents\Codex\4453\Project\src\algos\ppo_kl_runner.py). This must be fixed now because the protocol and build manifest explicitly freeze PPO-KL as `ratio_clipping: false`, and the central benchmark claim depends on PPO-KL not being presented as a clip+KL hybrid. Tradeoff: this is a small code change with large fairness value.

- [Engineering] PPO-KL update bookkeeping is misaligned with the benchmark’s update-level analysis. In [src/algos/ppo_kl_runner.py](C:\Users\hshre\OneDrive\Documents\42%20-%20Agents\Codex\4453\Project\src\algos\ppo_kl_runner.py), `_n_updates` is advanced by `epochs_completed`, and a partially completed epoch is counted as completed. This must be fixed now because `updates.csv`, `update_index`, and any cross-method instability alignment depend on one update meaning one rollout update. Tradeoff: low implementation cost, high downstream analysis clarity.

- [Engineering] TRPO currently calls `set_optimizer_lrs(self.policy, self.critic_lr, self.critic_lr)` in [src/algos/trpo_runner.py](C:\Users\hshre\OneDrive\Documents\42%20-%20Agents\Codex\4453\Project\src\algos\trpo_runner.py). This must be fixed now because it is a real parity error, not just a wording issue. TRPO should not silently assign an actor-side LR that shadows the critic-only design. Tradeoff: minimal code change, removes an avoidable implementation confound.

- [Engineering] The unstable-update KL threshold is hardcoded in [src/metrics/stability.py](C:\Users\hshre\OneDrive\Documents\42%20-%20Agents\Codex\4453\Project\src\metrics\stability.py) as `0.04` for PPO-KL and TRPO. This is valid and must be fixed now because the sweep plan varies `target_kl` and `max_kl_delta`; keeping a fixed threshold corrupts sweep instability counts. Tradeoff: slightly more plumbing through config or update metadata, but it protects a headline metric.

- [Engineering] Fairness-critical architecture and optimizer parity are not covered by tests. The code in [src/policies/actor_critic.py](C:\Users\hshre\OneDrive\Documents\42%20-%20Agents\Codex\4453\Project\src\policies\actor_critic.py) does create separate actor/critic parameter groups, but there is no regression test proving that all four runners instantiate the same 64x64 tanh separate actor/critic design. This must be fixed now because parity is the benchmark’s main validity risk. Tradeoff: modest test work now prevents silent drift later.

- [Engineering] PPO-KL runner cleanup is still needed even after the bigger fixes. Specifically, `self.policy.set_training_mode(False)` is missing at the end of `train()`, and the epoch logging field currently overstates completed optimization when early stopping breaks mid-epoch. These are small, local fixes and should be bundled into the PPO-KL repair pass instead of left half-done. Tradeoff: tiny scope increase, cleaner runner semantics.

- [Engineering] Hardware provenance for wall-clock claims is under-logged. [src/algos/common.py](C:\Users\hshre\OneDrive\Documents\42%20-%20Agents\Codex\4453\Project\src\algos\common.py) writes `run_metadata.json`, but it does not capture hostname, CPU, or CUDA device details. This should be fixed now if any final timing runs remain, because wall-clock is a first-class outcome in the protocol. Tradeoff: cheap to add now, hard to reconstruct later.

## 2. Which criticisms are valid but can be handled in the report limitations section instead

- [Framing] A2C gets one critic update pass per rollout while PPO-Clip, PPO-KL, and TRPO get ten critic epochs. This is a real confound, but it is frozen by the protocol rather than a coding mistake. Do not expand scope with a new ablation now; handle it as a limitations note and avoid attributing all A2C instability purely to lack of policy-step control.

- [Framing] The logged KL in [src/algos/_sb3_helpers.py](C:\Users\hshre\OneDrive\Documents\42%20-%20Agents\Codex\4453\Project\src\algos\_sb3_helpers.py) is a post-update approximate KL computed on the rollout buffer, not the exact same minibatch statistic that triggers PPO-KL early stopping. That is a valid caveat, but it does not invalidate the benchmark if described accurately. Report it as “logged update diagnostic,” not “the exact stopping signal.”

- [Framing] Collapse detection and unstable-update counting have limited early-training sensitivity because evaluation happens every 10,000 steps and the instability rule requires enough evaluation history. This matters most on Pendulum and early checkpoints, but it follows from the frozen protocol. Treat it as a resolution/power limitation, not a code emergency.

- [Framing] AUC in [src/analysis/aggregate.py](C:\Users\hshre\OneDrive\Documents\42%20-%20Agents\Codex\4453\Project\src\analysis\aggregate.py) is unnormalized reward-by-steps area, so it is useful within an environment but not for cross-environment aggregation. The report should explicitly avoid “overall AUC across all environments” language rather than changing the metric late in the project.

- [Framing] Sweep evidence is weaker than main-benchmark evidence because sweeps use 3 seeds. This is valid, but it belongs in interpretation language, not in a new engineering branch.

- [Framing] Collapse-rate comparisons with 5 seeds carry wide uncertainty. This is valid and should be handled with Wilson intervals and cautious wording rather than extra implementation.

- [Framing] TRPO has a narrower sweep surface than the other methods because there is no actor LR sweep. That asymmetry is defensible and should be disclosed, not “fixed.”

## 3. Which criticisms should be rejected as overkill

- [Reject] Rewriting PPO-KL to check KL only after `optimizer.step()` should not be treated as mandatory. The current logic is conservative, but it is not clearly inconsistent with the frozen “stop remaining policy epochs once mean KL exceeds the target” rule. Changing this now would alter algorithm behavior more than it protects fairness.

- [Reject] Replacing the approximate KL statistic with a full true-KL computation is overkill for this project. The protocol and common practice are already built around the Schulman approximation, and the right fix is wording, not metric reinvention.

- [Reject] The eval loop’s single-env termination handling in [src/callbacks/eval_callback.py](C:\Users\hshre\OneDrive\Documents\42%20-%20Agents\Codex\4453\Project\src\callbacks\eval_callback.py) is fragile in principle, but this benchmark uses a one-env `DummyVecEnv` for evaluation. It is not the right place to spend scope unless evaluation actually breaks.

- [Reject] The `ENV_ID_BY_KEY` map in [src/envs/make_env.py](C:\Users\hshre\OneDrive\Documents\42%20-%20Agents\Codex\4453\Project\src\envs\make_env.py) is a maintainability annoyance, not a current validity problem. The env roster is frozen and already tested. Do not expand the plan with config-schema refactors.

- [Reject] Guarding `range_so_far <= 0` in the instability rule should be rejected unless the protocol itself is amended. The current implementation follows the literal benchmark rule closely; changing it now would be a definition change, not a bug fix.

- [Reject] Treating step-0 normalization as a benchmark-breaking flaw is too strong. Step-0 evaluation effectively uses identity normalization because no training statistics exist yet, which is acceptable and symmetric across methods for the baseline checkpoint.

- [Reject] Adding a new A2C “10 critic passes” ablation is good science but bad scope for this deliverable. Keep the protocol fixed and disclose the confound instead.

## 4. Exact code-change tasks for Codex

1. Update [src/algos/ppo_kl_runner.py](C:\Users\hshre\OneDrive\Documents\42%20-%20Agents\Codex\4453\Project\src\algos\ppo_kl_runner.py) so PPO-KL no longer carries a misleading finite clip fuse.
   Use a clearly non-operative clip setting only because SB3’s PPO constructor requires one, and add an inline comment stating that PPO-KL’s policy loss is the plain ratio surrogate with `ratio_clipping: false`.
   Add a local assertion in `build_model()` that the resolved config still has `early_stop_on_target_kl: true` and `ratio_clipping: false`.

2. Fix PPO-KL update accounting in [src/algos/ppo_kl_runner.py](C:\Users\hshre\OneDrive\Documents\42%20-%20Agents\Codex\4453\Project\src\algos\ppo_kl_runner.py).
   Increment `_n_updates` by exactly 1 per `train()` call.
   Track `epochs_fully_completed` separately from “entered epoch.”
   Log `epochs_completed_before_early_stop` as fully completed epochs only.
   Keep `early_stopped` as a separate boolean.

3. Finish PPO-KL mode cleanup in [src/algos/ppo_kl_runner.py](C:\Users\hshre\OneDrive\Documents\42%20-%20Agents\Codex\4453\Project\src\algos\ppo_kl_runner.py).
   Ensure `self.policy.set_training_mode(False)` is called before returning from `train()`, including the early-stop path.

4. Fix TRPO optimizer handling in [src/algos/trpo_runner.py](C:\Users\hshre\OneDrive\Documents\42%20-%20Agents\Codex\4453\Project\src\algos\trpo_runner.py).
   Stop assigning an actor LR equal to `critic_lr`.
   Either skip `set_optimizer_lrs()` entirely for TRPO or add a TRPO-specific helper that updates only the critic parameter group if the optimizer layout exposes one.
   Add a short comment documenting why TRPO is treated differently from A2C/PPO.

5. Parameterize the instability KL threshold in [src/metrics/stability.py](C:\Users\hshre\OneDrive\Documents\42%20-%20Agents\Codex\4453\Project\src\metrics\stability.py) and [src/algos/common.py](C:\Users\hshre\OneDrive\Documents\42%20-%20Agents\Codex\4453\Project\src\algos\common.py).
   Extend `StabilityInputs` to carry a `nominal_kl_budget`.
   For PPO-KL use `config["algo"]["target_kl"]`.
   For TRPO use `config["algo"]["max_kl_delta"]`.
   Keep A2C and PPO-Clip on the fixed `0.05` threshold.
   Use `2 * nominal_kl_budget` only for the constrained methods.

6. Add fairness tests in `tests/`.
   Create a new test module that instantiates each runner with a tiny config and verifies: separate actor/critic parameter groups exist, both networks are `[64, 64]`, activation is `tanh`, `share_features_extractor` is `False`, and `log_std_init` is `-0.5`.
   Add a PPO-KL-specific test that asserts the resolved config path is `ratio_clipping: false` and that the runner does not report `clip_fraction`.
   Add a TRPO-specific test that asserts the optimizer-LR patch does not overwrite actor-side settings.

7. Extend run metadata in [src/algos/common.py](C:\Users\hshre\OneDrive\Documents\42%20-%20Agents\Codex\4453\Project\src\algos\common.py).
   Include hostname, platform, Python version, CPU string, torch version, CUDA availability, and CUDA device name when present.
   Keep the format simple JSON so existing analysis code does not break.

8. Add one small analysis guard in [src/analysis/aggregate.py](C:\Users\hshre\OneDrive\Documents\42%20-%20Agents\Codex\4453\Project\src\analysis\aggregate.py) or the reporting layer.
   Do not create or describe any cross-environment aggregate based on raw AUC.
   If there is already a summary sentence doing that later, remove it rather than inventing a normalized AUC metric now.

## 5. Exact report-language changes to avoid overclaiming

- Replace “PPO-KL is a clip-free PPO variant” with:
  “PPO-KL uses the same rollout and minibatch scaffold as PPO-Clip, but its policy update is controlled by a target-KL early-stop rule rather than PPO-Clip’s clipped surrogate.”

- Replace “the comparison isolates clipping versus KL control” with:
  “the benchmark is designed to contrast clipped-surrogate control with target-KL early stopping under a matched actor-critic setup.”

- Replace “constrained methods are more stable than A2C” with:
  “in this matched benchmark, the constrained-update methods generally showed fewer instability events than A2C, subject to critic-update asymmetry and small-sample uncertainty.”

- Replace any sentence implying A2C is unstable purely because it lacks trust-region-style control with:
  “A2C also differs in critic optimization intensity, so the results should be interpreted as a benchmark-level comparison of matched implementations, not a pure single-factor causal identification.”

- Replace “the sweep shows robustness” with:
  “the 3-seed sweeps provide directional robustness evidence only.”

- Replace “collapse rates differ meaningfully” with:
  “collapse-rate differences should be read with their Wilson intervals; with 5 seeds, small count differences are not strong evidence on their own.”

- Replace any sentence that treats logged KL as exact KL divergence with:
  “KL statistics are approximate update diagnostics computed from old-versus-updated policy log-probabilities on the rollout data.”

- Replace any sentence that compares AUC across environments directly with:
  “AUC is interpreted within each environment; cross-environment comparisons rely on per-environment tables rather than a pooled AUC.”

- Add one explicit limitations sentence:
  “The evaluation cadence and short Pendulum budget reduce sensitivity to very early collapses or instability events, so these metrics are better interpreted as coarse benchmark diagnostics than exhaustive failure detectors.”

- Add one explicit wall-clock sentence:
  “Wall-clock comparisons are only interpreted for runs produced on the same hardware and software stack.”

## 6. Save as 06_fix_plan.md

Use this document as the gate before any final reruns and before locking report claims. The engineering fixes above protect benchmark validity; the framing fixes prevent the write-up from claiming more than the implementation can support.
