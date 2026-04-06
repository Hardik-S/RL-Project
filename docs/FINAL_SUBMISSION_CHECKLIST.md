# Final Submission Checklist

## Final paper

- Hedge every "TRPO strongest overall" sentence with the Hopper-v4 variance caveat.
- State PPO-KL precisely: unclipped ratio surrogate plus target-KL early stopping; no active clip term.
- Disclose the unstable-update threshold asymmetry: `0.04` for TRPO/PPO-KL vs `0.05` for A2C/PPO-Clip.
- State once that collapsed runs remain in all averages at their last logged evaluation return.
- Treat collapse incidence as the primary robustness metric.
- Remove any leftover mojibake or encoding issues before PDF export.

## Final repo

- Submit `RL-Project`, not the outer workspace.
- Keep the repo clean and pushed to the final commit.
- Do not rely on legacy aggregate exports in `report_assets/main_benchmark/` unless revalidated.
- Keep the submission anchored on `README.md`, `docs/final_report_draft.md`, `docs/REPRODUCIBILITY.md`, and the tracked benchmark code.

## Figures and tables

- Use only figures/tables that match the final paper narrative.
- Do not use the known-bad aggregate `final_evaluation_return*`, `instability_frequency*`, or `training_return_curves*` files without revalidation.
- Make the table note explicit: final return means the last logged evaluation checkpoint, and collapsed runs remain included.

## Citations and references

- Verify every in-text citation appears in the reference list.
- Keep citation style consistent across all eight references.
- Check author names, venues, and years after PDF export.

## Reproducibility

- Ensure `README.md` and `docs/REPRODUCIBILITY.md` match the actual commands.
- Keep the Python and MuJoCo/Gymnasium requirements explicit.
- State that per-run logs are the source of truth when aggregate outputs disagree.

## Code submission recommendation

- Submit the GitHub link: [Hardik-S/RL-Project](https://github.com/Hardik-S/RL-Project.git)
- Use a zip only as fallback if the course platform or grader cannot access GitHub.
- If submitting the GitHub link, make sure the final commit is pushed and stable.

## Most likely point losses

- Overclaiming TRPO without the Hopper variance caveat.
- Leaving PPO-KL implementation details vague.
- Using legacy aggregate outputs that contradict the paper.
- Submitting the wrong folder.
- Missing the collapse-in-averages disclosure.
