# Development control screen, 2026-11-18

All 112 planned jobs completed: seven arms, seeds 100–115. Exact A/A matched
hold for every scoring-window entry, final accuracy and training loss. Code/data hashes matched the recorded freeze immediately after the run. Later
integration edits changed some dependencies; the original freeze is preserved.
Replaying the saved analysis matched its summary, and Claude independently
recomputed the statistics and reproduced two spot runs. This was a run-bound
development screen, not an externally preregistered study. No final seeds or API calls were used.

Mean exact-forward accuracy differences from hold, averaged over pre-update
epochs 88–119, in percentage points:

| Control | Mean difference | Paired seed SD |
|---|---:|---:|
| Mean of last two syncs | +1.638 | 1.621 |
| Capped gain −0.25 | +1.249 | 1.115 |
| Capped gain −0.5 | +0.945 | 1.599 |
| Capped gain +0.1 | −1.108 | 0.782 |
| Exact-detached G1 | +2.217 | 1.821 |

The G1 variance rule written before execution proposes 32 final seeds. That is a planning
estimate, not a guarantee of candidate precision. The mean-sync baseline is the
best of these four controls on development; its improvement is not confirmed
on fresh seeds. No new control is introduced or tuned after this screen.

The practical finding is that simple averaging/damping supplies meaningful
competition to hold. A generated program beating hold alone cannot establish
an advantage over these controls. This is calibration of the engine comparison,
not a new ML method or evidence that atomization works.

`stage0-summary.json` preserves the raw result hash, frozen dependencies and
all arm statistics. Local raw records and per-job checkpoints are retained in
`results/generation-comparison-stage0-v1/`.

Mean-sync is algebraically the gain −0.5 form at horizon one, not a distinct
mechanism. It was the maximum among four controls on these selection seeds;
that choice is subject to selection bias and is fixed for final evaluation.

Secondary read-only replay (`summarize_stage0.py`) reports percentage-point
means, paired SD/SE and positive-seed counts for all three metrics in
`stage0-secondary-metrics.json`. Mean-sync improves exact-window accuracy by
1.64 points, own-window accuracy by about 0.66, and endpoint exact accuracy by
about 2.56; these describe different evaluation conditions.
