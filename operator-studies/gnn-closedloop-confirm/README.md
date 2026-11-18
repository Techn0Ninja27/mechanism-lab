# Bounded confirmation and phase check: results

Date: 2026-11-18. Author: Claude; design by Codex. Plan: [PLAN.md](PLAN.md).
64 runs on new seeds 50–57, frozen by [freeze.json](freeze.json) before
running. All seven integrity checks passed. Raw: `results/raw.json`. Summary:
`results/summary.json`. Reproduce from the engine root with local Cora data and the pinned requirements:

```sh
../.venv/bin/python operator-studies/gnn-closedloop-confirm/reproduce.py --output /tmp/cora-confirm-fresh --workers 4
```

The destination must be new. Frozen inputs are hash-checked and copied into a
separate workspace before training, analysis and explicit interpretation.
Use `--prepare-only` to stage and verify inputs without repeating training.

Accuracy points on the 1,000-node development-evaluation split, paired by
seed, averaged over the final 32 epochs (two sync intervals, phase-balanced)
unless stated. Intervals are
pointwise 95% (n = 8). Rule: > 0.5 points and > 2 SE.

## Moderate lead: 3 layers, 4 partitions, sync 16

| Comparison | Mean ± SE | 95% interval | Seeds positive | Passes rule |
|---|---:|---:|---:|---|
| **Primary:** G1 − hold, exact forward | +2.71 ± 1.19 | [-0.10, +5.51] | 8/8 | yes |
| G1 − hold, hold scored on its own (stale) forward | +1.04 ± 0.40 | [+0.09, +1.98] | 7/8 | yes |
| hold own forward − hold exact forward | +1.67 ± 0.83 | [-0.28, +3.62] | 7/8 | yes |
| G1 − hold, original endpoint (exact, after epoch 120) | +3.82 ± 1.21 | [+0.96, +6.69] | 7/8 | yes |
| sni − hold, exact forward | -37.41 ± 2.10 | [-42.37, -32.45] | 0/8 | no |
| lag1 − hold, exact forward | -32.51 ± 1.80 | [-36.75, -28.26] | 0/8 | no |

Mean accuracy: G1 0.764; hold 0.737 exact and 0.754 own forward; sni 0.363; lag1 0.412.

## Negative control: 2 layers, 4 partitions, sync 4

Primary comparison +0.01 ± 0.02 points. Does not pass. Every arm sits at 0.776.

## Reading

- **The gap reappears on new seeds and passes the pre-written 2-SE
  allocation rule, but it is not confirmed at the 95% level.** The primary
  gap is 2.7 points, all eight seeds positive, and its pointwise 95% interval
  includes zero. The mean leans on one seed: per-seed gaps are 1.9, 2.3, 1.8,
  2.3, 2.4, 0.1, 0.1 and 10.7 points. Without the largest the mean is about
  1.6, and two seeds are about one node in a thousand. The 2-SE rule is the weaker criterion. (`summary.json` names
  this field `lead_confirmed_exact_forward`; read it as "passes the 2-SE
  rule".)
- **Scored on its own forward, hold loses 1.0 point instead of 2.7.** This is
  a descriptive decomposition of the observed gap, about 60% to 40%. The
  own-minus-exact difference for hold (+1.67 ± 0.83) clears the 2-SE rule by
  0.02 points and its 95% interval includes zero. It does
  not prove that 60% is caused by evaluation mismatch or that 1.0 point is
  the true underlying cost.
- **The negative control stays null.**
- **Both extrapolating predictors are far worse than hold here**, by 33 to 37
  points, in all eight seeds. They are unconstrained and can produce negative
  values for post-ReLU hidden states. The non-negative projection control was
  run afterwards in `../gnn-closedloop-clamp`: projected versions are still
  32 to 34 points below hold on average, so the failure is not explained by
  sign violations alone. This remains a statement about these two
  extrapolators, not about prediction in general.

## Phase profile of hold (mean over seeds)

| Epochs since sync | 0 | 2 | 4 | 6 | 8 | 10 | 12 | 15 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Exact forward | 0.738 | 0.755 | 0.749 | 0.732 | 0.725 | 0.728 | 0.733 | 0.739 |
| Own forward | 0.738 | 0.746 | 0.754 | 0.755 | 0.756 | 0.757 | 0.759 | 0.759 |

These are means over eight seeds with no intervals in the table. The window
is the final 32 epochs (88 to 119), not two aligned cycles: syncs fall at
epochs 100 and 116, so each phase appears twice but phases 0 to 3 are
measured on average 16 epochs later in training than phases 4 to 15. The
profile therefore mixes sync phase with training time. On its own
forward the mean rises through a cycle (one small dip at phase 13 to 14),
then is about 2 points lower at the sync epoch (2.1 ± 1.0, 7 of 8 seeds, 95%
interval includes zero), which is when its remote inputs change at once.
Under an exact forward the mean is lowest mid-cycle (phase 8 minus phase 2:
−3.0 ± 1.7, 7 of 8 seeds, interval includes zero). This is consistent with a sync acting
as a shock the model re-adapts to, which would also fit the first study's
never-resync and non-monotone observations. It does not uniquely explain
them: optimisation noise, the specific epoch-120 phase, and the compounding
of two stale layers remain open alternatives. One regime's profile, not a
tested mechanism.

## What this supports

One regime, eight seeds, one small graph, random partitions, an emulation.
Within that: a gap that passes the allocation rule on new seeds without
reaching 95% confidence, an observed 1.0-point gap when hold is scored on its
own forward, and two extrapolating predictors that are destructive with or
without projection. Depth remains confounded with initialisation. Nothing here
concerns a generated method.


## Machine-readable interpretation

The frozen summary retains the historical field `lead_confirmed_exact_forward`.
Its name is misleading: it means only that the weaker allocation rule passed.
The [hash-bound interpretation](results/interpretation.json), produced by
[interpret.py](interpret.py), explicitly records `allocation_rule_passed: true`
and `primary_positive_at_pointwise_95: false`. The original raw result and
summary remain unchanged. No automatic execution, training eligibility or
novelty follows from either record.

[Codex audit](codex-audit.md) independently checked the 64 jobs, frozen hashes,
window construction and paired arithmetic without repeating training.
