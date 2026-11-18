# Closed-loop reference gap on Cora: results

Date: 2026-11-18. Author: Claude. Status: **exploratory regime-selection
development measurement.** Flagged regimes need independent confirmation on
new seeds before any claim. Plan: [PLAN.md](PLAN.md). Pre-freeze changes and
the evidence behind them: [PREFREEZE.md](PREFREEZE.md).

## What was run

1,600 training runs, 16 seeds (30–45), frozen by [freeze.json](freeze.json)
before any measurement seed was used. Raw output: `results/raw.json`
(1237.7 s wall on 8 workers). Summary: `results/summary-v2.json`,
produced by [analyse_v2.py](analyse_v2.py), which was written after the freeze
but before any result existed, in response to Codex's integrity review. All
seven integrity checks passed (raw bound to freeze, frozen files unchanged,
exact expected 1,600 keys, unique, 16 seeds, finite metrics, calibration
coefficients matched). `run.py analyse` (the frozen analysis) was not run.

Reproduce from the engine root with the pinned requirements and prepared local
Cora files:

```sh
../.venv/bin/python operator-studies/gnn-closedloop-gap/reproduce.py --output /tmp/cora-gap-fresh --workers 8
```

The destination must not exist. The wrapper verifies frozen source/data hashes,
copies the required inputs into a separate workspace, freezes that copy and
runs training plus the corrected analysis there. `--prepare-only` verifies and
stages inputs without training. CPU only; no network or model calls. The
original completion-only worker pool has no checkpoint/resume support.

Arms: G0 is centralised full-gradient training. G1 uses exact remote hidden
states with no gradient through them. hold uses the last synced value.
Differences are accuracy points on the 1,000-node development split at epoch
120, paired by seed, mean ± standard error. G0/G1 and hold accuracy columns
are proportions, not points.

## Primary result: staleness gap (G1 − hold)

`*` passes the exploratory rule (> 0.5 points and > 2 SE, unadjusted).
`**` also survives a Bonferroni correction over these 24 comparisons.

| Partitions | Layers | Schedule | G0 | G1 | Sync 4 | Sync 16 | Sync 64 |
|---:|---:|---|---:|---:|---:|---:|---:|
| 4 | 2 | constant | 0.793 | 0.768 | +0.0 ± 0.0 | +0.0 ± 0.0 | -0.0 ± 0.1 |
| 4 | 2 | drop | 0.709 | 0.675 | +0.0 ± 0.0 | +0.1 ± 0.1 | +0.7 ± 0.2 ** |
| 4 | 3 | constant | 0.785 | 0.761 | +2.9 ± 0.8 * | +2.7 ± 0.6 ** | +6.4 ± 0.6 ** |
| 4 | 3 | drop | 0.778 | 0.713 | +0.3 ± 0.1 | +1.2 ± 0.4 * | +1.9 ± 0.5 * |
| 16 | 2 | constant | 0.793 | 0.771 | +0.1 ± 0.0 | +0.1 ± 0.1 | +0.0 ± 0.3 |
| 16 | 2 | drop | 0.709 | 0.640 | +0.1 ± 0.0 | +0.4 ± 0.1 | +1.2 ± 0.4 * |
| 16 | 3 | constant | 0.785 | 0.749 | +8.4 ± 1.2 ** | +29.1 ± 3.7 ** | +25.4 ± 2.1 ** |
| 16 | 3 | drop | 0.778 | 0.629 | +1.1 ± 0.3 ** | +3.4 ± 0.6 ** | +5.7 ± 1.4 ** |

Reading:

- **Two layers, constant learning rate: no flagged gap.** All six
  comparisons are within about 0.1 points. This is the setting of the
  original stale-embedding screen and of the proposed generation pilot.
- **Two layers with a learning-rate drop: a gap only at sync 64**, about 0.7
  to 1.2 points.
- **Three layers: large gaps.** With 16 partitions and constant learning
  rate, hold loses 8, 29 and 25 points at sync 4, 16 and 64. Thirteen of 24
  regimes pass the exploratory rule and nine survive the correction; eleven of
  the thirteen are three-layer.

## Do simple predictors recover the gap? (pointwise, descriptive)

For the 13 flagged regimes. Positive means the predictor beat hold.

| Partitions | Layers | Schedule | Sync | hold | sni − hold | lag1 − hold |
|---:|---:|---|---:|---:|---:|---:|
| 4 | 2 | drop | 64 | 0.668 | +1.2 ± 0.3 | +1.4 ± 0.2 |
| 4 | 3 | constant | 4 | 0.732 | +0.6 ± 0.9 | +1.7 ± 0.9 |
| 4 | 3 | constant | 16 | 0.735 | -32.9 ± 2.8 | -27.1 ± 3.8 |
| 4 | 3 | constant | 64 | 0.697 | -9.4 ± 1.6 | -2.1 ± 0.6 |
| 4 | 3 | drop | 16 | 0.701 | -6.5 ± 1.6 | -0.0 ± 0.4 |
| 4 | 3 | drop | 64 | 0.694 | -13.9 ± 2.8 | -1.8 ± 1.1 |
| 16 | 2 | drop | 64 | 0.629 | +2.4 ± 0.5 | +2.1 ± 0.5 |
| 16 | 3 | constant | 4 | 0.665 | +1.6 ± 1.6 | +3.2 ± 1.3 |
| 16 | 3 | constant | 16 | 0.458 | -17.6 ± 4.3 | -9.0 ± 6.4 |
| 16 | 3 | constant | 64 | 0.495 | -11.7 ± 2.6 | -1.4 ± 2.0 |
| 16 | 3 | drop | 4 | 0.617 | -0.5 ± 0.2 | -1.3 ± 0.3 |
| 16 | 3 | drop | 16 | 0.595 | -2.8 ± 1.7 | +0.8 ± 0.6 |
| 16 | 3 | drop | 64 | 0.572 | -14.1 ± 3.4 | -4.2 ± 1.4 |

Reading: in the two-layer drop regimes both predictors beat hold by 1 to 2
points. In the three-layer regimes, where the gap is largest, both
extrapolating predictors are usually much worse than hold, by up to 33
points. So where staleness matters most, the SNI-style recurrence and
one-lag extrapolation as implemented here do harm. These intervals carry no
multiplicity adjustment.

## Other families (pointwise, descriptive)

| Partitions | Layers | Schedule | G0 − G1 (truncation) | G1 − never resync | G1 − zeroed |
|---:|---:|---|---:|---:|---:|
| 4 | 2 | constant | +2.5 ± 0.2 | +2.0 ± 0.3 | +3.6 ± 0.5 |
| 4 | 2 | drop | +3.4 ± 1.1 | +3.5 ± 0.5 | +5.8 ± 1.0 |
| 4 | 3 | constant | +2.4 ± 0.4 | +0.7 ± 0.3 | +3.5 ± 1.2 |
| 4 | 3 | drop | +6.5 ± 1.5 | +6.8 ± 1.3 | +16.3 ± 2.0 |
| 16 | 2 | constant | +2.2 ± 0.4 | +5.3 ± 0.8 | +9.2 ± 1.4 |
| 16 | 2 | drop | +6.9 ± 1.3 | +4.9 ± 0.7 | +7.1 ± 1.7 |
| 16 | 3 | constant | +3.6 ± 0.6 | +8.7 ± 2.0 | +19.5 ± 2.0 |
| 16 | 3 | drop | +14.9 ± 1.8 | +14.1 ± 1.8 | +27.5 ± 2.1 |

Reading: cutting the gradient through remote states costs 2 to 15 points in
every regime, more than staleness does in the two-layer regimes. Zeroing
remote messages always hurts, so the remote path is live.

## Things that should make you distrust a simple reading

- **Never resyncing is often better than resyncing rarely.** With 4
  partitions, three layers, constant rate: G1 − never-resync is 0.7 points
  but G1 − hold at sync 64 is 6.4. A model trained on frozen remote inputs
  may adapt to them, while a sync changes its inputs abruptly. Alternatively
  this is a measurement-phase effect: accuracy is taken at epoch 120, which
  falls 3, 3 and 35 epochs after the last sync for sync 4, 16 and 64, and is
  computed with an exact forward pass the stale model never trained on. This
  run cannot separate "learned worse" from "evaluated off-distribution".
- **The gap is not monotone in the sync interval** in several three-layer
  regimes (for example 29 points at sync 16 against 25 at sync 64).
- **Depth is confounded with initialisation** (PREFREEZE.md). No claim about
  depth as such.
- **Learning-rate-drop models score lower.** G0 reaches 0.709 with the drop
  against 0.793 without, for two layers. Whether that is undertraining is
  untested: there is no longer-budget control.
- **Predictors are unclamped** and can produce negative values for post-ReLU
  states. In three-layer models staleness compounds across both hidden layers.
- **Partitions are random.** 75% (4 parts) and 94% (16 parts) of edges cross.
  A locality-preserving partitioner would be much milder.
- Single-process emulation of one small graph. The 1,000-node split is now
  used development data.

## What this does and does not support

Supports, as an allocation decision: the two-layer constant-rate Cora setting
shows no demonstrated staleness gap of 0.5 points or more in this emulation,
so it is deprioritised at the current resolution. That is not a proof that
nothing can be gained there. Supports, as a lead needing
confirmation: three-layer regimes have a large gap that simple extrapolation
does not close.

Does not support: any claim that a generated mechanism is useful; any claim
about real distributed GNN training; any claim about depth.

## Next check before building on the three-layer lead

Confirm on new seeds, and separate the two explanations above by also
evaluating at sync epochs and with the stale forward the model trained on.


[Codex numerical audit](codex-independent-audit.md) checked all expected jobs,
frozen hashes, paired primary statistics and Student-t thresholds without
repeating training. Eight separate synthetic analysis tests verify rejection
of damaged inputs and the scope of primary-family intervals.
