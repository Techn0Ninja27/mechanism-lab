# What was seen and changed before freezing (2026-11-18)

Everything here happened before `freeze.json` existed and used only
non-measurement seeds (98, 99, and small test seeds below 30). No measurement
seed (30–45) was run before the freeze.

## Three-layer initialisation (calibration-informed, not in the original plan)

Seed 99, three layers, inherited initialisation (normal, std 0.05), constant
learning rate, validation accuracy at epochs 40 / 80 / 120 and final loss:

| Arm | 40 | 80 | 120 | Final train loss |
|---|---:|---:|---:|---:|
| G0 | 0.250 | 0.424 | 0.542 | 0.713 |
| G1, 4 partitions | 0.054 | 0.110 | 0.098 | 1.9459 |
| G1, 16 partitions | 0.054 | 0.080 | 0.228 | 1.9459 |
| hold, 16 partitions, sync 16 | 0.054 | 0.072 | 0.284 | 1.9459 |
| lag1 (c = 0.5), 16 partitions, sync 16 | 0.144 | 0.066 | 0.152 | 1.9459 |
| sni, 16 partitions, sync 16 | 0.060 | 0.136 | 0.054 | 1.9459 |

1.9459 is ln 7: the model outputs zeros. In G1 the mean absolute weight fell
to about 4e-5 in every layer by epoch 120 while G0's grew. With signal of
order 3e-4 at initialisation, weight decay wins unless the full gradient is
present. This is an initialisation artefact, so **all three-layer arms use
Glorot-normal**. The two-layer model keeps std 0.05. Depth is therefore
confounded with initialisation and no pure depth claim can be made.

Check after the change (validation accuracy at 40 / 80 / 120):

| Seed | G0 | G1, 4 parts | G1, 16 parts | hold, 16 parts, sync 64 | zero, 16 parts (old no-warmup version) |
|---|---|---|---|---|---|
| 98 | 0.654 / 0.788 / 0.790 | 0.514 / 0.756 / 0.790 | 0.366 / 0.584 / 0.722 | 0.370 / 0.390 / 0.176 | 0.328 / 0.350 / 0.390 |
| 99 | 0.764 / 0.806 / 0.800 | 0.666 / 0.794 / 0.800 | 0.524 / 0.752 / 0.782 | 0.426 / 0.734 / 0.528 | 0.330 / 0.374 / 0.368 |

These two seeds were looked at, so the hold-at-sync-64 drop is an expectation
going in, not a result.

## Independent code review (separate agent, read-only)

Outcome: no implementation bug found in causality, gradient treatment,
seeding, evaluation, the sni recurrence or `fit_lag1` bookkeeping. The test
suite was weak: 10 of 13 injected bugs passed the original 9 tests. Eight
run-level tests were added (17 total, all pass), covering: estimates rebuilt
from sync-epoch states only at sync 4/16/64 and both hidden layers; sni
against the original recurrence and against `replay_gate` debug output;
`fit_lag1` against `Remote`; metrics and first-epoch recomputed independently
from captured weights; G1 gradient against a per-partition reference; the
learning-rate drop on partitioned arms; the never-resync control.

Semantic changes made in response:

- **zero arm now has the same warmup** as every other arm (exact for epochs
  0–20, zeros after). Before, it was zero from epoch 0, confounding warmup
  with removal.
- **lag1 extrapolation is capped at one observed interval.** At sync 64 the
  uncapped form extrapolated the epoch 0→20 transient up to three intervals.
- **lag1 uses one coefficient per hidden layer**, each fit on G0 traces of
  that layer.

Disclosed, not changed:

- sni and lag1 estimates are not clamped and can be negative for post-ReLU
  states. sni is kept identical to the original recurrence.
- In the three-layer model H2 at a sync epoch equals the true H2, but between
  syncs the locally computed H2 drifts far from it, so staleness compounds
  across layers and (G1 − hold) is not a per-layer effect.
- Layer 1 passes gradient to W1 across partition edges. That assumes cached
  halo features and shared weights synchronised every step.
- Reported "epoch N" accuracy is after step N, one step later than the
  original screen's validation number.

## Partition statistics (random, seed 321)

| Partitions | Non-self edges crossing | Edge weight crossing | Nodes with a remote neighbour | Training labels per partition |
|---|---:|---:|---:|---|
| 4 | 75.0% | 52.8% | 94.4% | 35, 27, 49, 29 |
| 16 | 93.9% | 66.1% | 99.2% | 5 to 15 |

These are extreme compared with a locality-preserving partitioner.
