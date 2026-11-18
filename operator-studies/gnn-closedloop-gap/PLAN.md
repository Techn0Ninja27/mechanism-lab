# Closed-loop reference gap on Cora: plan (Claude, 2026-11-18)

Status: plan reviewed by Codex (message c8bf0c5a) and revised. Frozen by
`freeze.json` when that file exists. CPU only, no downloads, no API calls.
Seeds 10–15 are never used.

A first rough run exists in `reports/ml-usefulness-review-claude-v1/`
(seeds 20–27). It is a pilot of the method: its reference arm was
fresh-detached (G1 below) and it had no full-gradient arm.

## Question

In which Cora regimes, if any, does stale remote hidden-state information
measurably change trained-model accuracy? This is a **regime-selection
development measurement**. It selects a task regime by reference gap. It does
not tune any candidate or hyperparameter, and it is never informed by
candidate outcomes. Any regime it flags needs independent confirmation on new
seeds before a claim is made.

## Semantics

Data, split (140 train, 500 validation), optimiser (Adam, lr 0.01, weight
decay 5e-4), 120 epochs and both learning-rate schedules come from
`operator-studies/gnn-sync-gate/screen.py`, imported unmodified. Hidden width
16. Random partitions, seed 321.

**Two-layer model.** `H1 = relu(A X W1)`, `Z = A H1 W2`. Initialisation as in
`screen.py` (normal, std 0.05).

**Three-layer model.** `H1 = relu(A X W1)`, `H2 = relu(A H1 W2)`,
`Z = A H2 W3`. Glorot-normal initialisation. Reason, found before freezing on
non-measurement seed 99: with std 0.05 a three-layer model sits at a
collapse point. Centralised training reached only 0.54 validation accuracy in
120 epochs and every partitioned arm decayed to all-zero weights (loss exactly
ln 7). That is an initialisation artefact, not a staleness effect, so the
three-layer model uses a standard initialisation. The two-layer model is
unchanged and reproduces `screen.train_trajectory` exactly.

**Layer 1** consumes raw neighbour features, which are static and cacheable,
so it is exact with full local gradient in every arm.

**Message layers that consume hidden states** (layer 2; and layer 3 in the
three-layer model). For node i at layer k:
`agg_i = Σ_same-partition A_ij H_j W_k + Σ_other-partition A_ij Ĥ_j W_k`.
`Ĥ` is a remote estimate with no gradient. The local weights `W_k` are
applied to it and do receive gradient through that term. Each hidden layer
has its own buffer. In the three-layer model "exact H2" means the H2 the
model actually computed that epoch, which itself used the layer-1 estimate.

**Gradient, three explicit levels.**
- G0: centralised, gradient through every hidden state. A full-gradient
  reference. No ordering is assumed between G0 and the other arms.
- G1: `Ĥ` is the exact current hidden state, detached. Isolates gradient
  truncation. Never described as full fresh training.
- Stale arms: `Ĥ` from a buffer, detached.
- Reported decomposition: G0 − G1 (truncation); G1 − hold (staleness);
  predictor − hold (what a predictor recovers).

**Sync.** Epochs 0–20 are warmup: `Ĥ` exact. After that a sync delivers the
exact current value at epochs t with (t − 20) mod s = 0, s ∈ {4, 16, 64}.
Between syncs an estimator never reads the exact value.

**Stale arms.**
- hold: value at the last sync.
- sni: the self-fed damped-trend recurrence of `screen._predict/_observe`
  with its constants (0.2, 0.2).
- lag1: `last + c · (last − prev) · min(1, elapsed / (t_last − t_prev))`.
  The horizon is capped at one observed interval. One scalar c per (layers,
  schedule, sync, hidden layer), grid-fit on open-loop G0 traces of that
  layer from calibration seeds 20–21.
- In closed loop the predictor changes the trajectory. That is intended.

**Destructive controls.** Hold never resynced after warmup; remote messages
zeroed after the same warmup.

Changes made before freezing, with the evidence that prompted them, are in
[PREFREEZE.md](PREFREEZE.md).

## Metrics

After every optimiser step an exact full forward pass gives accuracy. Recorded
at epochs 40, 80, 120: validation (500 nodes) and a second, disjoint
1,000-node split from the same shuffle. **That second split uses labels that
were previously untouched, so it is consumed here as development data and is
no longer available as a test set.** Primary metric: second-split accuracy at
epoch 120. Also: first epoch with validation ≥ 0.75; final training loss. Only
the 140 training labels enter any loss.

## Seeds and runs

Calibration 20–21 (lag1 fit only). Measurement 30–45, paired across arms by
seed. G0 does not depend on partitions or sync and is run once per
(seed, layers, schedule). G1 and zero are run once per partition count.
Total 1,600 training runs at about 2.7 s each: roughly 70 CPU-minutes, about
10 minutes on 8 workers.

## Regime grid (declared before running)

Partitions {4, 16} × sync {4, 16, 64} × layers {2, 3} × schedule
{constant, drop at 60} = 24 staleness comparisons.

## Rule and multiplicity (written before running)

For each regime report the paired mean of (G1 − hold) on the primary metric,
its standard error and 95% interval, in accuracy points.
- Exploratory flag: mean > 0.5 points and mean > 2 SE. **Not adjusted for 24
  comparisons.**
- Simultaneous flag: mean > 0.5 points and the Bonferroni-adjusted interval
  over the 24 comparisons excludes zero.
Flags mark regimes worth confirming, nothing more. If no regime carries even
the exploratory flag, Cora is recorded as "no measurable staleness gap in the
tested regimes" and usefulness claims are closed for this task. That is an
allocation decision, not a proof of no effect.

## Tests required before freezing (`test_gap.py`)

G0 two-layer equals the original trajectory bit for bit, both schedules;
single-partition G1 equals G0; G1 has the same forward values as G0 but a
different first-layer gradient and an identical last-layer gradient; any
stale arm at sync 1 equals G1 exactly; hold uses only the last sync value at
both hidden layers; estimators ignore the exact value between syncs; sni
matches the original recurrence; splits are disjoint and match the original.
Run-level: estimates rebuilt from sync-epoch states at sync 4/16/64; metrics
recomputed independently from captured weights; G1 gradient against a
per-partition reference; learning-rate drop on partitioned arms. 17 tests.

## Immutability

`run.py freeze` records hashes of code, plan, original `screen.py`, Cora files
and the configuration. `run.py run` refuses to start if any hash differs and
refuses to overwrite `results/raw.json`. `run.py analyse` refuses to overwrite
`results/summary.json`.

## Limits known in advance

Single-process emulation, not a distributed implementation. One small graph.
Random partitions, not METIS. Accuracy under an exact forward pass assumes
inference-time sync. Two different initialisations across depths, so depth
comparisons are confounded with initialisation.
