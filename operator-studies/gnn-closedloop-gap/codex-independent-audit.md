# Independent audit of the closed-loop Cora result

Date: 2026-11-18. Reviewer: Codex. This is a read-only audit; raw and summary
files were not modified, and no training run was repeated.

## Integrity checks

The completed run contains 1,600 rows and reports 1,600 runs. Reconstructing
the expected Cartesian job set from the frozen configuration gives exactly
1,600 keys: no missing keys, extra keys, or duplicates. The arm counts are:

| arm | rows |
|---|---:|
| G0 | 64 |
| G1 | 128 |
| zero | 128 |
| hold (never resync) | 128 |
| hold/sni/lag1 at the three sync intervals | 1,152 |

All rows contain finite values at validation and evaluation checkpoints 40, 80
and 120, plus finite final training loss. The raw freeze hash matches the
current `freeze.json`; every hash in that freeze matches the current frozen
code, protocol, and Cora input files. The summary's analysis-code hash also
matches `analyse_v2.py`. Calibration coefficients in every lag1 row match the
coefficients stored in `raw.json`.

I independently recomputed the 24 primary paired comparisons (`G1 − hold`,
`eval@120`) from the 16 per-seed rows. Every reported mean and standard error
matches. SciPy gives the same Student-t quantiles used by the summary:

- 95% pointwise, df=15: 2.1314495456 (summary rounds to 2.13145).
- Bonferroni-24 threshold: 3.7128961017 (summary rounds to 3.71290).

The independent flags are 13 unadjusted exploratory comparisons and 9 that also
pass the Bonferroni-24 primary-family interval. The correction is applied only
to the 24 primary staleness comparisons. The summary correctly labels
truncation, destructive controls, predictor recovery, and `val@120` intervals
as pointwise descriptive results without multiplicity coverage.

## What the result supports

The two-layer constant-learning-rate setting has no primary flagged staleness
comparison. This is the setting used by the earlier open-loop predictor screen,
so paying to generate a predictor for that exact setting has no demonstrated
downstream-accuracy target.

The three-layer regimes show large exploratory gaps, especially with 16 random
partitions and constant learning rate: the independently recomputed primary
means are about +8.4, +29.1 and +25.4 accuracy points at sync 4, 16 and 64.
These are development-regime findings and require new-seed confirmation. The
large differences do not validate the research engine or establish a new GNN
method.

In the flagged rows, the SNI-style and lag1 predictors often lose to hold by
large margins; they do not recover the observed gap. This is a negative result
for the current predictor implementations, not evidence that prediction is
impossible.

## Material limits

Three-layer arms use Glorot initialization while two-layer arms retain the
original standard deviation 0.05. The pre-freeze record documents why this was
necessary to avoid a zero-output collapse, but depth and initialization remain
confounded. No depth claim is warranted.

The partitions are random and extreme (roughly 75% or 94% of non-self edges
cross partitions), so the result is not a distributed-systems benchmark. G0 is
a centralized full-gradient reference; G1 uses exact remote values but detaches
their gradient. This separates a truncation effect from stale-value effects,
but G1 is not full fresh distributed training.

The evaluation is after 120 training steps with an exact full forward pass,
while stale models trained on their stale forward path. At sync 4 and 16 the
last sync is at internal epoch 116; at sync 64 it is at internal epoch 84.
Thus the apparent non-monotonicity and large late gaps can include an
off-distribution evaluation effect. The run does not evaluate at sync epochs or
with the stale forward used during training, so it cannot separate those
effects from learned-model degradation.

The 24-regime search is exploratory. The Bonferroni adjustment covers only the
predeclared primary staleness family, not all reported controls, predictors,
metrics, or regimes jointly. No result here is an independent confirmation;
the flagged regimes need a fresh confirmation run with the evaluation protocol
fixed in advance.

## Audit disposition

The files are internally consistent and the summary is numerically reproducible
from the saved rows. I recommend retaining the result as a bounded negative/
diagnostic study: the original two-layer target offers no measured accuracy
headroom, while a calibration-informed three-layer regime is a lead with strong
confounds. It does not justify paid generation or a claim of useful generated
research proposals yet.
