# Codex audit of the closed-loop confirmation

Date: 2026-11-18. Read-only review of the frozen confirmation artifact; no
training rerun and no result files modified.

## Integrity and arithmetic

The raw file contains 64 rows, exactly the expected 8 seeds × 2 fixed regimes
× 4 arms. All `(regime, seed, arm)` keys are unique and the expected key set is
complete. Every row has the required two-cycle window, finite accuracies and
finite final loss. The raw freeze hash matches `freeze.json`, and all hashes in
the freeze match the current confirmation code, tests, plan, original gap
code, original freeze/raw artifacts, screen code and Cora inputs.

The confirmation tests compare the new loop's weights and post-update endpoint
metrics with the original loop, verify the G1 own/exact forward identity, and
check the pre-update epoch-119 interpretation. The saved summary reports all
seven integrity checks true. I independently recomputed both primary paired
comparisons from the raw windows:

- Negative control (2 layers, four partitions, sync 4): G1 − hold exact-forward
  window mean `+0.0094 ± 0.0172` points; the summary matches.
- Moderate lead (3 layers, four partitions, sync 16): G1 − hold exact-forward
  window mean `+2.7082 ± 1.1869` points; the summary matches.

With 8 paired seeds, the Student-t 95% quantile is 2.3646243. The moderate
lead's pointwise interval is approximately `[-0.098, +5.515]` points, so it
includes zero. It passes the prewritten `>0.5` and `>2 SE` allocation rule,
but this is weaker than a conventional 95% significance result. The negative
control does not pass. The endpoint comparison also matches the summary:
`+3.825 ± 1.210` points for the moderate lead.

## Semantics

The code records, before each optimizer update in the final two sync cycles,
both the logits used to train the current model and an exact forward computed
from the same weights. G1's exact remote values are detached and its own
forward equals the exact forward, as checked both in tests and in all saved
rows. Hold's own-forward accuracy differs from exact-forward accuracy after a
sync, as expected. The endpoint is the post-update exact evaluation at the
original 120-step convention; the pre-update window deliberately uses the
weights before that epoch's update. This is an internally coherent phase check.

The comparison `G1 exact − hold own-forward` is a useful descriptive
decomposition, but it is not a causal estimate of “evaluation mismatch”: G1
and hold have different learned weights and remote histories. The reported
approximately 1.0-point versus 2.7-point difference should therefore remain a
descriptive contrast, as the README says.

The confirmation does not establish that the stale model would perform worse
under an actual distributed deployment. It is a single-process emulation with
random partitions, one graph and the same detached-gradient semantics as the
first study. The locked lag1 coefficients are correctly checked against the
first study's calibration, but both extrapolators are catastrophically worse
than hold in the moderate lead; that is a negative result for these unclamped
implementations, not for prediction in general.

## Disposition

This is a valid bounded confirmation of a large exploratory lead under the
specified emulation and a null negative control under new seeds. It does not
show a statistically secure effect at the stated 95% level, prove a causal
60/40 evaluation-versus-learning decomposition, establish a new method, or
license paid proposal generation. Any follow-up should predeclare the phase,
gradient treatment, partitioning and confirmation criterion again.
