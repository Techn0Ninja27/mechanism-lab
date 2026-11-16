# Prefinal audit: utility-ranker-v1

Audit date: 2026-11-16. This is a read-only audit of the completed
train/validation artifacts. I did not call the held-out generator, open a
held-out dataset, or inspect any final result.

## Decision

The train/validation evidence is internally consistent and supports freezing
the selected model for one explicit held-out evaluation. Candidate eligibility
is defined by the declared static contracts, not by observing held-out
roundtrip labels. The result remains a narrow
compression-utility result; it is not a novelty detector, a general research
usefulness judge, or a feasibility model.

## Checks that passed

- `train-metadata.json`: 120 tasks, 2,400 candidate rows, four families with
  30 tasks each (`walk`, `runs`, `periodic`, `uniform`).
- `validation-metadata.json`: 60 tasks, 1,200 candidate rows, three unseen
  families with 20 tasks each (`clipped_noise`, `modular_ramp`,
  `alternating_burst`).
- Both splits have one unique task, stream group, and input hash per task;
  train/validation stream-group and input-hash intersections are empty.
- Both splits have zero invalid/non-lossless candidate rows. The label path
  executes each candidate, decodes it, checks exact equality with the input,
  and includes the 9-byte framing charge in the byte count.
- The 321 feature columns are computed before candidate execution from raw
  stream statistics plus predeclared chain structure and interactions. Family,
  seed, task/group ID, measured compressed size, winner/oracle label, and
  roundtrip outcome are not feature columns. Feature names and counts match
  between splits.
- The selected pairwise model was fit only on train rows and chosen by
  validation regret. The fixed best-train chain was frozen from train means;
  random and oracle comparisons are present.
- Validation pairwise-logistic selection regret is `0.0051728346`, compared
  with `0.0255565386` for fixed-best-train and `0.1136589682` for random.
  This is evidence for the preregistered narrow pilot only; it is not a
  held-out claim.
- `numerical-audit.json` reports finite features, scaled values, scores, and
  coefficients; an independent einsum score calculation agrees within
  `1.07e-14` and gives identical selections.
- A fresh, non-held-out boundary stream containing signed-int64 extrema,
  adjacent extrema, zero, and repeated extreme transitions round-tripped
  through all 20 candidate chains. All 20 passed. This supports the static
  contract argument for the fixed 4,096-element signed-int64 task domain:
  candidate execution may fail at runtime and fall back to raw, but the router
  does not use a measured held-out success/failure label to choose a candidate.

## Eligibility policy (audited)

The original concern that `selection_scores` must mask candidates using observed
held-out roundtrip outcomes is not valid for this router. Such masking would
give the router test-outcome knowledge. The candidate set is fixed and its
numeric preconditions are statically declared for signed-int64 inputs of
length 4,096; there is no `delta -> raw64` chain, and the residual/unsigned
paths stay within the codec's 128-bit scalar bound. The boundary audit above
also covers the extrema transitions that could expose these preconditions.

The correct policy is therefore: rank from raw features and predeclared chain
structure, attempt the selected chain, and fall back to raw if execution fails;
record the failure, but do not promote a failing chain or use its measured
held-out validity to alter selection. The zero invalid rows in train and
validation are consistent with this policy.

## Score-scale and numerical notes

Pairwise logistic emits a learned ordering score, not an expected gain in the
`gain = (B_raw - B_candidate) / B_raw` units. Using it only for within-task
argmax is valid; do not report its score as predicted compression gain, average
it as a gain, or compare its magnitude to ridge/ExtraTrees predictions. The
reported gains and regrets come from executable candidate labels.

The numerical audit records runtime BLAS warnings (`divide by zero`,
`overflow`, and `invalid` during matrix multiplication). They were not
suppressed or treated as evidence of failure: all stored features, scores, and
coefficients are finite, and the independent einsum calculation agrees within
`1.07e-14` with identical selections. Preserve the warning record and verify
score finiteness/einsum agreement after the frozen held-out run; if either
check fails, invalidate the study rather than silently repairing the result.
This is a reproducibility guard, not a reason to inspect held-out labels before
selection.

## Freeze requirements

1. Freeze exact source hashes for the evaluator, fixture, codec
   library, feature code, plan, and selected model.
2. Verify the held-out gate is still closed and run exactly one held-out
   evaluation with no tuning, refitting, or feature changes.
3. Independently verify score finiteness and einsum agreement after the run;
   invalidate if either check fails.
4. Report per-family regret, raw/fixed-best/random baselines, invalid counts,
   and the narrow scope statement even if the result is negative.

