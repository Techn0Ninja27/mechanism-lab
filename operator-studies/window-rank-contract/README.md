# Window-rank contract diagnostic

Recorded 2026-11-16 for the graph-generated FIER → WindowQuant transfer. This
is a CPU-only synthetic contract check. It does not establish VLM usefulness,
novelty, accuracy, or speed.

The diagnostic checks three implementation contracts:

1. Rank-filling windows preserves a declared per-layer count of FP16, INT4 and
   INT2 windows, including WindowQuant's mandatory first-window FP16 rule.
2. Reordering K and V together and applying the inverse conceptual permutation
   leaves attention output unchanged up to floating-point tolerance.
3. Grouped 1-bit, grouped 2-bit, and full-precision key scores are finite and
   are calculated with `numpy.einsum`, avoiding the BLAS warning path seen in a
   separate matrix-function diagnostic. The grouped quantizers are generic
   affine min/max surrogates with exactly `2**bits` levels; they are not the
   FIER implementation.

The score implementation records two different observables. `raw_logit_means`
averages q·k scores within windows. `softmax_probability_means` first applies
row-wise softmax and then averages probabilities. They are not interchangeable,
and neither is labeled an upper bound.

Memory accounting reports persistent payload bytes, quantizer metadata bytes,
and permutation bytes separately. Metadata is counted only for quantized
windows, with K/V and min/max factors included. It also reports explicit array
byte lower bounds for scoring. Those arrays are not a peak-memory estimate and
do not predict GPU kernels or real packing overhead.

Run with the pinned research environment:

```sh
../.venv/bin/python operator-studies/window-rank-contract/experiment.py \
  --output /tmp/window-rank-contract-results.json
```

The runner refuses to overwrite an existing output. The prior v1 result is
preserved as `results-v1-invalid.json`: its 2-bit surrogate used seven levels,
counted FP16 metadata incorrectly, and called an incomplete byte sum a peak
estimate. The intermediate v2 result is also preserved as
`results-v2-invalid.json` because its 2-bit known-vector assertion exposed a
level-encoding bug. The corrected v3 result is `results-v3.json`.

Seeds are 4811, 4813 and 4817. The runner records Python/NumPy versions and a
fixed hash of the supplied WindowQuant method extract, verifying it only when
that local extract is present. Invalid-input checks exercise NaN scores,
boolean counts, and a missing mandatory FP16 slot. Synthetic traces are only
suitable for these contracts. The next valid test would require fresh VLM
attention traces, equal-byte controls, untouched transfer traces, task quality,
and measured prefill/repacking/decode latency.
