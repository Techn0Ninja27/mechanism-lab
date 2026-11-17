# Spectral-bound coverage diagnostic

Recorded 2026-11-17. [Per-block results](result.json).

This CPU-only diagnostic revisits the 56 frozen covariance blocks captured for
the Table-5 polynomial check. The original selector used `1.001 * ||K||_F` as
an upper bound on the largest eigenvalue. Here we compare that with two other
valid PSD bounds: `1.001 * max_i sum_j |K_ij|` (the symmetric induced-norm
bound) and `1.001 * trace(K)`. We also report `1.001 * lambda_max(K)` from a
float64 `eigvalsh` computation as an offline oracle. The eigensolver result is
diagnostic only; it is not a cheap runtime certificate and does not reproduce
the paper's bound-construction proof.

For every bound the published plan filters are unchanged: epsilon must be at
least the table epsilon, total SYPP must be at most 13, and the reported
residual must be at most 0.01. No polynomial is invented for uncovered blocks.
The input NPZ hashes and source hashes are retained in the result. This does
not rerun MPS, alter the original result, or make a timing/training claim.

Run from the repository root:

```sh
../.venv/bin/python operator-studies/newton-muon-bound-diagnostic/analyze.py \
  --capture results/newton-muon-polynomial-v1/capture \
  --output results/newton-muon-bound-diagnostic-v1/result.json
```

The command refuses to overwrite an existing result and performs no network or
API calls.

## Result on the frozen capture

The 56 blocks were all symmetric to machine precision and passed the existing
PSD check. Coverage of the unchanged published-plan filters was:

| Upper bound | Eligible blocks |
|---|---:|
| Frobenius (original policy) | 18/56 |
| Maximum absolute row sum | 5/56 |
| Trace | 0/56 |
| Float64 `eigvalsh` oracle | 28/56 |

The row-sum and trace bounds add no coverage because these covariances are
dense enough that they are looser than the Frobenius bound. The eigensolver
oracle exposes ten additional potentially coverable blocks, but it is an
offline diagnostic, not a cheap certified bound suitable for claiming a
runtime speedup. The current functional result therefore remains 18/56; the
coverage gap is partly the conservatism of the bound, while the table plans do
not cover all blocks even when the observed largest eigenvalue is used.

The result is descriptive for this one frozen capture and does not establish
that a tighter bound, an optimized SYPP implementation, or the polynomial
inverse improves training or time-to-loss.

The union of the three cheap-bound selections is also 18/56. All observed
minimum eigenvalues were positive (minimum 0.0008563331). Floating-point
eigenvalue checks and the 0.1% slack are numerical diagnostics, not interval-
arithmetic certificates. Requirements are pinned in `requirements.txt`.
