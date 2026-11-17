# Newton–Muon low-rank inverse preflight

2026-11-17. This is the cheapest algebraic check for the generated optimizer
proposal. It verifies the low-rank-plus-isotropic inverse against an explicit
inverse of the approximated matrix on seeded positive-semidefinite matrices, and
checks the full-rank branch separately.

It also tests the proposal's claim that rank zero is “Muon up to a scale that
`msgn` removes.” A scalar applied before momentum is not generally removed by a
later polar/`msgn` operation when that scalar changes over time. With gradients
`[1,0]`, `[0,1]`, momentum coefficient `beta=0.9`, and scales `1,4`, the second
polar direction differs from the constant-scale trajectory. One constant
positive scale gives the same polar directions.

This does not establish usefulness, training quality, timing, MPS behavior, or
novelty. It does not claim that a low-rank approximation is worth using. The
dense and low-rank apply costs are asymptotic accounting gates; refresh cost,
eigensolver behavior and end-to-end timing remain empirical.

Run from the repository root:

```sh
../.venv/bin/python operator-studies/newton-muon-preflight/check.py \
  --output operator-studies/newton-muon-preflight/result.json
```

The output is deterministic for the recorded seed. No data or model download is
required.
