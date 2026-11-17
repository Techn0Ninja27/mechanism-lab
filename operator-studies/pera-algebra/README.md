# PERA algebra diagnostic

The initial v1 run was exposed before the protocol hardening. Its result remains
unchanged in `results/pera-algebra-v1`. The v2 protocol is frozen 2026-11-17
before the v2 run. This is a small CPU float64 algebra check of
the update described in the selected primary-source excerpt for
[PERA, arXiv:2604.11841v1](https://arxiv.org/html/2604.11841v1). It does not
reimplement a transformer adapter and does not test task quality.

Question: does the concatenated-factor expression

`Delta W = Poly2(B) PolyH2(A)`

match an independent rank-one expansion for arbitrary learned coefficients, and
what does the paper's `B` Gaussian, `A=0`, `h=0` initialization imply locally?

The diagnostic uses `B` in `m x r`, `A` in `r x n`, and one coefficient `h_ij`
for every `i <= j`. `Poly2(B)` contains the original columns followed by
`b_i ⊙ b_j`; `PolyH2(A)` contains the original rows followed by
`h_ij(a_i ⊙ a_j)`. The expanded width is `r + r(r+1)/2` and the trainable
parameter count is `r(m+n) + r(r+1)/2`. The matrix rank is bounded by the
minimum of `m`, `n`, and that width; reaching the bound for random parameters is
only an algebraic diagnostic, not a proof of useful adaptation.

The special assignment `h_ii = lambda`, `h_ij = 2 lambda` for `i < j` gives
`Delta W = BA + lambda * ((BA) ⊙ (BA))`. At the paper's initialization, the
loss gradient is generically nonzero for `A` but zero for `B` and `h`; after an
`A` update, `h` can receive gradient. The word “can” is intentional: this is
not a claim that nonlinear terms are always activated after one step.

Run the unit diagnostic from the repository root with the repository environment:

```sh
../.venv/bin/python -m unittest discover -s operator-studies/pera-algebra -p 'test_*.py'
../.venv/bin/python operator-studies/pera-algebra/run.py --protocol operator-studies/pera-algebra/protocol.json --output /tmp/pera-algebra-fresh
```

The v2 result refuses to overwrite an existing output, records the protocol
hash, source/code hashes, environment, nine seeded algebra arms, and three
ten-update synthetic matrix-MSE traces. It checks factorization, gradients,
the square identity, initialization behavior, and rank bounds; a failed gate
is saved before the runner exits nonzero. The protocol and tests are source-faithful to the excerpt but make no claim
about the paper's reported benchmark improvements, which require the authors'
full implementation and task datasets.

## Observed result

The [v2 result](result.json) retains status **checks_failed**: two of nine
arms exceeded the predeclared absolute gradient discrepancy tolerance of
`1e-11`, with maximum `2.183e-11`. We did not loosen the gate and rerun it as a
success. A separate [post-run scale diagnostic](gradient-scale.json) found a
maximum relative L2 gradient discrepancy of `1.62e-15` across the parameter
groups, with reference gradients reaching `5.34e4`. This is consistent with
floating-point accumulation differences; the original absolute gate remains
failed. Neither number measures task quality.

Forward expressions agreed within `2.85e-14`; the special elementwise-square
identity agreed within `1.43e-14`. All nine random fixtures attained the
algebraic rank bound (ranks 2, 5 and 12), not a proof that every matrix of those
ranks is expressible. All three short synthetic training traces had zero
initial B/h gradients, nonzero A gradients, and nonzero h gradients after the
first parameter update. Ten synthetic matrix updates establish neither faster
learning nor a useful transformer fine-tuning result.

```sh
../.venv/bin/python operator-studies/pera-algebra/analyze_roundoff.py \
  --result operator-studies/pera-algebra/result.json \
  --output /tmp/pera-gradient-scale-fresh.json
```

The [prior-art audit](../../reports/pera-method-audit-v1/prior-art.md) relates
these equations to existing Hadamard adapters. The separate
[author-code probe](../pera-author-probe/README.md) exposes a missing coefficient
entry in the pinned public layer. This independent equation implementation is
not a repaired or reproduced version of that complete training system.

A publication-format replay after removing a trailing blank line from the module
reproduced every numeric row and failed check exactly. The public result binds
the final code hash; both local runs remain preserved.
