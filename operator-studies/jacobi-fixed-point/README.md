# Jacobi fixed-point diagnostic — 2026-11-16

This tests the literal formula generated in grounded-03. It is an algebraic/numerical diagnostic, not a trained-network benchmark or novelty result.

For an observed covariance block S and c equal to its largest eigenvalue, the proposal defines W = I - S/c and D = diag(I-W), then iterates z <- D^-1 W z + r. Its fixed point is (I-D^-1 W)^-1 r, which generally differs from S^-1 r. A correct diagonally preconditioned Richardson update scales the residual: z <- z + eta diag(S)^-1 (r-Sz). Jacobi's classical construction is documented in [Netlib's Templates](https://netlib.org/linalg/html_templates/node12.html).

Run from the repository root:

```sh
../.venv/bin/python operator-studies/jacobi-fixed-point/experiment.py
```

Fresh seeds 18011, 18013 and 18017 generate small SPD covariance matrices and observation patterns. The literal iteration is stable on these cases and reaches its own fixed point, but relative error to the intended inverse is 0.297–0.409. Even a best fitted global output scale leaves 0.027–0.050 relative error. Corrected Jacobi reaches the inverse to floating-point precision.

Two additional checks prevent interpreting a familiar correction as a new mechanism:

- Diagonal preconditioning matches Richardson iteration on symmetrically standardized coordinates, up to 6.7e-16 in this test.
- For fixed diagonal D, its action commutes with the observation projector and folds into the model's weight matrix, up to 1.4e-17. This is a representational equivalence, not a claim of identical optimization dynamics or unchanged regularization.

The formula is rejected as written. Correcting it yields an established preconditioned iteration; no predictive improvement or new primitive is established. No previous held-out prediction fixture was used.
