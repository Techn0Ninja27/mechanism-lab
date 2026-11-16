# NeuMiss shared-polynomial operator audit

Date: 2026-11-16. This is a fresh, development-only numerical audit of the proposal in `results/improvement-phase-2026-11-16/grounded-02/result.json`. It does not train a NeuMiss network, evaluate predictive MSE, or establish novelty.

The official source creates one matrix per layer. With `residual_connection=True`, the first step is `h_1 = r S_0`; only later steps add the residual. If a proposal shares `A` and adds scalar residuals `c_k r`, the exact recurrence is:

`h_L = r A^L + c_2 r A^(L-2) + c_3 r A^(L-3) + ... + c_L r`.

Thus the leading coefficient of `A^L` is fixed at one, there is no free `A^(L-1)` coefficient, and the coefficient vector has only `L-1` entries. A generic degree-`L` polynomial fit is too expressive and would overstate the proposal.

The pre-registered audit uses seeds 17011, 17013, and 17017, dimension 20, fixed `eta=0.5`, and fresh development/validation mask patterns. One coefficient vector per depth is fit across all development masks. Validation is read only after the coefficients are fixed. It compares the exact reachable family with tied coefficients (`c_k=1`), a conventional truncated Neumann sum, and the exact observed covariance inverse.

The metric is relative Frobenius error of the inverse operator, not predictive error. Coefficients may still be partly confounded with the downstream `Wc` scale, and a matrix-level approximation does not establish that a trained network benefits. Any apparent difference remains a diagnostic until a separately frozen network experiment survives controls for step-1 structure, global coefficient sharing, and downstream scale.

## Supplementary strong controls

`strong_controls.py` is a post-hoc diagnostic and leaves the v1 plan/results untouched. It compares the frozen candidate with Richardson iteration using `eta=2/(0.35+1.0)` and a Chebyshev inverse polynomial using the declared design bounds `[0.35,1.0]`. Bounds are not estimated from validation. All controls use degree `L` (therefore `L+1` polynomial terms); the original v1 comparator used `L` terms and remains unchanged. Chebyshev values are evaluated through an eigendecomposition oracle, not presented as runtime.

The supplement also directly unrolls the proposed recurrence on random matrices and checks it against its exact reachable polynomial basis. Any result remains an inverse-operator diagnostic, with no predictive or novelty claim.
