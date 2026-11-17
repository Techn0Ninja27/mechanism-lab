# PERA equation and initialization audit

2026-11-17. Inspected the retained primary HTML for
[arXiv:2604.11841v1](https://arxiv.org/html/2604.11841v1), including Appendix A.3.
This distinguishes the method's definition from claims about its capabilities.

For B in R^(m×r), A in R^(r×n), and one real coefficient h_ij per i≤j,

\[
\Delta W=BA+\sum_{i\le j}h_{ij}(b_i\odot b_j)(a_i\odot a_j)^T.
\]

The B expansion is ordinary polynomial expansion; the A expansion carries the
coefficients. The expanded factor width is `K=r+r(r+1)/2`, with
`r(m+n)+r(r+1)/2` trainable scalars. This represents a constrained family of
rank-at-most-K updates, not an unrestricted rank-K factorization.

At the stated B-Gaussian, A-zero, h-zero initialization, the update and its
quadratic part are zero. For loss derivative G with respect to Delta W,
`grad_A = BᵀG` can be nonzero, while `grad_B=0` and `grad_h=0`. After A moves,
h can receive a gradient, and later the nonlinear contribution can become
nonzero. Thus the zero initialization delays the nonlinear path; it does not
permanently disable it. This observation is not a stability or convergence
proof. See the [recorded diagnostic](../../operator-studies/pera-algebra/README.md).

The method equation does not explicitly show LoRA's alpha/r scaling. The full
appendix does list optimizer and hyperparameters: AdamW, learning rate 1e-4,
linear scheduling, warmup and alpha values. Those should be distinguished from
what was missing in the selected extraction excerpt. A matched implementation
still needs to resolve how those settings enter the actual forward/update path.

The elementary rank upper bound is valid. The stronger approximation statement
in Appendix A.3 is not implied by it: minimizing over a subset cannot improve
the minimum over its containing set. The valid sandwich is

\[
\sigma_{K+1}(E)\le\inf_{M\in S_{PERA}}\|E-M\|_2\le\sigma_{r+1}(E),
\]

where the upper bound follows from setting h=0 to recover LoRA. The explicit
[3-by-3 projector counterexample](../../operator-studies/pera-rank-counterexample/README.md)
proves a positive error lower bound for nominal rank one, despite the target
having rank two. Its proof covers the infimum, including unbounded parameter
sequences, rather than merely showing that no exact minimizer exists.

These checks do not establish a new adapter or measure downstream task quality.
The source-defined equation remains meaningful even though a theorem about its
approximation power fails. The [published-layer probe](../../operator-studies/pera-author-probe/README.md)
is a separate implementation finding; it should not be confused with the
mathematical counterexample or a reproduction of the full experiments.
