# Full-method triage: graphon NTK and randomized low-rank integration

2026-11-17. Root-agent analysis of two entries in this exposed development
packet. This is a source review and algebraic screen, not an independent
evaluation, new algorithm, or model-training label. Original extracted notes
and reviewer outputs remain unchanged.

## What the abstracts omitted

[Graphon Limit Hypothesis, v1](https://arxiv.org/html/2510.17515v1),
sections 4.1–4.2, explicitly assumes a retained fraction p > 0 that stays
constant as width increases, with comparable hidden-layer widths. Section 8
limits the analysis to fixed masks at initialization in MLPs and describes
graphon convergence as empirically supported rather than proved in general.

Our earlier question about fixed-degree masks therefore concerns an excluded
regime, not a counterexample to the stated result. For a nonnegative adjacency
step function with at most dn edges on n-by-n vertices, its integral and L1
norm are at most d/n. Its cut norm is no larger, hence it converges to zero
when d is fixed. That elementary observation alone is not a novel research
contribution and does not determine a suitably rescaled neural network's NTK.
Changing normalization changes the question and requires a separate theory.

The author's cited predecessor,
[Yang and Wang, AISTATS 2023](https://proceedings.mlr.press/v206/yang23b.html),
already studies randomly pruned NTKs: fixed-probability infinite-width
equivalence up to scaling, plus finite-width approximation bounds involving
sparsity. Any proposed experiment on random-pruning NTK convergence needs to
compare its assumptions and predictions against this result first. We have
checked the abstract here, not audited its full proof or exact width bound.

[Splitting-based randomized DLRA, v1](https://arxiv.org/html/2506.15259v1),
section 2.1, separates a matrix ODE into a stiff linear part and a nonlinear
part. It gives the exact linear flow in equation 4 and applies randomized
low-rank integration to the nonlinear subproblem. The linear flow already
preserves rank. The abstract's phrase “stiff matrix differential equations”
alone does not establish that an arbitrary kernel-training problem needs
both parts of the method.

## Cheapest falsification: specialize the equations

Candidate screened here: use that splitting machinery to accelerate residual
covariance evolution under a fixed Graphon NTK approximation. This is our
candidate mapping, not a proposal or experiment reported by either paper.

Assume a finite, fixed symmetric positive-semidefinite kernel K, squared-error
gradient flow, and residual matrix E with examples in rows and outputs in
columns. Absorb the loss-normalization constant into K. Then

    dE/dt = -K E
    X = E E^T
    dX/dt = -K X - X K^T
    E(t) = exp(-t K) E(0)
    X(t) = exp(-t K) X(0) exp(-t K^T).

This maps to A = -K and F(t, X) = 0 in the splitting equation. The nonlinear
flow is the identity. Lie and Strang splitting collapse to the same linear
flow when that flow is evaluated exactly. There is no nonlinear integration
task for the randomized stage. Factor propagation already avoids explicitly
storing X when E has few columns. Approximating matrix-exponential actions
can still cost time and introduce error; this argument claims no free exact
computation and no performance result.

The fixed-kernel least-squares dynamics are established background:
[Jacot, Gabriel and Hongler (2018)](https://arxiv.org/abs/1806.07572).
The equations above are our specialization of that background and the
splitting paper, not a new theorem from the graphon paper.

**Decision:** do not spend API calls or training compute benchmarking the
randomized nonlinear stage on this fixed-kernel covariance mapping. Its
claimed extra role disappears algebraically. This does not rule out a useful
time-varying kernel, forcing term, non-squared loss, or another operator.
Those would be different candidates: each needs a derived closed evolution,
nonzero term, matched baseline and task-level benefit before proceeding.

## Consequence for the engine

Input/output compatibility is necessary but insufficient. A candidate can
have compatible matrix types while the contributed operator becomes an
identity after substitution. Before approving an experiment, record:

1. The source assumptions and whether the target satisfies them.
2. The actual equation after substituting the target into the proposed method.
3. Which added operation remains nontrivial, and the simplest existing solver.
4. The measurable benefit that solver does not already supply.

These are review questions, not a new learned ranker or an automatic rejection
rule. In particular, “outside source assumptions” means unsupported transfer,
not proof of infeasibility. This packet cannot estimate how frequently the
engine makes these mistakes. The paired JSON stores this narrow disposition
without granting novelty or training eligibility.
