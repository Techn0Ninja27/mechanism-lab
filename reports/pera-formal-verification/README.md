# PERA approximation claim: formal verification and usefulness

**18 November 2026.** We formalized a concrete counterexample to the approximation
bound stated in Appendix A.3 of [PERA v1](https://arxiv.org/html/2604.11841v1).
The mathematical result is about the paper's real, polynomially tied update
family. It is not a refutation of the paper's reported benchmark scores.

## Result in plain language

A matrix can have low rank without belonging to every parameterization capable
of producing that rank. PERA ties its expanded factors together through
polynomials. That constraint matters: there is a simple rank-two target which
rank-one-base PERA cannot approach arbitrarily closely.

The paper's argument treats a rank upper bound as though it guaranteed access
to every matrix under that bound. This reverses the direction of minimization
over a subset. Our example makes the failure quantitative.

## Exact statement checked in Lean

Let

\[
P=\frac13\begin{pmatrix}2&-1&-1\\-1&2&-1\\-1&-1&2\end{pmatrix},
\qquad M=ba^T+h(b\odot b)(a\odot a)^T,
\]

where a and b range over all real three-vectors and h over all real scalars.
Lean proves that P is a symmetric rank-two projector and

\[
\forall a,b,h,\qquad\|P-M\|_2\ge1/\sqrt3.
\]

It also proves the positive lower bound for the **infimum** over all parameters.
Thus neither an unbounded parameter sequence nor the absence of an attained
optimizer avoids the result. The norm is mathlib's Euclidean operator norm,
which is the spectral norm used in the claim.

For this target and nominal rank one, the source bound would require zero
error: the expanded rank bound is two and P's third singular value is zero.
The formal project separately refutes the corresponding claim that all rank-two
matrices can be approximated arbitrarily closely by this family. Translating
the paper's prose into that formal claim is documented in the
[scope audit](claim-scope.md) and [independent audit](independent-formal-audit.md).

## Simplified proof

If a is nonzero, choose a nonzero vector v orthogonal to both a and its
entrywise square. Then Mv=0, regardless of b and h. Orthogonality to the
nonnegative, nonzero squared vector prevents v from having all coordinates of
the same strict sign. In three dimensions this gives

`(v₀+v₁+v₂)² ≤ 2(v₀²+v₁²+v₂²)`.

The projector subtracts the coordinate mean, so

`||Pv||² = ||v||² − (sum v)²/3 ≥ ||v||²/3`.

Consequently the operator norm of P−M is at least `1/sqrt(3)`. The a=0 case
is handled directly. Lean checks the construction, degenerate cases, scalar
inequality, norm conversion and infimum argument.

## What is useful—and what is not established

The correction is useful for **claim validation**: a source-faithful extraction
can reproduce an invalid theorem, so quoted evidence and reviewer acceptance
cannot automatically establish a mechanism's capabilities. Mechanism Lab retains
this counterevidence when generating from the source.

It would be wrong to conclude that PERA has no value. Choose rational parameters
`a=(1,-1,0)`, `b=(1/2,-1/2,0)`, `h=2`. They produce `diag(1,1,0)`. Lean proves
that its error is at most `sqrt(2/3) < 1` and that every rank-one LoRA update
has error at least 1 on this target. The independent exact-rational
and float64 checks give:

| Construction | Trainable scalars | Spectral error on P |
|---|---:|---:|
| Rank-one LoRA | 6 | 1 |
| This rank-one-base PERA witness | 7 | `sqrt(2/3) ≈ 0.8165` |
| Rank-two LoRA | 12 | 0 |

The rank-two control uses an explicit 3×2 by 2×3 factorization, not a full matrix
renamed as a low-rank adapter. Matrix identities are checked with exact rational
arithmetic; floating-point norm values are reported separately in the
[numerical companion](../../operator-studies/pera-usefulness-check/README.md).
The certified PERA optimum lies between `1/sqrt(3)` and `sqrt(2/3)`; we do not
claim the exact optimum or that the displayed witness is optimal.

This is one constructed matrix task, with unequal parameter budgets. It does
not demonstrate downstream accuracy, faster fine-tuning, better memory use,
a publishably novel adapter, or general improvement of our proposal engine.
Those would require separate controlled experiments against strong prior methods.
The [prior-art audit](../pera-method-audit-v1/prior-art.md) identifies the related
Hadamard adapter families. No claim of priority for this counterexample is made.

## Verification and reproducibility

**Verification completed: 18 named theorems, all three verification commands
returned exit code 0.** The [Lean project](../../operator-studies/pera-lean/README.md) pins Lean 4.24.0
and mathlib commit `f897ebcf72cd16f89ab4577d0c826cd14afaafc7`. Its verifier builds
the project and audits the listed theorem dependencies. The expected foundation
is only `propext`, `Classical.choice`, and `Quot.sound`, Lean's standard axioms;
there are no added axioms, unfinished proofs or native proof shortcuts.

- [Verification receipt](../../operator-studies/pera-lean/verification/result.json)
- [Build log](../../operator-studies/pera-lean/verification/build.log)
- [Axiom audit](../../operator-studies/pera-lean/verification/axioms.log)
- [Core formal theorem](../../operator-studies/pera-lean/PeraVerification/Counterexample.lean)
- [Rank and refutation](../../operator-studies/pera-lean/PeraVerification/TargetProperties.lean)
- [Constructive upper bound](../../operator-studies/pera-lean/PeraVerification/Usefulness.lean)
- [Strict comparison with rank-one LoRA](../../operator-studies/pera-lean/PeraVerification/LoRAComparison.lean)

This is stronger than numerical sampling: the formal theorem covers all real
parameters in the stated family. It remains a precise mathematical guarantee
relative to Lean's definitions and trusted foundation, not an unrestricted
“100% true and useful” certification of every claim about the paper.

The arXiv record retrieved on 18 November 2026 lists v1 only. Our correction is
explicitly version-scoped. No messages were sent to the authors. No model
weights, datasets or paid inference calls were needed; shared API spending
remains **$4.8168796 of $20**.

## Author-review draft

A subsequent [public-record check](../pera-author-submission/prior-check.md) found
that the final ACL version retains the same bound, with no prior public correction
found in the inspected sources. A concise [email draft](../pera-author-submission/email-draft.md)
and [technical note](../pera-author-submission/technical-note.md) are prepared.
Nothing has been sent; no priority claim is made.
