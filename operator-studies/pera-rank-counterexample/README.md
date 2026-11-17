# Counterexample to PERA's rank-approximation bound

2026-11-17. This checks **Theorem 1 in Appendix A.3 of
[PERA v1](https://arxiv.org/html/2604.11841v1)** under the real, tied polynomial
parameterization defined by its method. It does not challenge the elementary
rank upper bound or evaluate the paper's empirical benchmark results.

## The inference that fails

Let `K = r + r(r+1)/2`. The paper correctly places the PERA update family
inside the matrices of rank at most K, but then bounds its optimal spectral
approximation error **above** by the optimal error over all rank-K matrices.
Minimizing over a smaller set gives the opposite inequality.

Since PERA also contains every rank-r LoRA update when the polynomial
coefficients are zero, the justified bounds are

\[
\sigma_{K+1}(E)\;\le\;\inf_{M\in S_{\rm PERA}}\|E-M\|_2
\;\le\;\sigma_{r+1}(E),
\]

with zero singular values beyond the matrix dimensions. The lower bound
cannot be turned into equality merely because some PERA matrices attain rank K.

## A concrete positive-error counterexample

Take `r=1`, real 3-by-3 matrices, and

\[
n=\frac{1}{\sqrt3}(1,1,1)^T,\qquad
P=I-nn^T=\frac13\begin{pmatrix}2&-1&-1\\-1&2&-1\\-1&-1&2\end{pmatrix}.
\]

P is an orthogonal projector of rank two, with singular values 1, 1, 0.
The stated theorem would therefore require zero approximation error.
However, **every** rank-one-base PERA update

\[
M=ba^T+h(b\odot b)(a\odot a)^T
\]

satisfies `||P-M||₂ >= 1/sqrt(3)`:

1. If M has rank at most one, the rank-one spectral approximation lower bound
   is 1.
2. Otherwise M has rank two, so its row space is exactly the span of `a` and
   `a⊙a`. It contains the unit nonnegative vector
   `u=(a⊙a)/||a⊙a||₂`. This vector is well defined in the rank-two case.
3. Choose a unit vector v in the right kernel of M. Then `uᵀv=0`. Since u
   is nonnegative with unit Euclidean norm, `nᵀu=||u||₁/sqrt(3) >= 1/sqrt(3)`.
   Projecting n onto the orthogonal complement of u gives
   `|nᵀv| <= sqrt(1-(nᵀu)²) <= sqrt(2/3)`.
4. Consequently,
   `||P-M||₂ >= ||(P-M)v||₂ = ||Pv||₂ = sqrt(1-(nᵀv)²) >= 1/sqrt(3)`.

The bound is uniform over all real a, b and h, including divergent parameter
sequences. Thus this is a positive lower bound on the **infimum**, not merely
failure to attain an exact representation. Negative h does not alter the
row-space argument. No optimization failure or numerical rank threshold is
used in the proof.

The counterexample addresses the actual polynomially tied family. If the
square/cross updates were instead independent arbitrary low-rank matrices,
that would define a different model. The v1 theorem states no restriction
excluding r=1.

## Reproduction and limits

The proof was checked separately by native agents and the parent agent. It is
not formally verified in Lean or another proof assistant. The accompanying
[script](check.py) checks the target's symmetry, zero row sums, trace and
`Q²=3Q` using exact integers for `Q=3P`. These establish the projector's
singular values; the script does **not** mechanically prove the universal
row-space argument above.

```sh
../.venv/bin/python operator-studies/pera-rank-counterexample/check.py \
  --output /tmp/pera-rank-counterexample-fresh.json
```

Python 3.12.14, standard library only; no datasets, model weights, API calls or
random seeds. [Protocol](protocol.json) binds the inspected primary HTML hash;
[result](result.json) records the exact checks and proof qualification.

This is a scoped correction to a source claim. We have not established that
this observation is previously unpublished, that PERA is ineffective on real
tasks, or that replacing it produces a useful new method. Its relevance to
Mechanism Lab is that a correctly extracted paper statement can still be
mathematically wrong and must not automatically become a validated capability.
