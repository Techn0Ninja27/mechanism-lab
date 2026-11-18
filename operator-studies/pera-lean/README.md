# Lean verification of the PERA counterexample

This project checks a quantitative counterexample to Appendix A.3, Theorem 1
of [PERA v1](https://arxiv.org/html/2604.11841v1). It uses Lean 4.24.0 and
mathlib commit `f897ebcf72cd16f89ab4577d0c826cd14afaafc7`, pinned in the manifest.

For every real `a,b ∈ R³` and scalar `h`, define

\[
M_{ij}=b_i a_j+h b_i^2 a_j^2,
\qquad P=I-\mathbf1\mathbf1^T/3.
\]

The central theorem is

\[
\forall a,b,h,\qquad 1/\sqrt3\le\|P-M\|_2.
\]

Here the norm is mathlib's **Euclidean operator norm**, selected explicitly
with `Matrix.Norms.L2Operator`. It is not the elementwise, Frobenius or row-sum
norm. The theorem has no bounded-parameter, nonzero-factor, optimizer,
initialization or attainment assumptions.

The project also proves that P is symmetric, idempotent and exactly rank two;
that the **infimum** error is positive; and that nominal-rank-one PERA therefore
cannot approximate every rank-two matrix arbitrarily closely. A rational PERA
witness gives an error at most `sqrt(2/3) < 1`, yielding a certified interval
for the best possible error. The exact optimum is not claimed.

## Reproduce

Install Elan using the [official Lean instructions](https://lean-lang.org/install/manual/),
then run from this directory:

```sh
export PATH="$HOME/.elan/bin:$PATH"
lake update
lake exe cache get Mathlib.Tactic Mathlib.Analysis.CStarAlgebra.Matrix Mathlib.LinearAlgebra.Matrix.Rank
python3 verify.py --output /tmp/pera-lean-verification-fresh
```

The first setup downloads compiler/library dependencies. No models, datasets
or paid inference are needed. Subsequent checks use the local pinned build.
The output directory must not already exist.

`verify.py` builds the project, runs the explicit axiom audit in `Audit.lean`,
checks that every listed theorem is accounted for, rejects proof escapes in
local Lean files, and saves source/dependency/log hashes. The admitted foundation
is Lean's usual `propext`, `Classical.choice`, and `Quot.sound`; no new axioms,
unfinished proofs or native proof shortcuts are accepted.

All **18 audited theorems** passed the recorded verification. The [verification receipt](verification/result.json), [build log](verification/build.log)
and [axiom audit](verification/axioms.log) record the actual checked artifact.
Lean checking establishes these formal statements relative to its kernel,
mathlib definitions and standard axioms. Correspondence with the paper's prose
is assessed separately in the [report](../../reports/pera-formal-verification/README.md).

## Proof structure

- `ScalarBound.lean`: a nonnegative nonzero weight vector orthogonal to v forces
  `(sum v)² ≤ 2 sum(v_i²)` in three dimensions.
- `KernelWitness.lean`: explicit cross-product/coordinate constructions provide
  a nonzero vector orthogonal to a nonzero a and its elementwise square.
- `Counterexample.lean`: applies that vector to P−M and bridges the squared
  inequality to the Euclidean operator norm and the infimum error.
- `TargetProperties.lean`: proves P's projector/rank properties and refutes the
  universal rank-two approximation consequence of the source claim.
- `Usefulness.lean`: proves the rational witness and its strict error upper bound.
- `LoRAComparison.lean`: proves error at least 1 for every rank-one LoRA update
  on this target, and the strict comparison with the PERA witness.

The correction does not invalidate the paper's empirical tables, establish
priority for this observation, or prove a new adapter useful on downstream tasks.
The original mathematical definition remains meaningful even though this
particular approximation guarantee fails.
