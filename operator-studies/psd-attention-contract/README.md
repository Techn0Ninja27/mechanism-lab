# PSD approximation does not preserve attention probabilities

Recorded 2026-11-16. This manually selected contract diagnostic is a known
mathematical distinction, not a graph-selected discovery or a trained-model result.

Question: does a positive-semidefinite, optimal rank-four approximation of a
strictly positive RBF Gram matrix necessarily remain entrywise nonnegative?
No: six of nine fixed-seed cases contain negative entries and negative normalized
weights. Positive semidefiniteness constrains quadratic forms, not individual
entries. Row normalization alone does not repair this failure.

The construction uses 24 Gaussian points in four dimensions, seeds 88031, 88033,
88037, bandwidths 0.3, 1, 3, and the top-four eigenspace as the Nyström sketch.
It verifies agreement with the optimal spectral truncation and positive
semidefiniteness to numerical tolerance. This sketch uses an eigendecomposition;
it is a counterexample construction, not a proposed efficient attention method.

[Persson, Meyer and Musco](https://arxiv.org/abs/2311.14023) study matrix functions
of PSD Nyström approximations under stated operator-monotonicity assumptions.
Here f(t)=t. Their guarantee does not promise entrywise positivity.
[SLAY](https://arxiv.org/abs/2602.04915) constructs positive attention-kernel
approximations by a different route. This experiment refutes neither paper.
Matrix spectral functions and entrywise kernel constructions must not be conflated.

Reproduce without overwriting the saved result (NumPy 2.5.4, Python 3.12.14):

```sh
out=$(mktemp -d)
cp operator-studies/psd-attention-contract/experiment.py "$out/experiment.py"
../.venv/bin/python "$out/experiment.py"
```

`results.json` records versions and the experiment source hash. The original
BLAS matrix multiplications emitted divide/overflow/invalid warnings despite
finite outputs; their cause remains unresolved. A separate einsum contraction
audit agreed on all nine negative-entry counts, with minimum-entry differences
at most 3.61e-16 (`numerical-audit.json`). No task-quality or speed conclusion
follows. Any proposed attention composition must separately check entrywise
nonnegativity and positive row sums before treating its output as probabilities.
