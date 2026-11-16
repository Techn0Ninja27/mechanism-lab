# Random-feature normalization-axis audit

Date: 2026-11-16. This is a deterministic operator audit of the paid proposal
saved at `results/improvement-phase-2026-11-16/grounded-01/result.json`. It
does not train a reweighter, run a predictor, or claim novelty.

Run:

```sh
../.venv/bin/python operator-studies/normalization-axis/experiment.py
```

The experiment AST-extracts only five reviewed functions from the official
StableMiss `metric.py` snapshot (source commit and SHA are in `results.json`).
It compares the supplied `dim=1` normalization with the proposal's single
change to `dim=0`, using the same fixed phases, `num_f=1`, and truthy `sum=True`.
It checks binary masks at `(n,d)=(64,5)` and `(2048,20)`, with one fully observed
column to test the zero-denominator case. It then calls the extracted source
loss functions without importing the repository module.

## Result and interpretation

For the exact proposed construction, each nonconstant binary column under
`dim=0` is mapped to normalized endpoints `0` and `pi/2`; `cos(0)+sin(0)` and
`cos(pi/2)+sin(pi/2)` are both `1`. Thus the proposed mask feature is constant
for every binary column, while an all-ones column produces NaNs from division by
zero. This is a removal of the mask-loss signal, not a successful decorrelation
operator. The result is algebraic and does not depend on a model benchmark.

The shape pre-check also records that `intra_feature_correlation` and
`inter_correlation` fail when `n != d` because a `(d,d)` covariance is
multiplied elementwise by an `(n,d)` mask in the supplied source path. The
mask-only correlation path does not use that multiplication and is separately
shape-safe. These are observations about the released implementation; this
audit does not silently repair the code or infer the authors' intended shapes.

The proposed change should therefore be rejected as written. A variant changing
the feature representation or using `sum=False` would be a different proposal
and requires a new pre-registered operator audit.
