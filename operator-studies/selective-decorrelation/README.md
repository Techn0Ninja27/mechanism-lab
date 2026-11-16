# Selective decorrelation operator audit

Date: 2026-11-16. This is a cheap numerical audit of the operator used by the
official StableMiss+ source snapshot at commit `fd02c06ce24b7ae153403bcbe6a477ffc6f76fe2`.
It is not a reproduction of the paper, a model benchmark, or a novelty result.

Run:

```sh
../.venv/bin/python operator-studies/selective-decorrelation/experiment.py
```

The source code's reweighting loss is in `reweight_stablemiss+.py:34-52`:
Wasserstein distance plus off-diagonal mask covariance, cross covariance, and
`10 * (std(w) / mean(w))`. `metric.py:34-40` defines weighted covariance as
`(w*x)^T x - sum(w*x) sum(w*x)^T`; the model supplies sum-one softmax weights in
`model.py:117-123`. `metric.py:17-31` constructs a random Fourier transform,
defaults to one feature, and normalizes each row across variables before taking
covariances. `metric.py:71-88` then applies a mask matrix elementwise to the
feature covariance. These details are recorded as source observations, not
correctness claims.

The independently written check uses fresh seed `16017`, correlated continuous
features, and mask dependencies on an observed feature and on another mask.
It measures direct weighted covariance and a finite-difference diagnostic of
only the mask-covariance plus coefficient-of-variation terms. It deliberately
omits the source Wasserstein, feature-correlation, and cross-correlation terms.
It also checks a collapsed single-row weight and an arbitrary rescaling of
weights. An AST-only extraction of the official `cov` function compares its
output to the independent implementation for normalized random and one-hot
weights; no source module is imported.

## Result

The JSON result records the exact values. The attempted finite-difference update
did not lower this partial objective under the source coefficient-of-variation
penalty on this fixture. That result is only a diagnostic, not a claim about the
paper's Adam run. The full objective's omitted terms could change the result.
The effective sample size and feature covariance are reported to make the
tradeoff visible. A single-row weight is a degenerate distribution, not evidence
that the original population was decorrelated. Arbitrary scaling changes the source
`cov` expression unless weights sum to one, although the source softmax does
enforce sum one.

The operator therefore has two separate interpretation limits. First, reducing
weighted mask covariance is not by itself proof of independence under unseen
mechanisms. Second, the random-feature and elementwise masking path is not the
same as preserving all original feature-feature correlations while removing all
feature-mask correlations. This study does not determine whether StableMiss+
is useful on a fresh benchmark; it only exposes diagnostics a faithful next
evaluation must report.
