# Mask embedding contract audit

Date: 2026-11-16. This engineering study tests a reusable representation
contract for binary observation masks. It does not claim independence,
predictive utility, or novelty. The standard random-feature precedent is Rahimi
and Recht, *Random Features for Large-Scale Kernel Machines* (NeurIPS 2007):
<https://papers.nips.cc/paper_files/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html>.

Run:

```sh
../.venv/bin/python operator-studies/mask-embedding-contract/experiment.py
```

The candidate in `generation/mask_embedding.py` uses seeded, shared frequencies
and paired sine/cosine features. It performs no data-dependent min/max
normalization, so constant columns remain finite and the output is independent
of batch composition. Row and column permutations have predictable equivariance.
For binary inputs, every per-element embedding obeys the exact identity
`phi(m) = phi(0) + m * (phi(1) - phi(0))`; the audit checks this numerically.
It therefore adds no information beyond the raw binary indicator. Raw indicators
pass the same structural contracts and are the simplest preferred baseline.

The audit compares it with raw indicators, the rejected per-column cos+sin axis,
and the source per-row normalization on fresh masks at seeds `16021`, `16023`,
and `16027`. The rejected axis collapses finite binary columns to a constant and
produces NaNs for constant columns. The source row-normalized representation is
also invariant to adding rows when its random phases are held fixed, but the
official function samples fresh phases on every call; the candidate makes its
seeded phases explicit and reproducible.

Passing these contracts only makes the representation safer to evaluate. It does
not show that mask independence holds, that a predictor improves, or that the
operator is new. No primitive is promoted, and there is no evidence here to
choose the sine/cosine encoding over raw indicators.
