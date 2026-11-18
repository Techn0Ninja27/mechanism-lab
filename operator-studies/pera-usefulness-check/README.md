# Bounded PERA usefulness comparison

18 November 2026. On the target P=I-11ᵀ/3, a concrete rank-one-base PERA update
beats the best rank-one LoRA error, while a higher-budget rank-two LoRA
reconstructs P exactly. This illustrates why a false approximation theorem
must not be confused with a useless parameterization.

| Construction | Factor/coefficient scalars | Spectral error |
|---|---:|---:|
| Rank-one LoRA | 6 | 1 |
| Explicit PERA witness | 7 | sqrt(2/3), approximately 0.8165 |
| Rank-two LoRA | 12 | 0 |

The rational PERA parameters are a=(1,-1,0), b=(1/2,-1/2,0), h=2. Their
update is exactly diag(1,1,0). The rank-two control uses explicit 3×2 and 2×3
factors listed in [protocol-v2.json](protocol-v2.json); its output is computed
by matrix multiplication. Counts are checked from the factor sizes.

[run_v2.py](run_v2.py) verifies the matrix identities with Python Fraction and
computes spectral norms with CPU float64 PyTorch. The seven checks passed in
[result-v2.json](result-v2.json), including a separate parent replay. Python
3.12.14 and Torch 2.15.0 are checked by the runner; no random seeds are needed.

```sh
../.venv/bin/python operator-studies/pera-usefulness-check/run_v2.py \
  --output /tmp/pera-usefulness-fresh.json
```

Existing outputs are refused. The [Lean companion](../pera-lean/README.md)
separately proves error at least 1 for every rank-one LoRA update and error at
most sqrt(2/3)<1 for this PERA witness. It certifies the PERA infimum interval
[1/sqrt(3),sqrt(2/3)], without claiming the exact optimum.

This is one hand-constructed matrix with unequal parameter budgets, not a
transformer task, optimizer comparison, speed result or new adapter. The scalar
counts omit base weights and biases. No downstream usefulness or novelty is
established.

The initial v1 implementation assigned the full target directly as a nominal
rank-two control. That did not verify the claimed factorization and is not the
current reproducer. Its code/results and the first v2 debugging attempt remain
locally preserved under `results/pera-usefulness-check-v1` and
`results/pera-usefulness-check-v2`; the successful v2 replay uses actual factors.
