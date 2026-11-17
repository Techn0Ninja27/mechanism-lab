# MPS kernel capability probe

2026-11-17. This probe checks whether the local PyTorch MPS backend can execute
the matrix operations needed by a possible Newton–Muon cost profile. It runs a
single seeded 256×256 float32 positive-definite matrix through dense matmul,
Cholesky, inverse-from-Cholesky, eigendecomposition, QR and synchronization.

The process requires `PYTORCH_ENABLE_MPS_FALLBACK=0`, so an unsupported MPS
operation is recorded as an explicit capability failure rather than silently
executing on CPU. CPU calculations are used only afterward to check residuals
of an operation that completed on MPS. No timings are collected, and the probe
does not train a model, compare optimizers, or establish a speed/memory result.

Run from the repository root:

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 ../.venv/bin/python \
  operator-studies/newton-muon-kernel-probe/probe.py \
  --output operator-studies/newton-muon-kernel-probe/result.json
```

The output records each operation's status, bounded exception text and numeric
residuals where available. A successful matmul and synchronization are required
for the overall `passed` field; unsupported decompositions remain useful input
to the next protocol decision. This is a capability snapshot for this PyTorch
and macOS environment, not a general statement about MPS or Apple hardware.

Observed on this environment: synchronization, dense matmul, Cholesky,
inverse-from-Cholesky and QR returned without a fallback warning. `eigh` emitted
an internal CPU-fallback warning for the 256×256 input because it exceeded MPS
threadgroup staging memory. The general fallback setting does not suppress this
internal implementation choice. The probe records it as `internal_cpu_fallback`,
not as an available GPU eigensolver. This was the reason to exclude full `eigh`
from the first cost-screen implementation.
