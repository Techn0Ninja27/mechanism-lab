# Newton–Muon Appendix B.3 polynomial plans

2026-11-17. This artifact transcribes all eleven plans printed in Table 5 of
the local primary-source snapshot [Newton–Muon, arXiv:2604.01472v1](https://arxiv.org/html/2604.01472v1), lines 963–1077. The source specifies `CMAX=32`, `INTERVAL_PAD_REL=0.001`, and `NOISE_ABS=0.001` for the table.

The source does not specify one universal polynomial. It first scales the
damped covariance using an upper bound on its largest eigenvalue, forms
`R0 = I - alpha K_gamma`, and selects a plan whose tabulated epsilon is no
larger than the scaled damping and whose SYPP cost fits the budget. Each plan
uses the residual recurrence `r+ = 1 - (1-r) q(r)`; the matrix version updates
`X` and `R` with symmetric polynomial products. Later polynomials are designed
on shrinking symmetric residual intervals, with interval padding and an
absolute noise margin.

`plans.json` preserves coefficients, SYPP costs and published residual bounds;
it does not invent a plan-selection rule beyond the paper. `scalar_check.py`
evaluates the scalar residual recurrence in float64 over one million points per
epsilon. The check is useful for catching transcription errors, but it is not a
matrix proof: it does not assess noncommuting floating-point matrix products,
the eigenvalue upper-bound estimate, grid-LP construction, dynamic programming,
or the paper's precision/noise certificate.

Run:

```sh
python operator-studies/newton-muon-polynomial/scalar_check.py --output operator-studies/newton-muon-polynomial/scalar-result.json
```

The command is deterministic, requires only NumPy 2.5.4, refuses to overwrite
an existing result, and performs no network/API calls. A plan is marked as
passing only when its numerical final residual is at most the published bound
plus `1e-5`; every plan's peak intermediate residual is separately reported.
No optimizer training or timing claim follows from this artifact.

## Captured training matrices: 2026-11-17

The functional dense implementation was tested on 56 covariance blocks from
32 module snapshots: widths 256/512, first/last Transformer blocks, refresh
steps 32/96 of the existing real-text Newton–Muon training harness. Selection
uses a conservative Frobenius upper bound on the covariance's largest
eigenvalue, damping 0.2 times its mean eigenvalue, a published SYPP budget at
most 13, and reported residual at most 1%. It chooses the smallest eligible
published budget. This is our explicit selection policy, not a new plan.

| Width | Eligible blocks | Total blocks |
|---|---:|---:|
| 256 | 14 | 28 |
| 512 | 4 | 28 |

All 18 eligible blocks met both the 1% residual threshold and the published
bound plus 1e-5 rounding allowance. Worst spectral inverse residual was
0.00387155 (0.3872%). Only 12 of 32 module snapshots had every block covered;
there, the maximum relative error in the raw preconditioned gradient was
0.00278947. These are descriptive development checks, not independent trials.
The remaining 38 blocks have no eligible plan under this conservative policy;
a tighter bound or additional plans might cover them. This is not a failure
of the paper's general method. No fallback warning was observed.

The matrix recurrence runs in float32 on MPS. Selection and reference inverse,
eigenvalue, and residual calculations run on CPU in float64. The generic dense
products do not reproduce optimized SYPP kernels or their operation costs.
Covariances were exactly symmetric in this capture (the checker permits tiny
asymmetry and symmetrizes both inputs). This does not test momentum/NS updates,
training quality, time-to-loss, or novelty. The capture and harness include
full covariance storage. No low-rank candidate is justified by this result.

[Raw matrix results](matrix-result.json) and [pre-check code freeze](matrix-freeze.json)
retain per-block eligibility, residuals, and source hashes. Large NPZ captures
remain local under `results/newton-muon-polynomial-v1/capture/`.

Reproduce from the repository root with the pinned environment and the existing
cost-screen text fixture (its SHA is recorded in the result). Choose fresh
output paths; commands refuse to overwrite results:

```sh
for width in 256 512; do
  PYTORCH_ENABLE_MPS_FALLBACK=0 ../.venv/bin/python operator-studies/newton-muon-polynomial/capture.py --width "$width" --output "results/polynomial-replay/capture/d$width.npz" --manifest "results/polynomial-replay/capture/d$width.manifest.json"
done
PYTORCH_ENABLE_MPS_FALLBACK=0 ../.venv/bin/python operator-studies/newton-muon-polynomial/matrix_check.py --capture results/polynomial-replay/capture --output results/polynomial-replay/matrix-result.json
../.venv/bin/python -m unittest discover -s operator-studies/newton-muon-polynomial -p 'test_*.py' -v
```

Capture uses seed 0, the source harness, and its existing text preparation;
MPS training is not claimed to be bitwise reproducible across fresh captures.
Neither capture nor checking downloads data or makes paid API calls.

Audit clarification: the harness refreshes K and its inverse before applying the
current raw gradient. Thus the captured pre-step G and post-refresh K match
the same-step preconditioning operation; this is not a timestep mismatch.
The [refresh-frequency follow-up](refresh-followup.md) is protocol only.
