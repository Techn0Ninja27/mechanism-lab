# Matched refresh-frequency control

This is a short, reproducible cost control for the frozen Newton--Muon
development harness. It compares refresh intervals 16, 32, and 64 at widths
256 and 512 with the same real-text fixture, seed, batches, model settings,
and optimizer implementation. Each arm is one fresh process: 8 warmup steps,
then 128 measured steps in four 32-step windows. The runner stores every loss
and window, but loss is only a training-stream sanity signal.

Run one arm from the repository root (choose a fresh output path):

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 ../.venv/bin/python operator-studies/newton-muon-refresh/refresh_control.py \
  --text results/newton-muon-cost-screen-v1/text.txt \
  --output results/newton-muon-refresh-v1/d256-r16.json --width 256 --refresh 16
```

Run each `(width, refresh)` arm in a fresh process. The runner refuses to
overwrite output, requires fallback disabled, checks finite loss and Cholesky
status, and verifies exact per-matrix refresh counts (8, 4, and 2 over the
measured horizon). The frozen cost-screen source and text hashes are recorded.

This control reports end-to-end milliseconds per step and refresh counts. It
does not test time-to-target loss, held-out quality, large models, optimized
kernels, or memory peak. A cheaper interval is a cost observation, not an
optimizer win. Preserve failures and partial results.

Tests:

```sh
../.venv/bin/python -m unittest discover -s operator-studies/newton-muon-refresh -p 'test_*.py' -v
```

## Run result (2026-11-17)

The provenance-complete fresh-process runs are in
`results/newton-muon-refresh-v1-v2/`. All six arms completed with no warnings,
no Cholesky failures, finite losses, and exact refresh counts. The initial
model hash matches across refresh values within each width and the batch-stream
hash matches across all six arms.

| Width | Refresh | Median ms/step | Loss first → last |
|---:|---:|---:|---:|
| 256 | 16 | 38.215 | 5.7302 → 2.6399 |
| 256 | 32 | 37.103 | 5.7302 → 2.6518 |
| 256 | 64 | 37.194 | 5.7302 → 2.6621 |
| 512 | 16 | 134.243 | 5.6836 → 2.6484 |
| 512 | 32 | 131.445 | 5.6836 → 2.7029 |
| 512 | 64 | 129.381 | 5.6836 → 2.6683 |

At this scale, less frequent refresh was cheaper in these short windows: about
2.9% lower median cost at width 256 for refresh 32 versus 16, and about 3.6%
at width 512 for refresh 64 versus 16. The loss traces are not held-out or
time-to-target measurements, so these are cost observations and a sanity check,
not evidence that any refresh value is a better optimizer. The four windows are
within one process and are not independent uncertainty intervals. Earlier
`results/newton-muon-refresh-v1/` files are retained as a pre-provenance pilot;
they are not used for this table.
