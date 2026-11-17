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
