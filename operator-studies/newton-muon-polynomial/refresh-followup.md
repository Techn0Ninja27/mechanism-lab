# Refresh-interval control: 16, 32, and 64 steps

Status: protocol only (2026-11-17). This document specifies the next bounded
control for the Newton--Muon cost screen. It is not a result and makes no claim
that a less frequent covariance refresh improves training.

## Question

On the same tiny real-text Transformer and the same Cholesky Newton--Muon
implementation, does changing the covariance/inverse refresh interval from the
source-aligned value of 32 steps change end-to-end cost or the observed training
loss trajectory? The comparison is a control for the cost screen and for the
later polynomial-inverse experiment. It does not establish time-to-target loss,
large-model behavior, or optimizer quality.

## Matched design

Use the frozen code/data and environment from `newton-muon-cost-screen-v1`:

* widths 256 and 512; six layers; four heads; byte vocabulary 256;
  batch 4, sequence length 64, dropout 0, FP32, seed 0;
* exactly the recorded 96 January-2025 abstracts (`text.txt` and its SHA-256),
  with the same batch generator and 136 preloaded batches (8 + 128);
* the same model initialization, parameter partition, learning rates, weight
  decay, momentum/Nesterov rule, Cholesky damping, covariance EWMA, block
  layout, gradient clearing, and `PYTORCH_ENABLE_MPS_FALLBACK=0` requirement;
* one fresh process per `(width, refresh)` arm, with the arm order fixed in the
  manifest. A separate repeated process may be added only if all primary arms
  complete and the budget remains available.

Run refresh values **16, 32, and 64**. Every arm performs 8 warmup steps and
128 measured steps, timed as four consecutive 32-step windows. Synchronize the
device only at the warmup/window boundaries, as in the original end-to-end
screen. Do not add component-timer fractions to the end-to-end time. The first
refresh is intentionally allowed to occur in a measured window; report its
location rather than moving it outside the timing interval.

The 128-step horizon is long enough to contain eight, four, and two refreshes
after warmup respectively. Record the exact step numbers and per-window
refresh counts. The 32-step arm must reproduce the existing refresh schedule
under this longer horizon; if it does not, stop and diagnose the harness before
comparing arms.

## Required harness changes

Add a separate, versioned runner or explicit `--refresh` and `--steps` options;
do not edit the frozen cost-screen outputs. The runner must:

1. refuse an existing output path;
2. write the complete command, Python/PyTorch/platform versions, source/data
   hashes, seed, device, arm order, refresh value, warmup, window length and
   number of windows into each result;
3. capture warnings and fail if a fallback warning, non-finite loss, failed
   Cholesky factor, missing refresh activation, or unexpected refresh count is
   observed;
4. keep all raw windows and per-step losses, while reporting medians across
   windows only as descriptive summaries;
5. synchronize at the same boundaries for every arm and exclude preparation,
   text loading, and result serialization from timed windows;
6. report covariance and inverse logical bytes separately from end-of-run MPS
   snapshots, which are not peak memory measurements.

Use the same arm initialization and batch stream for each refresh value. A
single process running all three arms is acceptable for smoke testing, but the
primary timing comparison should use fresh processes to reduce allocator and
compilation carry-over. Keep arm order fixed and disclose it.

## Measurements and analysis

For each width and refresh value, report each 32-step window in milliseconds
per step, the four-window median and range, first/last training loss, all
per-step losses, refresh counts per matrix module, Cholesky status, warnings,
and state sizes. Compare 16/32/64 descriptively within width. Treat the
existing 32-step measurements as a historical reference unless they are
re-run with this exact 128-step protocol; do not splice windows from different
horizons into a confidence interval.

The primary cost quantity is end-to-end measured milliseconds per step. A lower
per-step cost is only a cost observation: it does not imply fewer steps to a
target loss. The loss trace is a sanity check for divergence and a descriptive
quality signal. No claim of optimization quality should be made without a
pre-registered held-out stream and a time-to-target analysis.

## Stop criteria and negative outcomes

Stop the run if any arm violates the fallback, finite-loss, Cholesky, refresh
count, source-hash, or matched-batch checks. Preserve the partial outputs and
write a failure note; do not silently drop an arm or substitute a different
refresh value. If width 512 is too slow or memory constrained, complete width
256, record the resource failure, and label the comparison incomplete.

Do not continue to polynomial timing, low-rank extraction, or training-quality
claims based only on this control. A result in which refresh 64 is cheaper but
has an unmeasured time-to-loss penalty is a valid cost/quality trade-off, not a
win. A result with indistinguishable windows is also useful: it provides no demonstrated scheduling gain in this short measurement,
and does not establish a statistical bound on possible benefits.

## Reproduction

The eventual runner should expose one command with explicit paths and no
network/API calls, for example:

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 python operator-studies/newton-muon-polynomial/refresh_control.py \
  --text results/newton-muon-cost-screen-v1/text.txt \
  --output results/newton-muon-refresh-v1 \
  --widths 256 512 --refreshes 16 32 64 --warmup 8 --steps 128 --window 32
```

Pin the existing environment from `operator-studies/newton-muon-cost-screen/requirements.txt`.
The command above is a target interface for the implementation; this protocol
file alone does not create or run that script.
