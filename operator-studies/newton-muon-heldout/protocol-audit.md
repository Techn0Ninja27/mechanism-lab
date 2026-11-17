# Held-out Newton--Muon decision control

Protocol audit, 2026-11-17. This document is a pre-run design review for the
cheapest experiment that can decide whether the local full Newton--Muon
candidate deserves approximation work. It is not evidence about optimizer
quality and does not make a claim about other models or datasets.

## Question and scope

Compare the frozen full-Cholesky Newton--Muon arm (`newton_cholesky`, refresh
32) with the frozen Muon arm on the existing tiny byte Transformer at width
256. Use two seeds, 0 and 1, 256 training steps, and checkpoints at step 0 and
every 32 steps. Training uses the existing 96-document `text.txt`; evaluation
uses the reserved 32-document `unused-validation.txt` from the same frozen
manifest. The files and their document IDs must be disjoint and their SHA-256
hashes must be recorded in the result.

This is deliberately a local falsification screen. Only a strict positive
result under the pre-registered allocation rule below justifies a larger
held-out control before any low-rank or polynomial timing work. A negative or
mixed result closes this optimizer candidate at the current settings and does
not justify additional approximation engineering. Neither outcome establishes
a universal ranking of optimizers.

## Fairness requirements

For each seed, construct both arms from the same explicit model initialization
and the same training batch stream. Counterbalance arm order by seed (for
example, Muon then Newton for seed 0 and Newton then Muon for seed 1). Use one
fresh process per arm and record the process order. The optimizer, learning
rates, weight decay, momentum/Nesterov rule, model architecture, precision,
batch size, sequence length, and refresh schedule must come from the frozen
cost-screen harness; only `mode` changes.

Training timing starts at the first measured training step and includes model
forward/backward, optimizer steps, and synchronization only at checkpoint
boundaries. It excludes validation passes and result serialization. Do not
copy every loss to CPU or synchronize every step. The Newton arm must fail on
any MPS fallback warning or Cholesky failure. Muon must run under the same
fallback policy. Record per-step or checkpoint boundary times and cumulative
training wall time.

The validation set must be deterministic and independent of training. Convert
the complete validation file to bytes, use fixed non-overlapping teacher-forced
64-token windows, and batch consecutive windows deterministically. Drop only a
final incomplete window and record the exact token/window counts. Run the
model in evaluation/no-grad mode at each checkpoint; do not call either
optimizer, mutate optimizer state, or include validation time in training
timers. Restore training mode before continuing. Validation loss is the mean
cross-entropy over the same fixed windows, not a sampled minibatch estimate.

One subtlety is that the frozen training generator samples random byte starts;
that is acceptable only if the batch stream is generated once per seed and
reused byte-for-byte by both arms. The validation stream must not use that
generator or draw from the training file.

## Pre-registered analysis

Report the raw held-out loss at every fixed-step checkpoint and the training
wall time at the corresponding checkpoint. The fixed-step endpoint after 256
updates is primary for describing quality. Since the arms may take different
time per step, additionally compare the curves at the last common elapsed
training-time horizon (the shorter final time). Report both declared linear
interpolation between adjacent checkpoints and the last observed checkpoint at
or before that horizon; the latter is a sensitivity check, not an additional
measurement. Never claim a time-to-target
win unless a target is specified before the run and both arms cross it; if
neither crosses it, report censoring. Do not smooth away non-monotonic losses.

Also report, descriptively:

* endpoint held-out loss after exactly 256 updates;
* held-out loss at the last common elapsed-time horizon, with that horizon
  printed explicitly;
* trapezoidal area under the held-out-loss versus training-time curve over the
  common observed horizon;
* paired Newton-minus-Muon differences at each common-time checkpoint and
  separately for each seed;
* training loss, finite-loss status, fallback warnings, Cholesky status,
  model/input/source hashes, and exact validation window counts.

With two seeds this is a paired descriptive screen, not a significance test or
calibration exercise. A single favorable seed is insufficient evidence for
escalation. Predeclare the qualitative action rule: proceed to a larger
held-out/time-to-target study only if Newton is lower at the fixed 256-update
endpoint for both seeds, lower at the last common elapsed-time horizon for both
seeds, and its paired time-area is lower for both seeds. If either seed is worse
on both fixed-step endpoint and time-area, stop this candidate
on this workload. Otherwise classify the result as mixed and close the
candidate at these settings; do not spend on low-rank/polynomial engineering.
This rule is an allocation rule based on sign consistency and dominance, not a
claim that the two-seed threshold is statistically meaningful.

The common-time comparison must use the shorter final training time of the two
arms for each seed; extrapolation beyond observed time is prohibited. If the
arms finish at different step counts due to failure, the run is incomplete and
the qualitative rule is not applied. Preserve failed and partial artifacts.

## Is 256 steps enough?

It is not enough to establish a general optimizer result, and a late benefit
could be missed. It is deliberately budget-limited: the existing source
harness already gives a finite 96-step trace, so this screen asks whether a
paired held-out signal is strong enough to justify spending more. The
checkpoint curve and raw losses make a null or crossing result visible rather
than hiding it behind one endpoint. A negative result closes only this
candidate on this workload; it does not reject the optimizer generally. If
both seeds satisfy the strict escalation rule, run a pre-registered longer
control (at least 1,024 steps, more seeds, and a second held-out corpus) before
claiming a useful optimizer.

## Reproducibility and stop criteria

Freeze this protocol, the runner hash, Python/PyTorch versions, source and
data hashes, command lines, arm order, device, seed, and environment before
execution. A result is invalid for comparison if the model initialization or
training batch hashes differ within a seed, validation windows differ between
arms, fallback occurs, a Cholesky factor fails, losses are non-finite, or
validation updates optimizer/model state. Stop and retain the partial files;
do not silently replace the arm or relax the rule.

The only supported claim from this experiment is a bounded local decision about
whether to continue this candidate. It does not measure novelty, usefulness of
research proposals, large-model behavior, optimized CUDA kernels, or the
eventual value of polynomial/low-rank approximations.
