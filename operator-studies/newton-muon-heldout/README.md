# Held-out Newton–Muon versus Muon control

Status: completed development screen, 2026-11-17; pre-run rule not met. This is a bounded local
falsification test for the tiny development harness, not a claim about large
language models.

The control compares source-aligned Muon with full-Cholesky Newton–Muon at
width 256, refresh 32, using two seeds. Each arm trains for 256 steps on the
first 96 frozen January-2025 abstracts and is evaluated at steps 0, 32, …, 256
on the disjoint final 32 abstracts. Training and evaluation use byte tokens,
sequence length 64 and batch size 4. Evaluation windows are non-overlapping;
unused tails are reported rather than silently padded.

Each arm is run in a fresh process. Training timing starts after batch/model
preparation and includes synchronization at checkpoint boundaries. Evaluation
time is excluded from the training-time curve. Muon and Newton–Muon receive
the same seed, batches and initial model state for each seed. Seed 0 runs Muon
then Newton–Muon; seed 1 reverses that order to reduce process-order bias.
Loss checkpoints and cumulative training wall time are descriptive. The fixed
endpoint and loss-versus-time curves are inspected without choosing a target
after seeing the results. No claim is made without an independent repeat or a
larger benchmark.

The runner records source, split, model, batch-stream and runner hashes,
device/fallback warnings, Cholesky status, all checkpoint losses, and the
evaluation batch construction. It refuses to overwrite output and does not
make network or API calls.

Run one fresh arm from the repository root:

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 ../.venv/bin/python \
  operator-studies/newton-muon-heldout/run_arm.py \
  --mode muon --seed 0 --order-index 0 \
  --train-text results/newton-muon-cost-screen-v1/text.txt \
  --heldout-text results/newton-muon-cost-screen-v1/unused-validation.txt \
  --manifest results/newton-muon-cost-screen-v1/text-manifest.json \
  --output results/newton-muon-heldout-v1/muon-seed0.json
```

The protocol deliberately runs only after review of this frozen document and
the unit tests. It is not a low-rank or polynomial test; it establishes whether
the full Newton–Muon baseline earns further approximation work on this local
task.

## Result: favorable endpoints, unfavorable full-curve average

All four arms completed. Initial model and training-batch hashes match within
seeds; both optimizers used the same 688 held-out windows (44,032 target
bytes), excluding 22 tail bytes and cross-window transitions. Validation did
not change model parameters or the matrix optimizer state. No fallback
warnings, non-finite losses, or failed Cholesky factors occurred.

| Seed | Muon final loss | Newton final loss | Muon training seconds | Newton training seconds | Newton − Muon common-time loss | Newton − Muon normalized time-area |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 2.520924 | 2.508002 | 8.965 | 9.653 | −0.013234 | +0.006300 |
| 1 | 2.545850 | 2.506398 | 9.011 | 9.684 | −0.032508 | +0.008419 |

Loss is mean next-byte cross-entropy in nats; lower is better. The common-time
endpoint uses linear interpolation at the shorter total training time per
seed. The last-observed checkpoint comparison also favors Newton in both
seeds (differences −0.013471 and −0.027062), so the endpoint direction does not
rely solely on interpolation. Newton nevertheless has a larger normalized
area under the full loss/time curve in both seeds.

The frozen allocation rule requires strict improvement in all three quantities
(fixed-step endpoint, common-time endpoint, and time-area) for both seeds.
It therefore returns `close_local_candidate_at_fixed_settings`. This is mixed
evidence, not a finding that Newton has no quality benefit: its endpoints are
better here. We do not change the rule after observing them. No low-rank or
polynomial implementation is promoted for further work on this fixture.

The 32-step checkpoint spacing makes the time-area a coarse piecewise-linear
estimate, especially during the steep initial loss drop. Timed training includes
initial training-kernel overhead after an initial validation pass. Validation
and data/model preparation time are excluded. These timings are a narrow
training-time comparison, not end-to-end job time. Two seeds, a tiny selected
corpus, one width and untuned shared hyperparameters cannot establish general
optimizer rankings. Later training benefits may be missed.

All per-arm JSON and [summary](result.json) are saved alongside the code.
[freeze.json](freeze.json) records runner, summary, audit and data hashes before
training; the original audit is preserved as [protocol-audit.md](protocol-audit.md).
No paid API calls or data/model downloads were used.

Replay the deterministic aggregation (choose a new output path):

```sh
../.venv/bin/python operator-studies/newton-muon-heldout/summarize.py --input operator-studies/newton-muon-heldout --freeze operator-studies/newton-muon-heldout/freeze.json --output /tmp/newton-heldout-summary.json
../.venv/bin/python -m unittest discover -s operator-studies/newton-muon-heldout -p 'test_*.py' -v
```

Fresh training reproduction uses the one-arm command above in the frozen order:
Muon seed0, Newton seed0, Newton seed1, Muon seed1, order indices 0–3.
MPS training/timing need not reproduce bitwise across fresh runs. The reserved
validation fixture is now exposed development evidence and must not be reused
as an untouched final test for a later candidate.
