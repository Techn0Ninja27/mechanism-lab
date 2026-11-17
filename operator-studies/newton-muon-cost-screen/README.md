# Newton–Muon baseline cost on this Mac

2026-11-17. This is a necessary cost screen for the
[generated low-rank optimizer proposal](../../reports/optimizer-problem-v1/README.md).
It does not implement that approximation, test time-to-target loss, establish
memory savings, or demonstrate a new optimizer.

## Question and controls

Does covariance preconditioning take enough of a small real-text training step
to justify further work? We compare plain Muon with full Cholesky Newton–Muon
on six-layer byte-level causal Transformers of width 256 and 512. Both use the
same initialization, batches, model, FP32 arithmetic, optimizer learning rate
and weight decay. The main source is the
[pinned author implementation](https://github.com/zhehangdu/Newton-Muon/blob/df78af0db523d8bceb25af4919a3e3e7082b80f3/train_gpt_newton_muon_1.py).

Its details matter: covariance updates only every 32 steps; inverses are
identity until the first refresh; damping is 0.2 times mean covariance diagonal;
raw gradients are right-preconditioned before momentum; QKV orthogonalizations
are separate; MLP contraction uses four covariance blocks. The source's NS
iteration and update scaling are retained, but FP32 NS, generic PyTorch kernels,
a small model and untied token/head weights make this a development port, not
an exact reproduction of the optimized CUDA training pipeline.

Native implementation and protocol agents worked separately. Root review found
and corrected first-draft bugs before any cost run: one rather than five NS
iterations, wrong block state shape, missing block damping, covariance work in
the plain control, uncleared matrix gradients, wrapped target labels, missing
weight decay/scaling/QKV splitting, and incorrect timing boundaries. Source
inspection also corrected an initial every-step covariance interpretation.
The initial protocol freeze is retained locally; the amended protocol and code
were frozen before training. No buggy draft timings were produced or selected.

## Data and timing

The input is 96 real January-2025 paper abstracts, selected deterministically
from the existing local corpus, totaling 138,012 UTF-8 bytes. Another 32 documents
are reserved but unused. These are real text, not synthetic labels; this is not
a representative language-model benchmark. Source IDs and text hashes are in
`text-manifest.json`; abstracts remain local. Batch=4, context=64, dropout=0,
seed=0, byte vocabulary=256.

Each arm runs 8 warmup steps and three consecutive 32-step end-to-end windows.
The timer synchronizes only at window boundaries. Batches are already on MPS;
data loading/transfers are outside timing. A separately initialized arm runs
8 warmup and 32 instrumented steps with synchronization around components.
Those barriers change overlap: component fractions must not be added to or
substituted for end-to-end timing. The first refresh occurs inside a measured
window; compilation/allocator effects are not fully separable from this small
screen. Consecutive windows are not independent runs or uncertainty intervals.

The preceding [capability probe](../newton-muon-kernel-probe/README.md) detected
an internal CPU fallback for `eigh`. This cost screen uses Cholesky and no
eigendecomposition. It requires `PYTORCH_ENABLE_MPS_FALLBACK=0`, records warnings,
and checks Cholesky info and finite losses outside timed windows. No fallback
warning appeared in the reported training runs; this is not a low-level proof
that every backend operation stayed on the GPU.

## Reproduction

Python 3.12.14, PyTorch 2.15.0 and NumPy 2.5.4 are pinned in `requirements.txt`.
Prepare the same text from the recorded local corpus:

```sh
python operator-studies/newton-muon-cost-screen/prepare_text.py --corpus results/corpus-ml-complete-mirror-v6/corpus.sqlite --output results/newton-muon-cost-screen-v1
python -m unittest discover -s operator-studies/newton-muon-cost-screen -p 'test_*.py'
PYTORCH_ENABLE_MPS_FALLBACK=0 python operator-studies/newton-muon-cost-screen/cost_profile.py --text results/newton-muon-cost-screen-v1/text.txt --output results/newton-muon-cost-screen-v1/new-d256.json --width 256 --device mps
```

Repeat the final command with `--width 512` and a fresh output filename. Existing
outputs are refused. Timings naturally vary. The frozen code/data hashes and
all measured windows are retained. Seven tests cover inverse/block application,
refresh schedule, NS recurrence, preconditioning-before-momentum, QKV split and
weight decay, gradient clearing and true next-byte labels.

## Decision limits

The 5% component-cost rule is only a necessary screening criterion. A native
reviewer initially interpreted slower per-step Newton–Muon as falsifying the
proposal; that conclusion is not adopted. Fewer steps could still yield better
time-to-loss. Conversely, an appreciable cost fraction does not show that the
low-rank proposal removes it or preserves quality.

Still required before low-rank training: source polynomial-inverse and larger
refresh controls, measured low-rank extraction/apply cost on real logged K/G,
operator fidelity, and evidence that full Newton–Muon is a worthwhile tiny-scale
time-to-loss target. No held-out quality evaluation has been performed. End-of-arm
MPS memory snapshots are explicitly not peak memory; logical covariance and
inverse tensor sizes are reported separately. No API calls or model downloads
were needed for this screen.

## Measured result

Times below are medians of three consecutive windows, divided by 32 steps.
Repeat 1 is a fresh process with the same inputs and arm order. Neither the
windows nor these two repeats provide a robust population confidence interval.

| Width | Process | Muon ms/step | Newton–Muon ms/step | Ratio | Instrumented preconditioning fraction |
|---|---:|---:|---:|---:|---:|
| 256 | Initial | 34.48 | 36.92 | 1.071 | 14.93% |
| 256 | Repeat | 36.65 | 38.52 | 1.051 | 14.75% |
| 512 | Initial | 119.86 | 130.20 | 1.086 | 10.38% |
| 512 | Repeat | 126.80 | 134.05 | 1.057 | 10.18% |

All losses stayed finite and decreased on the training stream. No held-out
loss was evaluated. End-to-end runs recorded three refreshes per matrix module;
instrumented runs recorded one, matching the protocol. Plain Muon stored no
covariance or inverse. Newton–Muon stored 10.5 MiB each of covariance and inverse
at width 256, and 42 MiB each at width 512, excluding momentum, activations and
other allocations. Replacing the inverse would not remove covariance storage.

**The necessary cost screen is met, but the candidate is not ready for training.**
Full Newton–Muon adds measurable per-step cost in this implementation. That
supports testing whether a cheaper preconditioner can preserve its benefit;
it does not establish that full Newton–Muon improves time-to-loss here, or that
the proposed approximation saves any of this cost. Generic kernels, FP32 and
host dispatch may account for part of the overhead. In particular, the source's
optimized polynomial inverse and a larger refresh interval remain untested.

Code/data hashes matched the pre-run freeze in every process. Timings are not
byte-reproducible; the raw windows are preserved in the four `d*.json` files.
The fresh-process variation is why the result is reported as a range of observed
costs rather than a precise hardware speedup claim.
