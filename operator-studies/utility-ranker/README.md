# Task-conditioned utility ranker pilot

This pilot asks a narrow engineering question: can cheap, pre-encoding
measurements of an integer stream predict which *already executable* lossless
operator chain will use the fewest bytes on that stream?  The target is
compression utility, not novelty, scientific usefulness, or a claim about
general research agents.

The study uses the existing typed operator registry and integer codec
implementations in `generation/operator_composition.py` and
`operator-studies/integer-composition/`.  A task is a fresh raw stream plus a
fixed candidate library (at most 20 chains).  Every candidate is actually
encoded, decoded, and round-trip checked before it can receive a label.  A
failed or non-lossless candidate is invalid and is never treated as a cheap
but good result.  The byte label is measured after all common and
method-specific framing, including the fixed raw-signed-64 alternative.

## Split and leakage controls

The split is by whole task and by stream-group, never by rows from the same
stream.  A stream-group includes a family and hidden parameter-regime bucket;
near-duplicate parameter settings stay in the same split.  No task-family identifier, seed, stream-group ID, measured compressed
size, winning chain, or held-out result is an input feature.  Features are
computed from the raw stream before any candidate is run: length, min/max and
range, mean/std/variance, first-difference mean/std/zero rate, run statistics,
sampled entropy, and value bit-width summaries.  Chain features are only
predeclared structural indicators (operator one-hot/counts, depth, and
whether the chain contains a delta, zigzag, varint, block, or general
compressor); they are not derived from its output.

Train uses four families: `walk`, `runs`, `periodic`, and `uniform`.  A
separate validation split uses families not used for fitting (`clipped_noise`,
`modular_ramp`, `alternating_burst`) and is used only for model/regularization
choice.  The final held-out test uses three further families:
`piecewise_jump`, `regime_switch`, and `bitwidth_extremes`.  Their generators,
seeds, and labels must remain unopened until the parent explicitly authorizes
the final evaluation.  In particular, the exposed
`integer-composition/results.json` and prior final results are not training,
validation, or test rows.

There are 120 train tasks, 60 validation tasks, and 60 held-out tasks, with
20 candidate chains per task (4,800 candidate/task rows total).  Seeds are
disjoint and start at 20,000.  A task has one stream only; multiple views or
chunks of one stream never cross a split.  The generator should make each
family contain several parameter regimes, so memorizing a family mean is not
enough.

## Label and utility definition

For task `t`, candidate `c`, let `B(t,c)` be the exact encoded byte count and
`B_raw(t)` the exact fixed raw-signed-64 framed byte count.  The primary
regression target is

`gain(t,c) = (B_raw(t) - B(t,c)) / B_raw(t)`.

This is a relative byte saving, so chain choice is invariant to stream length
while the evaluator still reports absolute bytes/sample.  A candidate is
eligible only when `decode(encode(x)) == x`; otherwise its gain is missing and
its predicted score is masked out at selection time.  The training table must
retain `valid_roundtrip`, `bytes_total`, `bytes_per_sample`, and `gain`, so
invalid behavior is auditable.  The model never receives `gain` or any
candidate output feature as an input.  Do not add novelty labels or human
ratings: this experiment does not measure those constructs.

The strongest fixed baseline is the single chain with the best mean training
gain among valid rows, frozen before validation.  Comparators are raw,
uniform-random valid chain, and an oracle that chooses the best valid chain per
task.  A model may select only among valid candidates in the candidate
library; if no candidate improves raw, raw is selected.

## Models and preregistered metrics

Fit a small baseline set: ridge/linear regression on standardized features,
random forest (or ExtraTrees if available), and pairwise logistic ranking
trained on within-task candidate pairs.  Choose model and hyperparameters on
validation only.  Use fixed seeds, no tuning after inspecting held-out rows,
and report the selected configuration and feature schema.  If sklearn is not
available, the linear model and a numpy pairwise scorer are sufficient; this
pilot is intentionally CPU-cheap.

Primary endpoint is mean selection regret on held-out tasks:
`gain(t, oracle) - gain(t, selected)`, with lower better.  Also report
held-out relative gain, top-1 oracle-chain rate, fraction of tasks beating raw,
and per-family results with bootstrap 95% confidence intervals over tasks.
Report train/validation/test separately, plus fixed-best-train and random
baselines.  A useful result requires lower held-out regret than fixed-best
train and random, with the direction stable across all three held-out
families; otherwise record a negative or inconclusive result.  This criterion
is not a novelty claim and does not justify retraining on held-out tasks.

## Reproduction and stopping rules

The implementation should expose one command that regenerates only
train/validation rows, fits the predeclared models, and writes a manifest with
code hash, dependency versions, seeds, split counts, feature names, and row
hashes.  A separate explicit `--final`/approval gate may generate the held-out
families exactly once.  Never call the final generator during development or
use already exposed results as a substitute.  If a codec, feature, or split
violates the contract, stop and record the negative result rather than
silently repairing labels.

