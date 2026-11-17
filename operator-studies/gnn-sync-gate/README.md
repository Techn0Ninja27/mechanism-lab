# Sync-revealed expert gating: Cora trajectory screen

2026-11-17. This study tests the application proposal in
[the generation report](../../reports/gnn-useful-proposal-v1/README.md).
The operator is known Fixed Share mixing; no new algorithm is claimed.

The question is whether delayed observations at real synchronization endpoints
are useful for choosing between extrapolation and last-observed hold during
stale epochs. The experiment uses a real node-classification training trajectory,
then replays it with restricted information. It does not train the gated GNN in
closed loop, estimate communication speedups, or reproduce a distributed SNI-GNN
implementation. Test labels are not evaluated.

`protocol.json` was frozen before any training or trajectory inspection.
Cora contains 2,708 nodes, 1,433 features, 7 classes and 5,278 unique undirected
nonself edges. Raw citation edges are deduplicated before normalization. Features
and labels come from LINQS, not a synthetic labeling procedure. The custom
20-labels-per-class split is explicitly different from the Planetoid public split.
Random partitions support a cheap algorithm screen; they do not model METIS or
distributed hardware costs. Two seeds tune the gate rate and constant mixture;
two further seeds screen the frozen choices on the same exposed graph.

The predictor uses an Algorithm-2-inspired self-fed recurrence with lambda and
beta both 0.2. Warmup uses real observations. `h_sync` is maintained separately
from the evolving predictor state. At sync, the endpoint prediction is evaluated
before incorporating its observation. Only grouped endpoint losses update the
gate. Intermediate truth may change reported error, but never gate state or
predictions. Predictor state never consumes the gate mixture.

Code review caught errors in the first implementation before experiments:
feedback from the previous stale step rather than the endpoint, sequential
per-node rather than group updates, omitted self-fed slope terms, a squared
metric where the protocol called for relative L2, and superficial causal tests.
Those checks must pass before the corrected implementation is used. No metrics
from the initial implementation were produced or selected.

## Data preparation

Python 3.12.14, pinned NumPy/PyTorch in `requirements.txt`. No PyG/DGL installation
is required. From repository root:

```sh
mkdir -p results/gnn-sync-gate-v1/data
curl --fail --location --max-time 30 --max-filesize 1000000 https://linqs-data.soe.ucsc.edu/public/datasets/cora/cora.tar.gz --output results/gnn-sync-gate-v1/data/cora.tar.gz
python operator-studies/gnn-sync-gate/prepare_data.py --data results/gnn-sync-gate-v1/data
python -m unittest discover -s operator-studies/gnn-sync-gate -p 'test_*.py'
```

The archive was 168,206 bytes, SHA256
`4f67c1434d39c076036039597d5326edb32c9b03c91e007ad49831c01b794ba0`.
Retain this archive for exact reproduction; an unversioned upstream URL can
change. The preparation script extracts only the two named data files, not
arbitrary archive paths. Python's bundled certificate store initially rejected
the endpoint; system curl verified TLS and fetched it successfully.

## Result: candidate failed the frozen screen

The suite ran eight synchronous training trajectories (four seeds, two schedules).
Seeds 0/1 selected gate learning rate 10 and constant mixture weight 1, i.e. the
predictor alone. The table averages the separate screening seeds 2/3:

| Schedule | Adaptive gate relative L2 | Predictor alone | Hold alone | Gate / best control |
|---|---:|---:|---:|---:|
| Constant learning rate | 0.01285886 | 0.00616359 | 0.02936828 | 2.0863 |
| Learning-rate drop | 0.01150376 | 0.00571101 | 0.02386346 | 2.0143 |

Both required 5% improvement gates failed. The fixed mixture selected on the
development seeds equals the strongest control, so there is no screening-set
mixture tuning. Endpoint/shorter-horizon winner agreement was 99.73–99.92% on
screening trajectories; the source proposal's feedback-horizon concern did not
explain the failure in this setting. The selected gate still mixed in enough
hold to lose to prediction. We do not retune the rate/share or choose new shocks
on these screening results. This negative does not rule out other gates or
regimes; it does rule out advancing this particular configuration on this evidence.

The graph yielded 4,677 unique directed boundary node/destination-partition pairs
and 12 partition-pair groups. Each group received 24 endpoint updates. All
boundary pairs receive prediction; no importance subset is selected. The first
hidden layer is recorded before each optimizer step, no dropout is used, and
warmup observations initialize the trend. Relative-error denominators add 1e-8.
The recurrence retains exponential slope accumulators, not a finite-window
subtraction, and remains an explicitly labeled SNI-inspired surrogate.

Synchronous validation accuracies range from 0.728 to 0.814 as a training sanity
check only. They are not results for the gate. Test labels were not evaluated.
The stronger-baseline failure stops this candidate before closed-loop training,
a larger graph download or DPU implementation.

## Reproduce the screen

With the extracted data and pinned dependencies, one command runs the complete
suite (choose a fresh output path):

```sh
python operator-studies/gnn-sync-gate/screen.py --data results/gnn-sync-gate-v1/data --output results/gnn-sync-gate-v1/new-result.json --suite
```

A fresh Python process reproduced the full parsed JSON exactly. This was a full
repeat of all eight trajectories, not reuse of cached traces. Code and protocol
hashes in `freeze.json` match the execution; data hashes are recorded separately.
`reproduction.json` reports byte and parsed-JSON equality. Control dictionary
key ordering can differ between processes; it does not change any numeric result.

Six tests pass, including intermediate-truth invariance of predictions/weights,
future-prefix invariance, separate hold state, endpoint-before-observation loss,
group update counts and an independent scalar recurrence across sync boundaries.
Root review additionally corrected elapsed-time growth after sync, erroneous
winner-agreement denominators, and screening selection of an oracle mixture.
All corrections preceded the first training run. PyTorch warned that sparse
invariant checking is disabled by default; indices are constructed in range,
deduplicated, symmetrized and coalesced, and graph dimensions were checked.
