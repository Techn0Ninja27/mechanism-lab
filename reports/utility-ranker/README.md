# Utility ranker: held-out failure (2026-11-16)

A task-conditioned pairwise logistic model ranked 20 executable, known lossless
compression chains from raw-stream statistics and chain structure. It trained
on 120 synthetic streams, selected its configuration on 60 validation streams,
and was evaluated once on 60 streams from three further families. No family
names, seeds, compressed sizes, or test labels were model features.

| Mean selection regret (lower is better) | Validation | Held-out |
|---|---:|---:|
| Learned ranker | 0.00517 | 0.09412 |
| Best fixed chain selected on training | 0.02556 | 0.02238 |
| Random chain | 0.11366 | 0.18243 |

Regret is the difference in byte-saving fraction between the per-task oracle
and the selected chain. The learned ranker lost to the fixed baseline in all
three held-out families. The task-bootstrap 95% interval for its regret
reduction versus fixed was [-0.12026, -0.03083]. The preregistered usefulness
criterion failed. This model is **not promoted** into the research ranker.
The result neither measures scientific novelty nor establishes a general
research-utility predictor.

The largest error was on bit-width extremes (regret 0.24702 versus fixed
0.04602). This is descriptive diagnosis of an exposed test set, not a reason
to tune and retest on it. A subsequent experiment needs new frozen holdouts.

BLAS emitted divide/overflow/invalid warnings. Independent inference using
NumPy einsum found finite features, coefficients, and predictions; all selected
chains matched, with maximum score discrepancy 1.42e-14 on held-out data.
The warning's underlying cause remains unresolved. No fitting was repeated.

Frozen code, model hash, validation and final reports are adjacent. Local full
artifacts are in `results/utility-ranker-v1`. Reproduce in a fresh output folder
using the benchmark environment (Python 3.12, NumPy 2.2.6, scikit-learn 1.7.2,
SciPy 1.16.3):

```sh
../.venv-benchmark/bin/python operator-studies/utility-ranker/experiment.py train --output results/utility-ranker-reproduction
../.venv-benchmark/bin/python operator-studies/utility-ranker/experiment.py freeze --output results/utility-ranker-reproduction
../.venv-benchmark/bin/python operator-studies/utility-ranker/experiment.py final --output results/utility-ranker-reproduction
```

Protocol clarification made before final: eligibility comes from declared
input contracts; selection never masks candidates using their test outcomes.
A selected execution failure is recorded and receives raw fallback utility.
The initial study README's wording about masking invalid candidates is
superseded by the pre-final audit and frozen implementation. Pairwise scores
are ordering scores, not calibrated predictions of expected gain.
