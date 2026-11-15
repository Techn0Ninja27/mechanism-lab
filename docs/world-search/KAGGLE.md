# From world-generated mechanisms to executable ML improvements

This note defines a small bridge from the Alien/World-Diversity search to
machine-learning engineering tasks. It is a benchmark design, not a claim
that a world-generated idea is novel or useful.

## What the existing benchmarks establish

**MLE-bench** is the closest reference point: it packages 75 offline Kaggle
competitions and grades agents on data preparation, modelling, experimentation
and submission quality. The official benchmark recommends three seeds and
reports any-medal rates with uncertainty; its full setup expects roughly 36
vCPUs, 440 GB RAM and a 24 GB GPU, so the full suite is not a sensible first
run on this Mac. The 22-competition low-complexity split is the intended
smaller evaluation. Sources: [paper](https://arxiv.org/abs/2410.07095),
[official repository](https://github.com/openai/mle-bench), and its
[offline instructions](https://github.com/openai/mle-bench/blob/main/environment/instructions.txt).

**MLAgentBench** evaluates an agent that proposes, executes and interprets ML
experiments. Its useful lesson for us is that success cannot be reduced to one
final score: experiment design, execution and interpretation are separate
capabilities. Source: [ICML paper](https://proceedings.mlr.press/v235/huang24y.html).

**AIDE** treats ML engineering as a searchable program space: an agent proposes
code, runs it, observes validation results, and expands promising branches.
That is valuable for the execution loop, but an improvement found by search is
not automatically a new research idea; the branch history and data split must
be retained. Source: [paper](https://arxiv.org/abs/2502.13138),
[code](https://github.com/bjzgcai/AIDE).

## Cheap first task

Use the **Kaggle Digit Recognizer** data only as a local smoke test. Kaggle's
rules permit non-commercial participation and academic research, but the
competition test labels are private and leaderboard submissions are outside
this experiment's scope. Source: [official rules](https://www.kaggle.com/competitions/digit-recognizer/rules).

Do not use the public leaderboard as the metric. Downloading the competition
data requires accepting Kaggle's rules and an account, so the default fixture
should be a deterministic, locally generated MNIST-style split (or a small
scikit-learn digits fixture) with a held-out test set that the agent never
reads. A Kaggle download may be added later by a human who has accepted the
rules. No credentials, submissions, or hidden labels belong in this repo.

The first local task should be deliberately small:

1. Create a fixed train/validation/test split with a recorded seed. Keep the
   test set untouched until the end.
2. Give the direct baseline and world-search arms the same data description,
   time limit, Python environment and evaluation budget.
3. Ask each arm for a mechanism-level proposal, not a prose trick: what
   measurable failure does it target, what intervention changes it, and what
   ablation would falsify the claim?
4. Let an implementation agent convert the proposal into a bounded experiment.
   Reject code that changes the split, reads test labels, or imports external
   leaderboard solutions.
5. Run each implementation from a clean checkout, retain stdout, metrics,
   random seeds, elapsed time and the exact diff. Score held-out performance,
   variance over at least three seeds, compute cost and whether the proposed
   mechanism was actually present in the code.

The useful outcome is not “highest accuracy.” It is the paired quantity:

`validated_gain = held_out_gain - control_gain`

where control gain is measured from an equally budgeted direct-search arm. A
proposal only counts as an improvement if its confidence interval excludes zero
or the pre-registered effect threshold and its ablation removes most of the
gain. A proposal is retained as a candidate research mechanism only if it also
survives prior-art retrieval and an independent reviewer can state the exact
difference from the closest method.

## World-search comparison

The 24-proposal pilot should use two tasks and four fixed seeds, with three arms
per task/seed:

* **Direct:** solver sees the task and produces an intervention.
* **Random seed:** a seed word changes the framing, but no causal world is
  modelled.
* **World detour:** a target-blind builder makes a world, a fresh solver solves
  the task inside it, and a compiler extracts a mechanism back into the target.

The builder must never see the target. The solver must not see the seed or the
reason for the world. The compiler must see the in-world solution but not the
other arms' outputs. Prompts, model, token cap and wall-clock budget are
recorded. Four seeds are not enough for a scientific claim; they are a cheap
falsification pilot.

Evaluate the final normalized proposals with blind reviewers and executable
tests. Report: distinct mechanism count after semantic clustering; fraction
with a precise falsification test; external-prior-art overlap; implementation
success; held-out metric gain; unnecessary-component count; tokens and time.
Keep the source condition hidden from the reviewer. Do not use model-rated
originality as the primary novelty label.

## Preventing the search from becoming a taxonomy trap

The Ontology-of-the-Alien release explicitly says its 196 records come from one
problem and path-dependent traces, and that usefulness and external novelty
were not tested. Its Taxonomist is still valuable as search state: it can merge
duplicates and direct exploration toward thin branches. To avoid early labels
becoming a new median, maintain two stores:

* a persistent, content-addressed mechanism ledger with definitions,
  provenance, rejected candidates and exact tests; and
* a rotating challenge set whose fresh reviewers can split or reject taxonomy
  categories without access to the previous verdict.

Only human-adjudicated, source-backed records may become training labels. A
model's “novel” response remains an unadjudicated hypothesis.

## Recommended sequence

1. Run the world/direct/random pilot on the local fixture before accepting
   Kaggle terms or downloading competition data.
2. Use the same evaluator to retrieve prior work for every surviving mechanism.
3. Port only the top few mechanisms to Digit Recognizer or one MLE-bench low
   complexity task, with a clean test split and three seeds.
4. Add a Kaggle task only after the local harness catches data leakage and
   records reproducible code. Never compare to the public leaderboard as if it
   were a held-out scientific test.

This keeps the attractive part of world search—the change in causal premises—
connected to an outcome that can actually falsify it: a reproducible improvement
under a hidden split, after controls and prior-art review.
