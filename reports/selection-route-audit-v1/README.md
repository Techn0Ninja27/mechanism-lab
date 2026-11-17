# Selection-route audit and correction

2026-11-17. The goal is useful, testable ML proposals. This audit checks whether
the current selection code justifies calling an output highly ranked.

## Findings and changes

`local_runner/proposal_v2.py` said its seeds were unranked, but called
`rerank(..., beta=.6, ...)` on the legacy saved pool before discarding scores.
Thus the invalid atom/scorer mapping still determined which concepts were
selected. A disclaimer did not remove that influence.

The selector now draws unique triplets directly from IDs with original source
assignments in `papers/clusters.json`. A lazy seeded permutation samples the
combination space without materializing every tuple. Reviewed combinations are
excluded. Its manifest explicitly identifies uniform selection and denies
ranking claims. It is a baseline, not a substitute claim that random selection
is better or that the learned ranker has been repaired. No scored-pool file is
read by this path. Prior run artifacts are preserved.

On the real catalogue, seed 45 selected six triplets from 273 eligible atoms
(3,353,896 possible triples), excluding 13 previously reviewed triples. A fresh
output directory reproduced the selection bytes exactly. `selection-audit.json`
contains IDs, hashes and scope. These combinations were not decoded or tested,
and should not be presented as new research proposals.

Separately, `generation/dual_ranker.py` set `promoted_confidence=True` whenever a
selected row had all supplied numeric axes. Completeness does not demonstrate
calibration or truth. Selection now always leaves that flag false. Its supplied
score/Pareto ordering and exploration behavior remain available, but they do
not increase scientific confidence. No production caller of this scaffold was
found in generation routes inspected for this audit.

## What remains unresolved

The active grounded proposal pipeline generates one proposal plus a critique;
it does not compare and select among a pool using a validated usefulness ranker.
The original beta scorer remains an unvalidated numerical diagnostic elsewhere.
The small source-coherence model measures co-occurrence, not scientific value,
and availability was not repaired by training it. The narrow utility model's
previous held-out failure still stands. Neither larger corpus size nor these
code fixes establish novelty or usefulness prediction.

The workbench still permits inspection of historical `proposals-v2` artifacts
and offline sampling/reranking diagnostics. Those routes are not repaired by
this selector change. Candidate rows are labeled untrusted; prompt export warns
about rankings. The webapp's existing default is a historical run, and old saved
selections do not retroactively acquire this new selection provenance. No live
UI actions or paid decoders were invoked in this audit. Use the current budgeted
`grounded_research.py` pipeline for proposal work rather than assuming the legacy
decoder has its shared-budget behavior.

The next meaningful ranking experiment must compare methods on the same
candidate pool and source/model budget, with strong controls and an untouched
task split. Numeric ratings from the same generating model are not outcome
labels. We currently have several scoped negative experiments and insufficient
diverse success/failure evidence to claim a trained general usefulness model.

## Verification

Seventeen targeted tests pass. They include exhaustive coverage of a small
combination space, exclusions, assignment-order invariance, repeatability,
source-less/outlier exclusion, and a subprocess running the real legacy selector
against a fixture repository with **no scored pool at all**. Another regression
checks that complete high supplied scores do not promote confidence.

```sh
python -m unittest tests.test_exploratory_seeds tests.test_dual_ranker
```

The real selection used explicit `ALIEN_WORKSPACE`, `ALIEN_REPO`, and fresh
`PROPOSAL_OUTPUT` paths with `PROPOSAL_SEED=45`, then ran
`python local_runner/proposal_v2.py select`. No API calls, corpus downloads or
model training occurred. This is evidence of correct selection plumbing, not
better proposal quality.
