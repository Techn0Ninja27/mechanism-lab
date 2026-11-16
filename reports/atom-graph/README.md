# Atom graph baseline result

Run date: 2026-11-16. This is a link-recovery diagnostic over the original
assignment release, not a novelty or usefulness claim.

There are 30,580 raw non-noise assignment rows, which collapse to 22,150
unique paper-atom incidences across 13,419 papers and 4,503 unique canonical
atom sets. The sets were split by complete set with seed 73031: 3,152 train,
675 validation, and 676 test. The train graph contains 4,070 observed edges
and 33,058 candidate pairs after removing those edges. A positive is a pair
that occurs in a target split and is absent from the train graph. Each unique
canonical set contributes once to each pair; no citations or popularity
weights were used.

The fixed test results at k=100 were:

| method | precision | hits | recall |
|---|---:|---:|---:|
| random expected | 0.0187 expected | — | — |
| seeded random | 0.0100 | 1 | 0.0016 |
| common neighbors | 0.2100 | 21 | 0.0340 |
| Adamic–Adar | 0.2300 | 23 | 0.0373 |
| degree product | 0.2100 | 21 | 0.0340 |
| TF-IDF cosine | 0.1000 | 10 | 0.0162 |

At k=10, common neighbors and Adamic–Adar each recovered 4 held-out pairs
(precision 0.40); degree product recovered 4 and TF-IDF cosine 2. The graph
methods beat the random controls on this exposed split, but this mostly shows
that repeated literature combinations are predictable. It does not show that
the pairs are useful, surprising, causal, or unpublished. The TF-IDF vectors
use all 273 definition strings, so the representation is transductive and the
evaluation is not a temporal novelty test.

The input graph is explicitly undirected co-occurrence. These edges must not be
copied as typed molecule relations without a separate mechanism interpretation,
prior-art review, and falsifiable experiment. The top pairs are raw candidates
for that next stage.

The frozen results file contains lexicographically ordered ranking metrics under
the random_expected key; those fields are not random expectations. Only its
expected_precision fields represent random expectation. The corrected
labels and expected-vs-seeded control separation are in
corrected-summary.json. The top 20 pairs per method are in
top-candidates.json. All 273 vocabulary nodes occur in the training graph;
there are no zero-degree nodes.

TF-IDF uses cluster name strings only, not full paper abstracts or mechanism
annotations. Citation metadata collection for the expanded corpus is still in progress,
and no such metadata was joined to these assignment IDs. This run has no citation
weighting. A future weighting ablation must accept explicit, source-linked
per-paper weights and remain separate from this frozen result.

freeze.json and results.json contain the source hashes and complete metrics.
The study uses no API calls. Re-run:

    python scripts/run_atom_graph.py --output results/atom-graph-reproduction
