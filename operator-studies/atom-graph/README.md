# Atom graph baseline

This study projects the original paper-to-atom assignments into an undirected
atom co-occurrence graph. An edge records only that two atoms appeared in the
same source paper. It is not a causal, directional, or semantic relation and
must not be written directly into a molecule as a mechanism edge.

The 4,503 unique original atom sets are split by whole set with seed 73031
(70% train, 15% validation, 15% test). The graph is built from training sets
only. Candidate methods rank pairs absent from the training graph: common
neighbors, Adamic–Adar, degree product, and cosine similarity of deterministic
TF-IDF vectors made from the 273 atom definition strings. Random expected
precision and seeded random sampling are controls. Top-k is fixed at 10, 50,
and 100; test is not used for tuning.

Because definitions are available for every atom, the TF-IDF vocabulary is
transductive. This is allowed as a descriptive representation, but the study
is not a temporal novelty test. A hit means that a held-out paper contains the
pair, not that the pair is useful, surprising, or unpublished.

Run with:

    python scripts/run_atom_graph.py --output results/atom-graph-reproduction

Original results and hashes are in reports/atom-graph/; reproduction writes
to the specified fresh directory. Any high-ranked pair
is only a raw candidate for prior-art retrieval and a typed molecule review.
