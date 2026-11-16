# Public summary: source-coherence pilot

A small causal transformer trained on original cluster assignments achieved
held-out token NLL 4.0354 on validation and 3.9868 on a test set selected only
after validation checkpoint selection. Train-fit unigram baselines were 4.3650
(validation) and 4.3186 (test); a permutation-symmetric conditional-pair
baseline was 4.7097 and 4.6625.

The result is limited to co-occurrence modeling. It does not establish
scientific novelty, discovery quality, temporal generalization, or a repaired
full ranker. The source snapshot used for the run is not exactly recoverable;
the preserved trainer is labeled an unverified reconstruction, and the
original manifest hash is retained unchanged. Availability scoring was not
retrained or fused.
