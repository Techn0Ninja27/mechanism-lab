# Released-model audit — 2026-11-16

The released checkpoints are deliberately small relative to the paper corpus, but their training targets make several assumptions worth testing before changing architecture.

## What is actually trained

The coherence checkpoint is a 42.8M-parameter, six-layer causal Transformer over 273 atom IDs. Each paper with at least two atoms yields ten random permutations before the train/validation split. The released metadata reports 15,203 papers and 152,030 examples, split as 136,827 train and 15,203 validation records. The split is not paper-grouped: the validation rows contain 9,903 paper IDs that also occur in train (of 9,903 validation papers and 15,203 train papers). This makes the released validation loss unsuitable as a clean paper-level generalization estimate.

The availability checkpoint is a 26.1M-parameter two-tower model with 39,791 author embeddings in the inference bank. It trains on query/complement views of author repertoires using contrastive loss. The run config says four GPUs, global batch 4,096, 100 epochs, and `compile=true`. The author bank is not a neutral population prior: it is built from the training author set and its top-ten median is the operational definition of “community availability.”

## Bounded diagnostic

Run:

```sh
../.venv/bin/python scripts/model_diagnostics.py --n-random 256 --seed 123 --device cpu
```

On a seeded sample of 256 uniform random triplets and 256 triplets sampled without replacement from the unique real-paper triplet pool, the released scorers produced:

- the mixed-pool correlations are reported in `results/model-diagnostics/model_diagnostics.json`, together with source-stratified correlations;
- all 273 atoms appeared among papers with at least two atoms; frequencies ranged from 18 to 872 papers (median 136).

These are not global population estimates: the candidate source and finite sample matter. The diagnostic now reports random-only, real-only and overlap strata separately, so a correlation cannot be attributed to the mixture label. Persistent within-stratum correlation would motivate popularity controls; this audit alone does not prove a causal confound.

## Massive-data implications

Adding more papers while keeping the 273-atom vocabulary fixed mostly improves co-occurrence counts and author coverage; it does not create new conceptual primitives. The first cheap test is therefore a time-sliced re-estimation with the existing vocabulary, measuring whether score correlations and top-k overlap stabilize as paper count grows.

Adding papers and rebuilding atom extraction/clustering changes the state space. The checkpoints cannot score the new vocabulary. It requires new conceptual-unit extraction, clustering, canonical descriptions, coherence training and availability training. More data can also increase leakage if author or paper identities cross temporal boundaries.

## Prioritized model changes and falsification tests

1. **Popularity residualization (cheap).** Add atom-frequency and paper-age controls; compare raw unavailability with residual unavailability after predicting it from atom frequencies and author coverage. Falsifier: residual scores produce the same top-k and the same future recovery as raw scores, making the control unnecessary.
2. **Grouped temporal validation (essential).** Split by paper year and author identity, not by permutation rows. Hold out a future year and unseen authors. Falsifier: the current random/per-paper split matches the grouped temporal split, so leakage is not materially affecting results.
3. **Candidate-source ablation.** Compare autoregressive coherence sampling, uniform triplets, and a hybrid mixture under equal candidate budgets. Falsifier: no source changes recovered future combinations or portfolio diversity.
4. **Longer author history sensitivity.** Train/evaluate availability with author repertoires truncated by time and with held-out authors. Falsifier: rankings are invariant to author-history truncation and unseen-author evaluation.
5. **Only then change capacity.** A larger transformer or deeper set encoder is not justified until the above tests show underfitting rather than confounding or split leakage.

## Limits of this audit

The diagnostic uses the released checkpoints and a 512-candidate sample. It does not reproduce the authors’ exhaustive 3.35-million-triplet ranking, human quality judgments, or historical 2025 backtest. No scientific novelty claim follows from a high score.

The released row split finding does not invalidate the separately frozen historical backtest described by the authors. Our retraining evaluation must be independent of these permutation-row splits.
