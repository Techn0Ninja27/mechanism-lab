# Corpus scaling audit — 2026-11-16

The released collection contains 16,068 papers (2017–2025), 82,255 extracted conceptual units, 273 clustered idea atoms, and 46,412 author records. The downloaded SQLite database is about 44 MB; the paper artifacts are about 363 MB; the released model checkpoints are about 342 MB. The coherence dataset has 136,827 training rows and 15,203 validation rows.

The audit is reproducible:

```sh
python scripts/corpus_audit.py --output results/corpus-audit.json
```

It writes exact counts, concentration measures, duplicate hashes, and direct train/validation overlap checks. Results are intentionally descriptive and do not treat popularity as scientific value.

The main scaling issue is representation. Adding more papers while retaining the 273-atom vocabulary mostly improves co-occurrence statistics and author coverage. It cannot represent a genuinely new mechanism that never maps to one of those atoms. Expanding the vocabulary requires rerunning conceptual-unit extraction, embeddings/clustering, atom naming, paper assignments, coherence records, author-complement records, and both scorers. The released availability training set was roughly 1.8 GB compressed and 4.3 GB extracted; the paper reports availability training on four A100s. That is a very different project from feeding more rows to the existing checkpoint.

The current coherence split has a measurable limitation: rows were split randomly after generating permutations. The JSONL does preserve `paper_id` and `perm_idx`, and the audit checks both. In the local audit, 7,124 exact serialized sequences overlap between train and validation, 9,612 validation atom sets occur in training, and 9,903 validation papers are also represented in training. This makes the released validation score a within-corpus held-out-row diagnostic rather than a clean unseen-paper or future-time generalization test. It does not invalidate the sampler or the paper's separate historical backtest, which uses a temporally frozen setup; it does mean our proposed improvements should use paper-level temporal splits for new claims.

The sensible “absurd data” experiment is therefore staged:

1. **Fixed vocabulary scaling:** add more papers only from a temporally earlier cutoff, preserve paper IDs, and measure whether held-out future atom combinations become easier to recover. This is cheap enough for a bounded pilot and tests whether more co-occurrence evidence helps.
2. **Vocabulary expansion:** recluster with a larger or multi-domain corpus, freeze the new atom catalogue, and retrain both scorers. Compare atom coverage, duplicate concentration, held-out recovery, and portfolio diversity against the 273-atom baseline.
3. **Only then scale compute:** the bottleneck is not merely model size. Extraction and clustering are upstream, availability examples grow with author repertoires, and temporal leakage can make apparent improvement meaningless.

The audit currently reports exact title/abstract duplicates, author concentration, atom coverage, and split overlap at sequence, atom-set, paper and paper-permutation levels. A next implementation should use the existing `paper_id` plus venue year to construct paper-level temporal splits and evaluate on a future time window.
