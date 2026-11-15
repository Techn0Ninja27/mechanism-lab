# Temporal evaluation scaffold — 2026-11-16

`generation/temporal_data.py` creates paper-level year buckets from the released SQLite metadata and atom assignments. The default buckets are train through 2023, validation 2024, and test 2025. It canonicalizes per-paper atom sets, counts train triples, and labels later triples as strictly new when no train paper contained that exact triple.

Run:

```sh
python generation/temporal_data.py --train-only-atoms --output results/temporal-split.json
```

The SQLite uses slash-separated IDs (for example `conf/naacl/X`) while the released assignment file escapes slashes as double underscores (`conf__naacl__X`). The loader indexes both aliases and rejects collisions with conflicting years. The `--train-only-atoms` projection asks a fixed-vocabulary question: what happens when future papers use only atoms observed before 2024? Without it, future-only atoms are retained but are not scoreable by an actually pre-2024 atom catalogue. The output reports unique triples, papers containing triples, and a database join audit separately; these are different quantities.

This is deliberately not presented as a leakage-free replication of the paper's historical backtest. The released `clusters_80.json` vocabulary and atom descriptions were constructed from the full 2017–2025 corpus, so future terminology and clustering information remain visible. A genuinely frozen backtest must rebuild units, clustering, descriptions and both scorers from the pre-2024 corpus alone.

Future triples are target injections for evaluation, not candidates to be mixed into training. `uniform_candidate_triples` is a control candidate source; it draws from a supplied vocabulary and records its attempted and unique draw counts. A future implementation should score the same candidate pool with the released scorer and report recovery at fixed candidate budgets, alongside popularity/bigram baselines.

Tests cover strict-new labeling, reproducible candidate draws and invalid inputs:

```sh
python -m unittest tests/test_temporal_data.py
```
