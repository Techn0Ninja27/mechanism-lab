# GNN generation benchmark: development control screen

2026-11-18. Seven fixed arms on 16 development seeds, 112 completed runs.
This screen selects a stronger comparator for the planned source-representation
experiment; it is not an engine-generated method or a fresh-seed confirmation.

The mean of the last two synchronized states was the development-selected
control: +1.64 percentage points over hold on mean exact-forward accuracy over
epochs 88–119. Own-forward gain was about +0.66 points; endpoint exact gain
about +2.56. The best-of-four selection is biased upward until final evaluation.

[Findings and limitations](../../reports/generation-comparison-v2/stage0-findings.md)
contain the analysis. [Metric receipt](../../reports/generation-comparison-v2/stage0-secondary-metrics.json)
binds these displayed values to the raw result hash. [Run summary](../../reports/generation-comparison-v2/stage0-summary.json)
records the run-time dependency hashes. Later integration edits changed some
working files; the original frozen receipts remain unchanged.

The actual runner is `scripts/run_causal_stage0.py`; its seven-arm controls and
scoring window are explicit. Local raw records remain in
`results/generation-comparison-stage0-v1/`. No novel-method, general usefulness,
or representation advantage follows from this development screen.

`metrics.json` is a byte-identical display copy of the linked metric receipt,
kept here because the study registry requires result paths under
`operator-studies/`. Its raw-result hash and numerical contents are unchanged.

The original [raw records](results/raw.json) and [run-time freeze](results/freeze.json)
are now included as exact-byte copies. Recompute all displayed secondary
metrics without training or API access:

```sh
python3 reports/generation-comparison-v2/summarize_stage0.py
```

This replays numerical analysis of the saved records. The freeze records the
run-time code hashes; later code revisions mean it is not a promise that HEAD
will rerun training byte-identically.
