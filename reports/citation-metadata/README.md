# Citation metadata merge checks — 2026-11-16

At the time of this trial, the 115,077-paper corpus had no recorded citation metrics. Its
coverage audit preserves that absence as unknown, not zero. The subsequent [full merge](../corpus-207k/README.md) is now active.

A completed 400-record trial import into a separate snapshot produced 115,473
works: 396 new papers and citation observations for four existing papers. The
trial is not the active corpus. All 400 supplied citation counts were retained;
66 are explicitly zero. Field-weighted citation impact and normalized
percentiles are present for 346 rows and missing for the remaining 54.

Each influence observation now includes its provider, source ID, observation
date, record hash, input file, and original values. Merging does not replace
the paper's original source identity or silently change its retrieval date.
Existing aliases and auxiliary OpenAI metadata are preserved separately.
Mirror dataset/revision field names are also now recognized by the importer;
this changes future imports, not frozen prior snapshots.

`scripts/audit_corpus_metadata.py` reports known, missing, invalid, and zero
values separately. It rejects an input that changes during its scan. Its
output is a coverage report; it does not rate usefulness, feasibility, novelty,
or evidential quality. A reported high citation count is not a validation label.

```sh
python scripts/audit_corpus_metadata.py --input results/corpus-arxiv-100k/works.jsonl --output results/corpus-metadata-audit.json
```

Do not apply current citation observations to historical graph backtests as
though those counts had been available at the earlier cutoff.
