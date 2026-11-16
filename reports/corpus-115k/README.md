# ML-first corpus expansion — 2026-11-16

Historical snapshot. The later [citation-enriched corpus](../corpus-207k/README.md) is now active.

The active searchable corpus grew from **16,357 to 115,077 works**. This adds
**98,720 net records** after merging a 100,000-paper selection with the existing
snapshot. All 16,357 previous canonical IDs remain present. SQLite's integrity
check passed and the full-text search index contains exactly 115,077 rows.

The new selection contains **60,000 ML**, **20,000 systems**, and **20,000
mathematics** papers. These are ordered category groups, not inferred usefulness
labels: cross-listed ML papers go to ML first. A seeded hash sample was selected
from a complete scan of the pinned 3,148,796-record arXiv metadata snapshot.
Selection counts, eligible population sizes, and publication-year coverage are
in `selection-manifest.json`.

The [source snapshot](https://huggingface.co/datasets/secemp9/arxiv-complete)
documents an arXiv OAI-PMH metadata harvest on 2026-08-30. Only its 1.64 GB
metadata file was downloaded. Its exact published SHA-256 matched locally.
It is a third-party mirror and does not establish complete coverage through
November. Existing newer records were preserved. Full paper PDFs were not
acquired, and these added papers have **not** all been atomized or used to
retrain the Alien scorers.

The original 15 OpenAI-related metadata records remain a separate auxiliary
file; they were not promoted into primary abstracts. Author strings, category
metadata, source revision and paper-license fields are retained for audit.

## Retrieval checks

The eight known-source regression cases remain **8/8 at top 5**. Three broader
queries—distributed training, stochastic optimization, and spectral graph—each
returned bounded local results. These checks verify working retrieval, not
exhaustive prior-art recall or improved proposal novelty.

The engine and local webapp select this completed snapshot automatically. An
incomplete harvest cannot replace the active corpus. The webapp overview shows
the searchable-paper count.

## Reproduce

See [the pinned source and selection commands](../../docs/corpus-expansion/ARXIV_BULK.md)
and [the importer](../../docs/corpus-expansion/BULK_IMPORT.md). The optional
bulk-reader dependencies are pinned in
[requirements-bulk.txt](../../docs/corpus-expansion/requirements-bulk.txt).
The selection, import, and verification manifests are published here; large
metadata files and the search database stay local.

## Citation enrichment is separate

OpenAlex collection is also running under its bounded page-based plan. Its
cursor API rejected unauthenticated pagination; ordinary page requests were
verified to work within public limits. That collection retains citation
counts and normalized influence metrics, but it is not yet merged into this
snapshot. No citation-weighted usefulness model is deployed, and no paid LLM
calls were used for this expansion.
