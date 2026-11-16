# Remove the ML sampling cap from the local evidence corpus

2026-11-17. The active corpus contains **739,964 works**, up from 206,920.
This adds 533,044 unique works while preserving prior content and identities.
It is searchable metadata/abstract coverage, **not** new atomization, scorer
training, or evidence of improved research novelty.

## Why the previous search missed known work

FedAIS (2409.14655), SAT (2308.13466), and frequency-ordered tokenization
(2602.22958) were absent by identifier and title in the old corpus. All three
exist in the pinned local mirror and satisfy its ML category/date rules, but
the earlier seeded-hash selection kept only 60,000 of 605,681 eligible ML rows.
Their hashes fall outside that cap. They were not restored by the other sources
in the old merged corpus. This was a coverage failure, not just poor ranking.

The native audit initially inspected a stale partial-download path and wrongly
reported the full archive absent. That report is preserved locally. The corrected
`gap-audit.json` checks the actual complete archive, its hash, the three IDs and
the selector cutoff. Root independently completed the full scan before accepting
that correction.

## What changed

`select_arxiv_metadata.py --all-ml` streams all eligible ML records from the
existing pinned archive rather than retaining only the lowest 60k hashes. The
same 20k systems and 20k math selections remain under their previous rules and
seed. Cross-listed learning papers retain ML priority. Default sampled mode is
preserved for reproducibility.

The archive has 3,148,796 rows; selection emitted 605,681 ML, 20,000 systems and
20,000 math records. It is the previously downloaded third-party arXiv metadata
mirror, revision `cee894837962fede5612cccf2a4c7cacf49b4c3a`, documented as harvested
2026-08-30. **All eligible ML in this snapshot is not all ML research through
2026-11-16.** The date cutoff excludes later entries/revisions but cannot create
missing recent papers. This expansion required no new bulk download.

Before merging that selection, three v1-pinned primary abstracts were fetched
from arXiv and added to an intermediate snapshot, so their audited primary text
is retained when the mirror aliases merge. Those three HTML downloads totaled
126,986 bytes. SAT has a later v2; the explicit v1 pin is first-version evidence,
not a claim to have ingested the latest methods. The final merge keeps existing
canonical IDs and enriches missing DOI/category metadata without replacing old
text or influence observations.

Final groups: 650,677 ML, 39,791 math, 39,747 systems, and 9,749 unassigned records.
Citation counts remain known for 92,876 records; normalized influence for 60,237.
Missing influence is not zero or evidence of low usefulness. There are 135
existing empty abstracts; the expansion does not turn them into full evidence.
The 15 auxiliary OpenAI metadata entries remain separate and byte-identical.

## Preservation and retrieval evidence

`preservation-full.json` checks all 206,920 old rows and 209,946 old aliases.
No old source was lost; canonical IDs, titles, abstracts, authors, dates,
content hashes and evidence provenance are unchanged. Existing nonmissing
citation fields and influence provenance are unchanged. The 739,964-row full-text
index has no orphan/duplicate source IDs or changed indexed text; SQLite's quick
check passed. Every one of the 645,681 selected source IDs resolves in the merged
index as a canonical identity or alias, and no alias target is missing.
These checks do not independently validate bibliographic deduplication or paper
claims.

Six exposed queries cover names and mechanisms for the three known misses:

| Snapshot | Hits in top 10 |
|---|---:|
| Previous 206,920-record corpus | 0/6 |
| Three-paper primary repair only | 6/6 |
| Expanded ML mirror corpus | 6/6 |

The targeted repair alone fixes these known cases. The table must not be used
as evidence that half a million extra papers improve general ranking. Previous
missingness regressions retain 8/8 hits at 5. The four KV sources retain 4/4 hits
at 10 with their exposed mechanism queries; their earlier generic queries still
retrieve only 1/4. Query design remains a bottleneck. Precision, representative
recall, novelty and proposal usefulness were not evaluated here. Retriever code
was unchanged.

The registry now selects the completed v6 snapshot. The app and future grounded
runs use it; previous paid plans retain their original corpus hashes and must
not be silently resumed against the replacement.

## Reproduce

From the engine root, with the pinned archive already present and fresh outputs:

```sh
../.venv/bin/python scripts/build_arxiv_primary_packet.py \
  --id 2409.14655v1 --id 2308.13466v1 --id 2602.22958v1 \
  --output results/systems-primary-fresh
../.venv/bin/python scripts/import_bulk_corpus.py \
  --metadata-jsonl results/systems-primary-fresh/records.jsonl \
  --existing results/corpus-ml-grounded-v4/works.jsonl \
  --output results/corpus-primary-fresh --max-new 3 --group ml=cs. \
  --cutoff 2026-11-16
../.venv/bin/python scripts/select_arxiv_metadata.py \
  --input results/corpus-sources/secemp9/metadata/train-00000-of-00001.parquet \
  --output results/arxiv-all-ml-fresh --all-ml
../.venv/bin/python scripts/import_bulk_corpus.py \
  --metadata-jsonl results/arxiv-all-ml-fresh/records.jsonl \
  --existing results/corpus-primary-fresh/works.jsonl \
  --output results/corpus-all-ml-fresh --max-new 645681 --cutoff 2026-11-16
../.venv/bin/python scripts/audit_corpus_preservation.py \
  --before results/corpus-ml-grounded-v4/corpus.sqlite \
  --after results/corpus-all-ml-fresh/corpus.sqlite \
  --output results/corpus-all-ml-fresh/preservation.json
```

The primary extractor can rebuild offline from archived HTML with `--offline`.
The selection validates the pinned archive SHA before scanning; no network is
used in selection/import/audit. Fresh online HTML and retrieval dates may change
packet hashes. Exact observed receipts, selection/code hashes, preservation,
metadata and retrieval reports are retained here; large corpus/source files stay
in local ignored storage. `evaluate_systems_retrieval.py` reproduces the three-
snapshot diagnostic; existing KV/missingness evaluators cover their regressions.
No paid model calls were used.
