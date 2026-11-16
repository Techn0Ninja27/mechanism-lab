# ML-first arXiv metadata selection

The corpus expansion keeps ML as its main subject and includes systems and
mathematics. It does not atomize every acquired abstract or download full texts.

The bulk source acquired is the metadata component of
[secemp9/arxiv-complete](https://huggingface.co/datasets/secemp9/arxiv-complete),
a third-party snapshot that documents harvesting arXiv's OAI-PMH arXivRaw
metadata on 2026-08-30. Metadata is documented separately from the varied
licenses on paper full texts. Preserve attribution to arXiv and the snapshot.
This is a pinned mirror, not an independently verified copy of every paper.

- Revision: `cee894837962fede5612cccf2a4c7cacf49b4c3a`
- File: `metadata/train-00000-of-00001.parquet`
- Bytes: `1643927530`
- SHA-256: `7f7bea788af318a3f0fed1c446303c725d339c2cb189ede4412aeadd0ab9bf1f`
- Reported metadata rows: 3,148,796

Only this one metadata file is requested. Do not run an unfiltered repository
snapshot download: its PDF/source archive is several orders of magnitude larger.

`select_arxiv_metadata.py` verifies the exact file digest, scans bounded record
batches, and selects by deterministic seeded ID hashes within arXiv category
groups. Cross-listed papers go to ML first, then systems, then mathematics.
Targets are 60,000 / 20,000 / 20,000. If a group has too few eligible records,
its actual count is reported; targets are not reported as achieved counts.
The scan records eligible counts and selected publication years. It excludes
records beyond the cutoff and does not favor citation counts.

Install the optional reader dependencies, then acquire only the pinned file:

```sh
python -m pip install -r docs/corpus-expansion/requirements-bulk.txt
hf download secemp9/arxiv-complete metadata/train-00000-of-00001.parquet --repo-type dataset --revision cee894837962fede5612cccf2a4c7cacf49b4c3a --local-dir results/corpus-sources/secemp9
python scripts/select_arxiv_metadata.py --input results/corpus-sources/secemp9/metadata/train-00000-of-00001.parquet --output results/arxiv-selected-100k
python scripts/import_bulk_corpus.py --metadata-jsonl results/arxiv-selected-100k/records.jsonl --existing results/corpus-integrated-v3/works.jsonl --output results/corpus-arxiv-100k --max-new 100000
```

The selector needs PyArrow; the importer itself uses the Python standard
library. The completed import contains 115,077 works; see [verified results](../../reports/corpus-115k/README.md).
The August snapshot is not a complete account of research through November;
existing newer sources remain in the corpus and need continued targeted updates.
