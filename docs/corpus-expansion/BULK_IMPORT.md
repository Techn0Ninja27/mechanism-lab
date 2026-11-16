# Bulk metadata corpus import

scripts/import_bulk_corpus.py imports a downloaded JSONL metadata snapshot
without network or model calls. It accepts common arXiv and OpenAlex-flat
records. OpenAlex inverted-index abstracts are reconstructed from their
supplied word positions; no generated summary is used as an abstract.

The importer streams input and keeps deterministic hash reservoirs. Default
quotas split max-new evenly across ML (cs.LG, cs.AI, cs.CL, cs.CV, cs.NE,
stat.ML), systems (cs.DC, cs.OS, cs.DB, cs.PF, cs.NI, cs.AR), and math/theory
(math.*, stat.TH, cs.IT, cs.DS). Overlapping records are assigned to the first
matching group. A fixed seed makes selection independent of input order.

Example:

    ../.venv/bin/python scripts/import_bulk_corpus.py \
      --metadata-jsonl /path/to/metadata.jsonl \
      --existing results/corpus-integrated-v3/works.jsonl \
      --output results/corpus-bulk-100k \
      --max-new 100000 --cutoff 2026-11-16 --seed 73031

Output is a new snapshot; a nonempty output directory is rejected. The
existing corpus is never changed. bulk-manifest.json records input SHA-256,
line/skip/malformed counts, quotas, category counts, existing-row count, and
output hash. works.jsonl retains categories and influence fields such as
citation_count and field_weighted_citation_impact; SQLite adds categories_json.

Citation and influence fields are descriptors for stratification and evidence
retrieval, not paper-quality labels. Compare unweighted, field-normalized,
recency-aware, and evidence/code availability weights as separate ablations.
