# OpenAlex first-stage corpus expansion

The active arXiv-based corpus now contains 115,077 works. This separate plan collects at most
100,000 records with abstracts, distributed across machine learning, systems,
and mathematics. It uses the public OpenAlex Works API and reads no API key.

The frozen strata and topic IDs are in
[`OPENALEX_BULK_PLAN.json`](OPENALEX_BULK_PLAN.json). The topic labels and
subfields were checked against the OpenAlex Topics API on 2026-11-16. Topics
are machine-assigned retrieval strata, not ground-truth field labels.

The collector is resumable and bounded:

```sh
../.venv/bin/python scripts/fetch_openalex_corpus.py \
  --plan docs/corpus-expansion/OPENALEX_BULK_PLAN.json \
  --output results/openalex-100k
```

Before running the full command, inspect the exact plan without network calls:

```sh
../.venv/bin/python scripts/fetch_openalex_corpus.py \
  --plan docs/corpus-expansion/OPENALEX_BULK_PLAN.json \
  --output results/openalex-100k --dry-run
```

The command writes `records.jsonl`, `state.json`, and append-only
`receipts.jsonl`. A request cap of 700, a 200-record page size, a 15 MB response
cap, and a minimum 0.2 second interval are enforced. A request error or HTTP
429 preserves the last cursor and stops; rerunning resumes from that cursor.
Use `--save-responses` when raw compressed JSON responses are required in
addition to the receipt hashes. On restart, records and receipts are reconciled
before another request, preventing a partially written page from being counted
twice.
Deduplication uses DOI, arXiv ID, then OpenAlex ID. Every record retains the
abstract, citation count, FWCI, citation-normalized percentile, topic metadata,
locations, retrieval date, query stratum, and source hash.

Citation metrics remain separate evidence fields. They are not a usefulness or
quality label. Abstract retrieval and atomization are separate stages; this
collector makes no LLM calls and downloads no PDFs.

OpenAlex’s official API returns abstracts as an inverted index, which the
collector reconstructs deterministically. The official snapshot is much larger
(about 1.5 TB for both formats), so the API is the bounded first stage. See the
[OpenAlex work attributes](https://help.openalex.org/data/works/attributes/) and
[snapshot documentation](https://help.openalex.org/tutorials/download-the-snapshot/).
