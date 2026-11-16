# Bounded prior-art retrieval

`generation/prior_art_retrieval.py` is the retrieval boundary for Mechanism
Lab. It accepts one to three explicit query strings, searches the local
`results/corpus-expanded/corpus.sqlite` FTS5 index read-only, and returns
abstract-level records with source IDs, URLs, dates, evidence level, a source
content hash, and the queries that matched each record. It does not infer or
label novelty.

Local retrieval is always attempted first. FTS punctuation is tokenized and
quoted, so user text cannot inject FTS operators. A cutoff excludes records
published or updated after that date. Results are bounded by `limit` and the
artifact records truncation explicitly.

ArXiv is disabled unless `allow_arxiv=True` (or `--allow-arxiv`), uses at most
five results per query, imposes a response byte limit and timeout, and waits at
least three seconds between uncached requests. It only requests the fixed
arXiv API endpoint; arbitrary URLs and PDFs are never downloaded. Network and
parse failures are retained in `failures`. Cached responses are keyed by query,
limit, provider, schema, and cutoff; parameter changes fail rather than silently
reusing a cache.

Each run includes a deterministic `query_hash`. Pass `artifact_path` to write a
new JSON retrieval artifact. Artifacts are immutable: an existing path raises
`FileExistsError`. A retrieval artifact is evidence for review, not a novelty
decision and never invokes a paid model.

Example (offline):

```sh
python -m generation.prior_art_retrieval \
  --db results/corpus-expanded/corpus.sqlite \
  --query 'continual learning memory' \
  --artifact results/prior-art/run-2026-11-16.json
```

Tests use a temporary SQLite corpus and mocked arXiv responses:

```sh
python -m unittest tests.test_prior_art_retrieval
```

The grounded workflow prefers `results/corpus-integrated-v2/corpus.sqlite` when available. See [focused packet import](corpus-expansion/FOCUSED_PACKET.md). arXiv natural-language queries are compiled to explicit literal-token AND clauses; the cache schema records the compiled query. Results are interleaved across query/provider groups so a large local bucket cannot hide all external hits. Provider coverage includes retained counts. The API contract follows the [official arXiv manual](https://info.arxiv.org/help/api/user-manual.html).
