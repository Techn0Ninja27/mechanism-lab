# Known-source retrieval regression — 2026-11-16

Eight hand-selected queries check four known missing-data papers already used
in our method-grounding work. This is an exposed regression set, not a measure
of general search recall or scientific novelty.

| Condition | Known source in top 5 | Corpus coverage |
|---|---:|---:|
| Existing query ordering, 16,355 records | 2/8 | 4/8 cases |
| Interleave query/provider results, same corpus | 4/8 | 4/8 cases |
| Same interleaving plus two missing primary records | 8/8 | 8/8 cases |

The first query previously consumed the result limit when it returned enough
records. Round-robin retention now preserves opportunities for later specific
queries, while retaining each query's ranking and deduplicating papers. Schema
v3 prevents cached v2 retrieval artifacts from silently changing semantics.

The corpus lacked [NeuMiss](https://arxiv.org/abs/2007.01627v4) and
[StableMiss](https://arxiv.org/abs/2305.11197v1). Their primary abstract pages
were fetched into a local packet and merged without removing any existing
canonical IDs: **16,355 → 16,357**. Response hashes and counts are published in
`corpus-change.json`; source text remains in the local corpus. The initial
system-Python fetch failed certificate verification; the project environment
succeeded without disabling TLS checks. That failed directory is preserved.

Reproduction from the existing v2 corpus, using the project Python environment:

```sh
python scripts/fetch_known_arxiv.py --id 2007.01627v4 --id 2305.11197v1 --output results/prior-art-records-reproduction
python scripts/ingest_research_corpus.py --existing-corpus results/corpus-integrated-v2/works.jsonl --packet results/prior-art-records-reproduction/packet.json --offline --output results/corpus-retrieval-reproduction
python scripts/evaluate_prior_art_retrieval.py --db results/corpus-retrieval-reproduction/corpus.sqlite --output results/retrieval-reproduction.json
```

The fetcher uses at most ten explicit arXiv IDs, three-second spacing, response
size limits, and immutable output directories. It does not search or generate
abstracts. Current grounded research uses the integrated v3 corpus.
