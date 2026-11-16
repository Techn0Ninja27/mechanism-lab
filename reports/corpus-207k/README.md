# Citation-enriched corpus — 2026-11-16

The verified merged snapshot contains **206,917 searchable records**, up from
115,077. It preserves all previous canonical IDs, all 116,335 previous aliases,
and the 15 separate OpenAI auxiliary records. SQLite integrity and search-index
counts passed. Known-source retrieval remains 8/8 at top 5 on the exposed
regression cases; this is not a general recall benchmark.

The OpenAlex harvest returned **93,604 records**, not its 100,000 target:
53,604 in the ML query group, 20,000 systems, and 20,000 mathematics. Some ML
strata exhausted their available pages before filling their allocations. The
collector's original `targets_reached` label was therefore too broad; counts
and the explicit 6,396-record shortfall are authoritative. Future runs now report per-category deficits and use `strata_exhausted` instead of claiming a filled target. The run used 552
requests, including a preserved TLS failure, and resumed without restarting.

After conservative identity matching, the merge adds **91,840 records**. All
93,604 provider observations survive, including observations attached to
existing or duplicate provider records. Citation counts are available for
92,876 records; normalized influence measures for 60,237. Missing metrics stay
unknown. As-of dates and provider hashes accompany the observations. No
citation-weighted usefulness model is enabled.

Selection groups in the merged corpus are 112,318 ML, 39,742 systems, 39,791
math, and 15,066 unassigned legacy records. These are acquisition/category
labels, not independently adjudicated field identities. OpenAlex topic strata
were sampled using explicit descending publication dates within date windows;
the collection is not a random or comprehensive sample of ML research.

## Merge safeguards and corrections

The active v3 merge uses DOI/arXiv identifiers or exact title-and-abstract
content. A shared title alone does not establish identity. Both the existing
corpus and provider observations remain traceable. Earlier local merge
attempts are retained: v1 dropped seven citation observations at an exact-text
deduplication branch; v2 still permitted ambiguous title-only matching. Neither
was selected as the active corpus. The v1 audit is under `diagnostics/v1/`.

Use `results/corpus-openalex-enriched-v3/` locally. The engine and webapp
select only completed snapshots. These remain metadata and abstracts, not
206,917 atomized papers or a newly trained scorer.

See [the page-based collection plan](../../docs/corpus-expansion/OPENALEX_PAGE_PLAN.json),
[importer documentation](../../docs/corpus-expansion/BULK_IMPORT.md), and the
adjacent manifests, audit, and verification results. Large raw response files
remain local. All API collection used public metadata access; no model calls
were needed to build this corpus.

The saved archive audit verified 551 successful response bodies and all 93,604 normalized records against those bodies, including their field contents. The one failed request remains in the receipts. This verifies internal provenance, not the truth of paper claims.
