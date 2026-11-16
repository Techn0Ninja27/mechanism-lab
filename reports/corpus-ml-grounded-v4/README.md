# ML corpus repair after a KV-cache prior-art miss

2026-11-16. The active corpus now contains **206,920 works**, adding three
primary abstracts to the previous 206,917-record ML-first corpus. Systems and
math coverage is preserved. This repairs a known gap; it is not broad expansion
or a claim that the corpus is complete.

The four-paper packet pins arXiv v1 metadata for MixKVQ (2512.19206), ReadKV
(2610.11245), Attention-Aware Transform Coding (2608.14191), and MiKV
(2402.18096). MiKV already existed and resolves to its original canonical ID.
The other three are new. The extractor copies primary citation metadata from
saved arXiv HTML; it does not turn generated descriptions into abstracts.
Response URLs, byte hashes and extractor hash are in `primary-packet-manifest.json`.

Preservation checks confirm all 206,917 previous canonical IDs and 209,939
aliases survive, the 15 auxiliary OpenAI metadata records remain byte-identical,
and the SQLite full-text index has 206,920 rows. Citation-count coverage remains
92,876 records; normalized influence coverage remains 60,237. Those metrics
describe availability, not scientific usefulness. See `preservation.json` and
`metadata-audit.json`.

## Retrieval diagnostic

Four known sources were checked within the first ten results:

| Corpus | Original queries | Mechanism-focused queries |
|---|---:|---:|
| Previous snapshot | 0/4 | 1/4 |
| Expanded snapshot | 1/4 | 4/4 |

Original queries were “KV cache quantization”, “mixed precision cache”, and
“adaptive KV retrieval”. Development queries were “query cache precision”,
“attention cache distortion”, and “importance cache precision”. The latter were
adapted after seeing results: this is an exposed diagnostic, not held-out recall
or a general retrieval improvement. Irrelevant returned documents remain; we
have not measured precision. The retrieval algorithm was unchanged. All eight
earlier missing-data known-source checks still pass.

The result matters because both source coverage and query formulation caused
the failure. Increasing corpus size alone did not make the closest prior works
appear in a bounded proposal context. Full-method comparison is documented in
[the KV prior-art audit](../../docs/method-grounding/kv-window-ranking-prior-art.md).

## Reproduction

From the engine root, use fresh output directories:

```sh
../.venv/bin/python scripts/build_arxiv_primary_packet.py \
  --id 2512.19206v1 --id 2610.11245v1 --id 2608.14191v1 --id 2402.18096v1 \
  --output results/kv-primary-packet-fresh
../.venv/bin/python scripts/import_bulk_corpus.py \
  --metadata-jsonl results/kv-primary-packet-fresh/records.jsonl \
  --existing results/corpus-openalex-enriched-v3/works.jsonl \
  --output results/corpus-ml-grounded-fresh --max-new 4 --group ml=cs. \
  --cutoff 2026-11-16
../.venv/bin/python scripts/evaluate_kv_retrieval.py \
  --before results/corpus-openalex-enriched-v3/corpus.sqlite \
  --after results/corpus-ml-grounded-fresh/corpus.sqlite \
  --output /tmp/kv-retrieval-fresh.json
```

An offline packet rebuild can use copied archived HTML and `--offline`; current
retrieval dates and later HTML presentation changes can change new byte hashes.
The recorded manifests identify the exact observed inputs. Large corpus files
and HTML response bodies are local artifacts, not committed to Git.
No paid model calls were used for this repair.
