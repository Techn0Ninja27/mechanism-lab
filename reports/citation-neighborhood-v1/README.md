# Citation expansion around distributed GNN training

2026-11-17. Question: does following actual references, citing works and provider
similarity links surface plausible competitors missing from one local search?

## Method and result

Seed: SAT, arXiv:2308.13466, manually title-matched to OpenAlex W4409983225
(2025 proceedings DOI; the preprint is from 2023). Three bounded OpenAlex calls
retrieved 29 references, 10 related IDs and 9 citing works: 47 distinct records,
48 typed edges, no unresolved selected IDs. Responses totaled 249,351 bytes.
An earlier seed-resolution search used a separate request. No model calls,
framework installation or large downloads were required. OpenAlex reported
$0.001 for the search and $0.0001 for each batch/list call under its shared free
allowance; the single-work response did not provide a cost field.

The comparison used the existing corpus, three queries (`embedding prediction
staleness`, `historical embedding graph`, `staleness graph training`) and its
20-record interleaved limit. This is an exploratory, unequal-budget inspection,
not an improvement percentage, recall benchmark or evaluation of Connected
Papers. Relevant leads were selected after seeing results. The graph also
contains weakly relevant items, including proceedings and trajectory papers.

## What changes our next proposal

- **FreshGNN** appeared in both views. Citation expansion did not discover it
  uniquely; its presence alone would not justify adding the new retriever.
- **Sancus-Q** was absent from those 20 local results. The primary paper already
  combines bounded embedding staleness, skipped broadcasts and quantization.
  A proposal that merely joins staleness and quantization needs a more precise
  contribution. [Primary paper](https://link.springer.com/article/10.1007/s00778-024-00897-2),
  [author implementation](https://github.com/qyliu-hkust/sanqus).
- **SNI-GNN** was also absent from that list. The primary paper combines a
  lightweight embedding predictor, boundary-node importance sampling and DPU
  execution. Combining prediction with importance sampling is therefore already
  represented in close prior work. Its distributed hardware results have not
  been reproduced here. [Primary paper](https://arxiv.org/html/2608.06441v1).
- **Sift** is a channel-wise reuse lead absent from the local list. Its OpenAlex
  abstract reports a channel-level accuracy model and partial historical
  embedding reuse. The publisher page returned HTTP 403 during verification;
  the method details remain unverified here. [Publisher DOI](https://doi.org/10.1145/3809161).

These findings establish that this particular search omitted useful leads, not
that the papers are absent from our corpus or that citation expansion always
performs better. The next proposal must name its difference from these methods,
and account for the cost of observing whatever error signal it uses. A local
simulation can test algorithmic benefit but cannot establish multi-GPU or DPU
speedups. No new research proposal or experimental gain is claimed in this run.

## Implementation and reproduction

`generation/citation_neighborhood.py` uses only the Python standard library.
It requires an explicit seed ID, bounds each edge type, distinguishes citations
from provider-related links, records missing metadata, verifies response hashes,
and defaults to offline cache reads. Network errors raise rather than becoming
empty evidence. This is a standalone retrieval tool, not yet an automatic step
in the paid proposal pipeline or webapp. It does not reproduce Connected Papers'
co-citation/bibliographic-coupling ranking. OpenAlex documents its fields and API
recipes [here](https://help.openalex.org/how-to/api-recipes/).

From repository root (Python 3.12.14 used):

```sh
python -m generation.citation_neighborhood --seed W4409983225 --cache results/citation-neighborhood-v1/cache --output results/citation-neighborhood-v1/new-replay.json
python -m unittest tests.test_citation_neighborhood tests.test_prior_art_retrieval
```

The existing local cache reproduced the original output byte for byte. To fetch
on another machine, add `--allow-network`; live provider state can change, so
that is a new snapshot, not a historical reproduction. Raw responses and full
abstracts remain in local results; `inspection.json` publishes metadata, edges,
receipts, source hashes and the local comparison titles. Four new tests cover
edge direction/type, unresolved metadata, offline replay/tamper detection,
network opt-in, identity validation and malformed incoming edges/abstracts.
All 14 targeted tests passed.

Current graph membership and dates must not be used as a time-travel backtest:
the seed's proceedings record can have different coverage from its preprint,
provider matches can be incomplete, and related-work membership is computed
today. No novelty or training labels are produced.
