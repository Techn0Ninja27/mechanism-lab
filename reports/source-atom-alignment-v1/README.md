# Existing GNN inputs are not a matched source/atom comparison

2026-11-18. Before comparing raw-source prompting against atomized prompting,
we checked the suggested existing artifacts. **Zero of 53 atom quotations occurs
exactly in any of the GNN packet's nine chunks.** These files therefore cannot
be used unchanged to attribute a difference to representation of the same
source information. New extraction from a frozen common packet is needed.

This is an input-preparation finding, not evidence against atomization. The
check searches all chunks, irrespective of source ID; even a positive text match
would still require identity and interpretation checks. It does not inspect
scientific entailment or assess novelty. No model calls or downloads occurred.

Run from repository root with the local source artifacts present:

```sh
../.venv/bin/python reports/source-atom-alignment-v1/check.py \
  --pack results/gnn-useful-proposal-v1/pack.json \
  --atoms results/ml-graph-atoms-v1/extraction/atoms.jsonl \
  --output /tmp/atom-source-overlap.json
```

The output refuses overwrites and records both input hashes. The public result
contains IDs and counts, not source excerpts. Historical input files remain local.
