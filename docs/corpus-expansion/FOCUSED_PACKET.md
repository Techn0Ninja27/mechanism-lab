# Focused packet import

The atomization study's `results/atomization-study-v1/expanded-papers.json` is
a bounded, offline primary-abstract packet. Rebuild an integrated corpus with:

```sh
python3 scripts/ingest_research_corpus.py --offline \
  --existing-corpus results/corpus-expanded/works.jsonl \
  --packet results/atomization-study-v1/expanded-papers.json \
  --output results/corpus-integrated-v2
```

`--packet` is repeatable. `--offline` disables the arXiv API entirely; the
output extends the read-only expanded snapshot with packet rows. The verified
rebuild contains 16,355 deduplicated works and preserves all 16,346 prior
canonical IDs (the packet contributes 9 net new works).
Packet IDs retain their exact version as `source_id` while `canonical_id`
removes the arXiv version suffix. Both published and updated dates are checked
against `--cutoff`.

Packet file SHA-256 values and snapshot metadata live in `packet_sources` and
in the `packet_sources` array of `manifest.json`; they are deliberately
separate from generated OpenAI summaries in `openai_sources`. The packet's
declared `abstract` is the only field copied into the abstract index.
