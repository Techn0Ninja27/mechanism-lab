# Bounded interface graph

Generated 2026-11-16 by `scripts/build_interface_graph.py` from the recovered
60-atom packet, its 24-paper manifest, and the machine review. The JSON contains
at most 20 directed output-text → input-text proposals. Producers and consumers
must come from different source papers and different corpus groups; producers may be method or finding atoms from ML, systems, or math.
One endpoint must be ML. Consumers are method atoms.

The two scores are corpus-wide TF–IDF cosine heuristics: one over producer
outputs and consumer inputs, and one over their statements. They are displayed
as cleartext proposal scores only; they are not typed proofs, novelty claims,
compatibility claims, or molecule promotions. Each item carries source hashes,
stable atom IDs, and an explicit unresolved-compatibility warning.

Semantic review flags are excluded using exact `(source_id, source_hash, computed statement_sha256)`
matching. `relevance_only` flags remain eligible. Missing or ambiguous records
are rejected. The seeded-random shortlist in the JSON is a deterministic
comparison baseline and is not a second ranking claim.

Rerun:

```sh
python3 scripts/build_interface_graph.py --out /tmp/interface-graph-fresh.json
```

The 60-atom packet has 39 unflagged entries, 11 semantic flags, and 10
relevance-only tags. The corrected graph retains 49 atoms, enumerates 424
directed pairs, and ranks 20. The earlier review aggregate was a counting
error; individual judgments did not change. No missing twelfth semantic review
exists. The diagnostic-v1 JSON is superseded and must not be used for results.
