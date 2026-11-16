# ML graph atoms v1: machine review

Reviewed 53 initial atoms plus 7 valid batch-2 recoveries (60 total). The invalid batch-2 atom at `(batch=2, atom_index=1, source_id=arxiv:2510.12096)` remains excluded. This is machine semantic-faithfulness/relevance triage only: no human adjudication, novelty judgment, or claim that a flag proves the source is wrong.

Counts: 39 clean; 12 semantic flags (0 major, 12 minor); 9 relevance-only tags. Relevance-only means a math, systems, or physics result may be a transferable input but is not itself an ML experiment; it is not a semantic failure. Every JSON entry has a nonempty `source_hash`, exact `statement_sha256`, source ID, and extraction location.

The strongest semantic concern is atom 36: the supplied abstract has an entity-name mismatch around DistSim/DistDim. One sentence says DistDim breaks down the model into events, while surrounding text attributes the overall modeling to DistSim. The atom's normalization to DistSim is therefore unresolved/needs source verification, not a proven major factual error.

Other semantic flags concern quote-vs-abstract scope (atoms 2, 7, 10, 14, 20), omitted theorem/study conditions (17, 23, 46, recovered batch-2 atom 6), and ambiguous latency wording (43). Recovered batch-2 atom 4 adds VLM scope beyond its short quote and is tagged relevance/provenance-only rather than semantic failure.

The superseded review is preserved locally in `results/ml-graph-atoms-v1/review-history/`; this revised JSON is the authoritative calibrated report. Source and statement hashes were checked, and extraction locations were reconciled against the saved responses.
