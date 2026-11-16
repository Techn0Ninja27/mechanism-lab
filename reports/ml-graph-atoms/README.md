# ML-first atom extraction — 2026-11-16

The expanded corpus supplied a bounded packet of **24 papers: 16 ML, 4 systems,
and 4 mathematics**. Fixed lexical queries selected two distinct papers each,
from 2023 onward. The packet covers attention, compression, memory, adaptation,
training, caching, matrix approximation and stochastic optimization. This is a
query-selected sample, not a representative sample of all ML research.

Six Haiku 5.5 calls used an 8,192-token output cap each. All six finished
normally: **no output was truncated**. Total usage was 14,856 input tokens and
13,733 output tokens, costing **$0.0083521**. The shared phase ledger is now
$0.9713601 against its $10 cap; the extraction itself had a $0.50 limit.

Initial batch-level validation accepted 53 atoms from 20 papers. One batch was
rejected because one quote was not a contiguous abstract span. A separate
recovery pass validated each saved atom independently, accepting **60 atoms
across all 24 papers** and retaining the one invalid quote as rejected. It made
no new API calls and did not edit quotes to make them pass. Span checking
normalizes whitespace and case; it does not establish semantic entailment.

An independent agent reviewed all 60 accepted atoms. The calibrated report has
**39 clean entries, 12 minor semantic flags, and 9 relevance-only tags**. These
are machine triage labels, not human approval. Mathematical and systems results
remain legitimate transfer candidates even when their source has no ML
experiment. One named-entity mismatch is ambiguous in the source itself and
is not treated as a proven factual error.

The complete source packet, raw receipts and atoms are local under
`results/ml-graph-atoms-v1/`. Public artifacts include source identifiers,
review references, extraction/recovery summaries and verified usage. No atom
was promoted to a new scientific primitive or made training-eligible. No
research proposal or empirical benefit is established by this extraction.

## Reproduce

```sh
python scripts/select_atomization_packet.py --corpus results/corpus-arxiv-100k/corpus.sqlite --output results/ml-packet-reproduction
python scripts/atomize_budgeted.py plan --input results/ml-packet-reproduction/papers.json --output results/ml-extraction-reproduction
python scripts/atomize_budgeted.py run --input results/ml-packet-reproduction/papers.json --output results/ml-extraction-reproduction
python scripts/recover_atomization_atoms.py --papers results/ml-packet-reproduction/papers.json --run results/ml-extraction-reproduction --output results/ml-recovery-reproduction
```

Packet selection was independently rerun against the saved 115,077-record snapshot and matched the original bytes. Its database hash is recorded in selection.json; rebuilding with a changed importer can change selection. The original corpus build code was published at d98c9da.
The `run` command makes paid calls; the other commands do not. Model outputs
can vary across executions. Existing run artifacts are not overwritten and
failed/unknown API calls are not automatically retried.
