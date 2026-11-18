# Campaign 2: stopped during preparation review

2026-11-19. One paid Opus preparation draw produced twelve schema-valid atoms.
Twenty-four fresh native contexts reviewed them, two per record. Full launch
prompts and input hashes were saved before spawning; returned task-name receipts
were recorded afterward. The frozen program and raw outputs remain unchanged.

The gate could not complete: slot b, record 10 marked an unknown field supported
and placed its absence explanation in `reason`, but omitted the separate required
`absence_reason`. The frozen decoder rejected it. The protocol disallows repair,
replacement reviews, or advancing to a new draw on malformed review output, so
the campaign was explicitly stopped before any prose preparation or candidate
generation. No generated method was executed.

This was not the only issue. Of the 23 mechanically valid reviews, seven flagged
non-supported fields across five atoms. These are reviewer flags, not independently
adjudicated source errors. Fixing the missing metadata alone would not have made
this draw pass the all-supported gate. The result measures an extraction/review
contract failure, not whether atom representations improve research or algorithms.

The completed request used 10,661 input and 3,194 output tokens. The ledger charges
$0.106524, the larger reported upstream cost; phase spend is now $5.1417516 of $30.
The response reported `is_byok: false`. No provider-credit discount was assumed.
There were zero candidate-generation calls and zero new Cora executions.

The initial direct-script invocation failed to import the package before any
lifecycle action. Execution then used `python -m scripts.run_campaign2_preparation`;
this was not an API retry. The complete response, original raw reviews, failed
gate traceback diagnosis, and stop marker are preserved locally. `outcome.json`
contains hashes and a redacted review-validity summary.

Any future comparison must be separately declared. This run will not be reopened
or silently reclassified by a more permissive parser. A useful question for a
future design is whether end-to-end generation helps despite preparation defects;
that requires measuring those defects rather than pretending source fidelity.

Clarification: valid reviews had already disqualified this draw on fidelity.
Seven valid reviews flagged six unique fields across five records; both slots
flagged records 1 and 8, and one slot flagged each of records 2, 3 and 5. These
flags are not yet independent adjudications of source truth. With well-formed
reviews, that fidelity failure would have permitted the next predeclared draw.
The malformed additional review was the reason advancement became unavailable
and the campaign stopped. It was not the sole defect in the preparation.
