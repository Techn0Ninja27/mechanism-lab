# Fixed-artifact generation: format-compliance outcome

2026-11-19. The fixed-artifact generation stage completed its predeclared
campaign: 48 generation calls producing 192 candidate slots across four fixed
representations (`task-only`, `raw`, `atoms`, and `distill`) and two decoders
(`opus-5.5` and `fable-5.2`). Structural decoding accepted 139 candidates and
rejected 53. Opus produced 51 valid and 45 invalid candidates; Fable produced
88 valid and 8 invalid candidates.

This is a generation accounting and format-compliance result only. It says
which outputs passed the frozen parser/contract; it does not say that any
decoder, representation, program, or mechanism is scientifically better.
Development evaluations are running and no final evaluation or usefulness
result is available yet.

The authoritative artifacts are the frozen inventory and accounting receipt:
`results/fixed-artifact-comparison-v1/inventory.json` and
`results/fixed-artifact-comparison-v1/generation-accounting.json`. Their bound
SHA-256 hashes are recorded in `outcome.json`. The prior preparation snapshot
and the stopped Campaign 2 preparation attempt remain preserved; neither was
reopened or overwritten.

Accounting totals from the receipts are $7.149892 upstream inference and
$23.279655 conservative reservation. The broader recorded phase reservation
is $28.4214066; these are cost/accounting fields, not comparison outcomes.

## Scope and status

- status: generated; generation complete, development evaluation pending
- validity rule: frozen structural/parser contract only
- generation calls: 48
- candidate slots: 192
- valid candidates: 139
- invalid candidates: 53
- novelty established: no
- train eligible: no

This record promotes no candidate and makes no downstream-utility claim.
The separate development evaluations use the frozen inventory; selections
and final results will be reported separately.
