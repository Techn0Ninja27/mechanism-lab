# Fresh generation with contracts and experiment feedback

2026-11-17. The updated Opus proposal-and-review run completed with valid JSON,
exact quotations and no truncation. It declined a new research proposal and
considered a 1→2-bit retest. Its factual reasoning was not reliable enough to
accept without checking the saved experiment.

Both stages received two source-contract cards, the scoped negative pilot and
ten retrieved prior-art records. Retrieval included MiKV, MixKVQ, ReadKV and
Attention-Aware Transform Coding. This is one development run with a revised
question, not a controlled demonstration that contracts improve generation.

## Factual audit

The historical pilot used prefill queries for ranking and decode queries only
for evaluation. The reviewer incorrectly attributed future-query leakage to
that pilot. The newly proposed data path does mention a decode query and has a
real timing ambiguity, but it does not describe the saved experiment.

The 2-bit mean-logit arm was already tested. The reviewer also raised a logging
concern because the feedback rounded two exact-mass measurements to 0.01510;
the original values differ. Finally, exact attention mass is a comparator, not
a proved optimal precision allocator. The reviewer correctly challenged that
unsupported claim in the proposal.

`factual-audit.json` binds these corrections to the exact unmodified model
result. The webapp displays it separately as a machine-assisted post-run audit,
not as scientific validation. The feedback registry now explicitly distinguishes
selection inputs from evaluation inputs, lists already-tested controls, retains
more numeric precision and states that exact mass is not a fidelity ceiling.
This fixes missing context; it does not prove future reviewers will reason well.

## Suggested follow-up tested

The review suggested max-per-window aggregation, which was new relative to the
completed pilot. The [exploratory follow-up](../../operator-studies/trace-aggregation-ablation/README.md)
slightly lowers one-bit relative L2 on both exposed images, but still loses to
exact attention mass and slightly worsens cosine on one image. It does not
establish a useful new mechanism. The old 2-bit arm was not presented as a new
experiment.

## Cost and provenance

Two Opus calls consumed 97,389 input and 4,663 output tokens; both ended with
`finish_reason=stop`. Reported usage cost was $0.47798784. The shared ledger
conservatively accounted for $0.482816 of upstream inference cost, bringing the
phase total to **$2.1678281 / $10**. See `receipts.json`.

The original output, frozen plan, contracts and feedback snapshot remain in the
record. Source documents and raw API prompts remain local. No source/model bulk
downloads were made for this run. No novelty or training-eligibility claim is
promoted from it.
