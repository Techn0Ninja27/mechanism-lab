# Serving proposal and diagnostic

2026-11-17. One Opus proposal/review run completed, both responses ended with
`stop`, and all required IDs/quotes passed local validation. It cost **$0.463808**
using the conservative receipt amount; shared spending is now **$3.7441445/$20**.
This is a generated research candidate with unresolved issues, not a discovery.

The candidate changes only decode-admission order: largest retained prompt
cache first, aiming to release prefill memory. The model marked its proposal
**revise**; its separate reviewer recommended **test**, with stronger controls.
The central contribution received partial-overlap coverage, not novelty.

The native/root audit found unspecified lower-bound/restart semantics, a
5% hypothesis versus3% kill-rule mismatch, and missing smallest-first and
backpressure controls. A [corrected toy preflight](../../operator-studies/serving-admission-preflight/README.md)
checks accounting and reproduces a counterexample: largest-first flow time7
versus6 for smallest-reservation-first. That rejects uniform dominance only;
there is no stochastic efficacy, hardware, or publication-level novelty claim.
The first agent simulator had a next-token accounting bug; its output was
preserved locally and superseded before drawing an invariant conclusion.

## Evidence and limitations

The private pack contains70,673 characters in20chunks, without pack truncation,
from [DMRS](https://arxiv.org/html/2610.15579v1),
[Arrow](https://arxiv.org/html/2505.11916v2),
[Feng's GSA](https://arxiv.org/html/2601.22996v1),
[Feng's regime-aware routing](https://arxiv.org/html/2607.09248v1), and
[Chen's interval refinement, v2](https://arxiv.org/html/2508.14544v2).
These are selected excerpts, not complete proofs or exhaustive prior art.
The adjacent excerpts retain algorithmic steps, assumptions and omitted-section
notices. Full snapshots remain local. Pack digest:
`d54bdb6453e9a61e0d0e36aae305263c19a6e83a88ec655648be7103c8c468c7`.

The paid run's additional corpus retrieval returned **zero records**: overly
restrictive AND queries included author surnames. Its conclusions therefore
rest on the five supplied sources only. A subsequent [query diagnostic](retrieval-diagnostic.json)
found additional leads with shorter queries. The new [offline preview](retrieval-preview.json)
returns10 retained leads; those were **not** retroactively supplied to the model.
They must be checked before any efficacy or novelty assessment.

Protocol8 supplied an exact allowed-ID list to both stages. Protocol9 now runs
local retrieval during offline planning (never network retrieval), saves
`planning-diagnostics.json`, and refuses a paid execution when a requested
search has no hits. Source-only runs remain explicit when no queries are
provided. Retrieval hits are leads, not evidence of novelty or entailment.
The paid result remains frozen under protocol8; no silent repair or retry ran.

## Artifacts and replay

[Proposal/review export](proposal-result.json) replaces verbatim evidence
quotes with hashes. Exact-span validation applies to the original local result;
the redacted export is not a validation replay fixture. [Call receipts](call-receipts.json),
[plan](plan.json), [post-run audit](factual-audit.json), [question](question.txt),
and [source-selection receipt](adjacent-selection-receipt.json) preserve the
run's scope and provenance. The local original is
`results/serving-problem-v2/decode/result.json`; the workbench displays its
unchanged result and attaches the diagnostic separately.

To rebuild the identical pack from the existing local source tree:

```sh
../.venv/bin/python reports/serving-problem-v2/prepare.py --source-root results/serving-problem-v1 --output /tmp/serving-pack-replay
```

No model call is made by this command. A full fresh evidence collection requires
the versioned snapshots and inspected selectors in the source receipts. Next
work is to inspect the additional retrieved baselines and resolve the model's
state/threshold definitions before designing an efficacy experiment.

## Follow-up decision

The [frozen simple-control screen](../../operator-studies/serving-admission-screen/README.md)
completed180traces and1,080arms with no censoring. Largest-first failed every
primarygroup and did not reach5% median improvement even overFIFOalone. This
ordering candidate is closed on the tested synthetic settings. The original
model output remains unchanged. The [follow-up literature audit](followup-prior-art.md)
records stronger adjacent methods and their different information assumptions.
