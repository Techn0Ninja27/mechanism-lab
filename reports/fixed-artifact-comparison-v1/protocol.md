# Fixed-artifact usefulness comparison, prospective design

2026-11-19. Implementation draft; no calls or training under this protocol.
This is a new experiment. Campaign 2 remains stopped and immutable. The choice
to conduct this comparison was made after seeing its atom fidelity reviews.

## Question and inputs

For these particular source artifacts, does a generator produce executable
programs that improve the fixed mean-of-two-observations control? This is a
fixed-artifact case study, not a population estimate of atomization, a pure
representation effect, or evidence of novelty.

Use exactly the twelve atoms from Campaign 2 draw 0, byte-for-byte, with the
original call, draw, plan and failure hashes attached. Label them “unselected,
gate-failed draw 0.” Do not edit, remove or select records. Generate exactly one
prose artifact using the Campaign 2 prose preparation prompt at its frozen hash,
with Opus 5.5 and its original preparation settings. No redraw or fidelity-based
selection. Both artifacts are used regardless of semantic-review findings.

Mechanical validity means JSON parsing plus the complete frozen preparation
validator, including quotas, length and evidence-pointer/quote checks.
If prose fails mechanical parsing, retain the raw response and mark the prose
arm unavailable. Its six planned call slots under each decoder remain missing/
no-eligible outcomes, not successes or discarded calls. Stop before all candidate generation; preserve every planned slot as unattempted.
Do not spend on a reduced three-arm comparison. A provider failure
instead stops all further spending. An incomplete response is preserved; it
is never repaired or retried.

The raw arm uses the original complete source pack. Raw, atoms and prose receive
the same source identity header. Task-only receives no source material. Keep the
original TASK and GRAMMAR byte-identical. No performance, review, adjudication
or failure-status text enters any generator prompt. Preparation is task-blind.

## Decoders and call schedule

Cross both `anthropic/claude-opus-5.5` and `anthropic/claude-fable-5.2` with all
four arms: task-only, raw, atoms, distill. Each has six separate calls with four
candidate slots per call: 48 model calls and 192 candidate slots total. Use low
reasoning effort, omitted temperature, and 12,288 output tokens for both. Do not
vary model by arm. Preparation remains Opus; model comparison concerns generation.

Freeze the same blocked arm/call order for both decoders, with decoder order
within each case randomized by a fixed seed. Fable runs are planned now, not
added in response to an Opus result. Interpret each decoder separately. If either decoder is incomplete, make no cross-decoder comparison. For two
complete decoders, different descriptive labels mean “decoder-dependent; no common
representation reading.” Equal labels mean equal categories only, not equal source-arm results.
Never pool positive calls across decoders.

Reuse the 24 seeded random controls and four fixed controls from the final
Campaign 2 plan, by exact hash. They form six random-call groups and one fixed-
control group. Shared deterministic control executions may be reused only with
matching evaluator/data/seeds/program hashes; otherwise execute them once and
reference the same results in both decoder reports. No model results select controls.

## Fidelity is a fixed descriptive table

Before candidate generation, save and hash one table per prepared artifact:
record/field identifiers, field type, number of valid reviewer slots, flags by
either reviewer, flags by both, missing/malformed review outputs, and the existing atom adjudication, frozen before generation. No later
adjudication enters the primary fidelity table. Report `unknowns` separately. Primary fidelity count
is “flagged by both,” not “known false.” Neither agreement nor adjudication is
formal truth. Atom adjudication found two source-support errors, three false
reviewer flags and one ambiguous rule in this one draw.

Use the exact existing record-isolated source-only review instructions for the one prose
draw, bound by file hash. Its scientific-field rule applies to the prose summary
as well as typed fields; its compound-field rule applies to both. Field counts
and flags are descriptive within each artifact: a long summary field and a short
typed field are different denominators, so do not compare their rates as quality
estimates. Reviewers are native Codex contexts; the serving model/version is not
independently pinned, a limitation recorded with launch metadata. Two native review contexts per record; no output repair or replacement.
Malformed/missing review output is recorded as missing measurement and cannot
stop generation, choose a different artifact or change its text. Do not give
reviewers task results or other reviews. No fidelity/performance regression or
“as-prepared” estimand: one artifact per arm cannot estimate that relationship.

Every positive prepared-arm pick must display the same artifact's flagged-field
list and measurement coverage beside its performance. A lead is explicitly
“for this as-produced artifact, with N fields flagged by both reviewers.” A null
is a null for this artifact; do not attribute its cause to fidelity defects.

## Execution and descriptive decision rule

Freeze source/artifact/prompt/model/code/control/data hashes before generation.
Use development seeds 100–115 and exact-forward mean accuracy over epochs 88–119.
Run every valid candidate, preserve invalid slots, and select one candidate per
call on development only. Tie-break by candidate ID. Any failed development seed
makes that candidate ineligible, never erased. Bind all development results and
picks before fresh-seed evaluation on 200–231. The final seeds have not been used
in the earlier preparation-only attempts.

Primary final metric is paired candidate-minus-frozen-mean-control accuracy.
A positive allocation label requires no failed final seeds, mean improvement
>0.5 percentage points and >2 paired SE, with SE <=0.5 percentage points. This
is not family-wise statistical confirmation. Report own-forward and endpoint
metrics secondarily. Missing/failing accuracy bounds are 0 and 1; no survivor-
only average. Deduplicate exact ASTs within each arm's positive count, while
acknowledging that syntactically distinct programs may behave identically.

Apply the following to EACH decoder and EACH source condition's six calls:

| Outcome | Descriptive label |
|---|---|
| Any missing/no-eligible/failed/low-precision call | Incomplete; cannot close or attribute |
| Any random or fixed-control pick positive | No representation attribution |
| Task-only >=3 distinct positives | Task/DSL lead |
| Task-only exactly 2 | Attribution blocked |
| Atoms >=3, raw <=1, prose <=1, task-only <=1, controls 0 | Lead for this atom artifact |
| All three source arms >=3, task-only <=1, controls 0 | Source-artifact lead; no typed-structure reading |
| Raw or prose >=3, atoms <=1, task-only <=1, controls 0 | Evidence against this atom artifact on this task |
| Other positive combinations | Mixed source result; no predeclared lead |
| No positives and every call complete and precise | Close this fixed-artifact case at these settings |

The executable rule is `generation/fixed_artifact_decisions.py`, hash-bound in
the final plan. Three positives pooled across different source arms never qualify.
No automatic deployment, training-label promotion or novelty assertion follows.

## Spending and attempt boundary

One output path: `results/fixed-artifact-comparison-v1`. One new campaign cap:
$24.85, including the single prose preparation. This is inside the existing
$30 phase cap, which already has $5.1417516 charged. BYOK calls count toward the
same limits; credit balances are not exemptions. Provider routing and prices
must be frozen, and actual BYOK use is reported from receipts, not assumed.

Before the prose preparation request, reserve enough headroom for the full
schedule using the stated conservative prose-size bound. Before any
candidate-generation request, recompute the full remaining schedule's
conservative reservations from the actual frozen prompt bytes, output caps,
provider prices and applicable fees. It must fit both caps. If it does not, stop
before generation; do not reduce a model/arm after inspecting candidate results.
The read-only provider check supports the low-effort setting and output cap.
Reasoning shares that cap with visible output, so truncation risks can differ
by decoder. Standard-price reservation is $23.619524; include a separate 5%
BYOK-fee contingency, yielding $24.8005002 and a phase total of $29.9422518.
Do not assume the workspace free allowance or promotional credits.
See provider-check.md and provider-metadata.json; actual routing remains to be
verified from receipts. Exact configuration must pass preflight.

A provider exception stops further requests across both decoders. Truncated
responses retain their four invalid candidate slots and stop further spending.
No retries, alternate-directory resets, model substitutions or unrecorded repair.
All planned slots remain in the report, including unattempted ones. Local execution
is checkpointed and reproducible; resuming a verified local job is not an API retry.

A follow-up after this experiment requires a new prospective question and plan.
This protocol does not reopen Campaign 2 or relabel any of its failed gates.

## Execution records

The immutable base plan grants no paid execution authority. A separate initial
freeze, matching independent audit receipt, and explicit execution request are
required before either paid stage. Preparation and generated prompt files have
separate directories so the base plan is never rewritten after a call. Native
review launch intents precede spawning, with completion records saved separately;
these are procedural records, not proof of independent model judgement.

Completed response checksums are anchored in the shared phase ledger. A verified
completed receipt can rebuild a local checkpoint without a new request. An
unverifiable or failed request is never resent automatically. Candidate exports
are checked against those receipts, and exact code, runtime and data bindings
are retained. See execution-status.md for the command sequence. The initial
plan's zero paid-call count describes its creation, not subsequent runtime use.
