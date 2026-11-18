# Campaign 2: improving a disclosed baseline with source representations

2026-11-18. New campaign, implementation in progress; no calls made under this
protocol. The user raised the phase budget to $30 total, preserving $5.0352276
already charged. This campaign has a $12 cap including preparation. One campaign at the fixed output path `results/generation-comparison-campaign2-run`;
no repairs or reset through another directory. Preparation has the bounded
independent redraw schedule below, declared before any call. The stopped v5
attempt remains a reported failure and none of its prepared records is used for
generation here.

## Change motivated by the stopped attempt

The previous prompt requested sourced records, while its gate required every
claim to fit inside the quoted snippets. Exact quotes were valid, yet 22/24
records exceeded their snippets. That was an insufficiently aligned extraction
and evaluation contract, not evidence that those records were false.

Here quotes are exact pointers, and support is judged against the full text of
all chunks cited by that record. Every substantive statement, summary, input,
output and condition must be entailed by those chunks. Claims supported only by
an uncited chunk do not pass. Unknowns must describe absent or unresolved
information, not smuggle in unsupported positive claims. Both preparation
prompts state this criterion explicitly before any paid call.

The same five-source/nine-chunk pack, injected-header exclusion, shared source
view, 12-record quotas (4 for the largest source, 2 each for the others), text
limits and exact-pointer validation apply to both representations. Typed fields
remain a representational difference. Up to three independent preparation requests per representation are allowed.
Select the first draw that passes its two reviews, independently for atoms and
prose; atoms draw 0 with prose draw 2 is permitted. Each draw is reviewed and its verdict recorded before requesting the next
draw for that representation. Never draw again for an already accepted
representation. Report every draw index and both reviews, including failures;
selected indices may differ and are disclosed without correction. This conditions the comparison on passing
preparation, rather than estimating performance of unfiltered preparation draws. Four sources remain abstracts, one a selected method section.

## Review gate and calibration

Each 12-record draw gets two independent fresh source-only native reviewers.
Both receive the identical frozen `reviewer-instructions.md`, the source packet
and that draw only; neither sees the task, results, prior failed draws, calibration
answers or the other review. They adjudicate every field item separately with
supporting chunk text. Record verdict is the worst field verdict. Both must
support all 12 records for that draw to pass; exactly one non-supported field
fails the draw. Reviewers cannot be replaced and no third review can override.

A failed fidelity review permits the next independent draw of that representation,
up to three total. No record is repaired. Preserve every draw, both reviews and
its outcome; once a draw passes, its selection is immutable. Provider errors stop
the campaign, not trigger another draw. If either representation exhausts three
draws without passing, stop the whole campaign: no lopsided comparison. After
both qualify, generation uses only the two first-passing artifacts. This is a
predeclared selection on source fidelity, blind to generated-candidate outcomes,
not a reset of campaign 1. Reviewer independence is procedural, not proof of
statistically independent errors. The schedule can still false-stop.

Offline calibration on the closed attempt is explicitly separated from generation.
Reviewer A supported all 24 records; an independent deterministic 12-record
sample agreed on 11 and found one uncited fixed-interval condition. Root checked
that the cited chunk names Sync/Stale modes but does not state that condition.
This is evidence of residual reviewer error and citation-scope ambiguity, not a
population error-rate estimate. All reviews and the disagreement are preserved.

## Generation and allocation

Use the same model configuration (Opus 5.5, low reasoning, 12,288 output-token
cap, omitted temperature), common task, bounded AST grammar and literal
scientific-field rendering as the preceding design. The common task discloses
the two-observation mean comparator and first-cycle h(20)/h(19)/interval-one
semantics. No measured performance or evaluation results enter any prompt.

Conditions are task-only, raw excerpts, typed atoms and prose distillation.
There are six separately sampled requests per condition, four candidates per
request, in a fixed blocked random order. Each request is a generation replicate;
its four candidates are not independent replicates. Same preparation is shared
within each prepared condition. Comparisons remain exploratory and descriptive;
there is no population novelty or representation-superiority claim.

The generation inventory has 96 model slots, 24 fixed random-control slots and
four fixed baseline-control slots. Preserve invalid slots and provider failures.
Do not select or discard candidates using model opinions. A provider failure
stops further spending and retains completed responses; no automatic retry.
The full conservative reservation must fit $12, including preparation, before
starting generation. Actual usage and shared-ledger charges are reported.

## Execution and decision

The original 112-run control screen is not rerun. The exact mean-sync AST in
`reports/generation-comparison-v2/control-selection.json` remains the comparator.
The source-informed control family and disclosed baseline make this a test of
improving a known strong baseline, not discovering that averaging beats hold.

Run every model/random/control candidate on development seeds 100–115, select
the maximum exact-window accuracy per call, break exact ties by candidate ID,
and disqualify a candidate with any failed development seed while retaining all
records. Bind picks, complete development results, input data and code hashes
before final evaluation. Final seeds are 200–231, fixed at 32 by the earlier
reference-variance allocation. Its sensitivity for candidate-minus-control is
unknown until measured; do not adjust N after seeing outcomes.

Primary metric: paired candidate-minus-frozen-control mean exact-forward accuracy
at pre-update epochs 88–119. Report own-forward window and post-update exact
endpoint accuracy as secondary, and candidate-minus-hold as descriptive context.
No held-out graph/domain claim: this is fresh-seed replication on the same task.

A positive allocation label requires no failures, improvement >0.5 percentage
points and >2 paired SE, and paired SE <=0.5 points. This is not a 95% family-wide
confirmation test. Missing/failing seed bounds use accuracy 0 and 1, never a
conditional mean over survivors. Exact duplicate ASTs count once for decision
summaries; semantic duplicates may remain.

A source condition needs at least three distinct positive picks to motivate a
separate replication; this is an allocation heuristic, not a significance test.
Two or more distinct task-only positives block representation attribution;
three or more give a task/DSL lead. A source lead requires at least three
distinct positive picks in that same condition and at most one task-only positive.
Exactly one or two source positives are descriptive, with no escalation. Positive random/control picks against the stronger comparator also
block representation attribution. A control merely matching itself cannot veto
source picks. With no positive picks, closure is eligible only if every selected
call has complete adequately precise primary results; missing/low-precision
outcomes are inconclusive. Report six-call denominators and failures explicitly.

## Provenance and spending

New schemas and output paths distinguish this campaign from v5. Freeze source,
representation, prompt, model, code, control and selection hashes. Generation
requires the successful chunk-support gate before constructing an API client.
The original stopped gate cannot be reused. The shared phase ledger charges all
calls, including BYOK routing if present; no BYOK exemption or $50 cap is assumed.

### Six-call outcomes and limits

| Outcome | Recorded interpretation |
|---|---|
| Source condition >=3 distinct positives, task-only <=1, no random/control positive | Candidate for a separate replication, not representation superiority |
| Task-only >=3 distinct positives | Task/DSL lead |
| Task-only exactly2 or source only1–2 | Descriptive mixed outcome; no representation escalation |
| Any random/control positive | Baseline/DSL headroom; no representation attribution |
| No positives, all selected calls complete and adequately precise | Eligible to close this task at these settings |
| Any no-eligible call, failed or low-precision pick | Inconclusive for closure |

Random groups now match six calls per model condition (24 slots); the inherited
sampler is a narrow perturbation family, not exhaustive random program search.
Its duplicate and effectively-hold programs remain in the record. A/A was a
stage-0 identity check and is not an additional generation-inventory slot.
The final inventory capacity is 124: 96 model +24 random +4 fixed controls.
Preparation reserves at most $3 within the $12 campaign cap; all draw calls use
the same campaign directory and phase ledger.
