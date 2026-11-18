# Campaign 2: improving a disclosed baseline with source representations

2026-11-18. New campaign, implementation in progress; no calls made under this
protocol. The user raised the phase budget to $30 total, preserving $5.0352276
already charged. This campaign has a $12 cap including preparation. One attempt;
no repairs, regeneration or reset through another directory. The stopped v5
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
remain a representational difference. One independent preparation request per
representation is used; this does not estimate variability across preparation
draws. Four sources remain abstracts, one a selected method section.

## Review gate and calibration

A fresh source-only native reviewer sees only the packet and prepared records,
not the task, control results or generation prompts. It reviews all 24 records
against their cited chunks. Verdicts are supported, unsupported, contradicted or
indeterminate; the latter means malformed evidence or unresolved reference, not
hesitation. Record-level reasons, cited chunks and input hashes are retained.
Any non-supported record stops this attempt, without repair or resampling.
A second reviewer may audit the gate, but may not reverse a failed gate to
continue this campaign. Review is a fallible instrument, not proof of truth.

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

The generation inventory has 96 model slots, 12 fixed random-control slots and
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

A source condition needs at least two distinct positive picks to motivate a
separate replication; this is an allocation heuristic, not a significance test.
Two distinct task-only positives give a task/DSL lead without representation
attribution. Positive random/control picks against the stronger comparator also
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
