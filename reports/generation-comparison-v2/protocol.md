# Matched representation pilot

2026-11-18. Implementation review in progress; no generation or measurement
results yet. This supersedes the deferred open-loop design in
`reports/ml-usefulness-pilot-v1/protocol.md`. The user authorized paid work
within the shared $20 phase budget. This campaign has a $6 ceiling including
preparation, enforced by the existing shared ledger. Only one paid attempt is
allowed; creating a new directory does not authorize another $6 allocation.

## Question and scope

Does a typed representation of a fixed literature packet help the model produce
causal estimators that improve closed-loop GCN accuracy, compared with the task
alone, raw excerpts, or ordinary prose distillation? This tests one component of
Mechanism Lab. It does not test the learned Alien sampler, the full corpus,
scientific novelty, or general research usefulness.

Four model conditions receive an identical task and executable grammar:
`task-only`, `raw`, `atoms`, and `distill`. The latter is prose distillation only,
not raw excerpts plus distillation. Three separate requests per condition each
produce four candidates. A request, not a candidate or training seed, is the
generation replicate. Request independence cannot be guaranteed by the provider.
At three requests per condition, comparisons between conditions are descriptive.

Use `anthropic/claude-opus-5.5`, low reasoning effort, no temperature parameter,
and a 12,288-token completion cap. Calls run in three deterministically shuffled
blocks (seed 20261118). No repair calls, retries, or outcome-dependent additions.
Actual prompt/completion tokens and costs are retained; equal input lengths or
costs are not claimed.

## Matched preparation

The fixed packet is `results/gnn-useful-proposal-v1/pack.json`: five sources,
nine chunks, a selected SNI method section and four provider abstracts. SNI
accounts for about 72.5% of quotable characters. This is a narrow and uneven
packet, not comprehensive literature coverage. The common rendered source view
removes the known injected section header from quotable text while retaining
the original packet unchanged.

Two task-blind-intent preparation calls produce typed atoms and prose records.
Both require 12 records with identical quotas: four for the largest source by
included characters, two for each remaining source. Ties use source ID order.
Both preserve exact source-linked evidence and unknowns; prose records have no
typed input/output ports. Statement and summary fields share a 2,400-character
cap. Typed records still have additional port/condition fields: matching record
counts and overall caps does not guarantee equal information volume. The soft output target is 18,000 UTF-8 bytes and the
hard canonical validated cap is 24,000 bytes. Exact quotation validates source
attachment, not semantic fidelity. Review fidelity and scan for task/result
leakage before freezing generation prompts. Record all schema failures without
repairing them. One failed preparation ends this campaign attempt.

## Executable task and controls

Freeze the implemented AST compiler and evaluator before preparation. Programs
have bounded `predict` and `update` expressions, with only observed states,
independent per-layer memory, age and observed interval. Hidden exact current
states and labels are unavailable to the estimator between synchronization
steps. Candidate failures remain in the results.

The task is Cora, three GCN layers, four random partitions, constant learning
rate, synchronization interval 16, 120 epochs and the existing warmup through
epoch 20. The primary metric is mean exact-forward classification accuracy at
pre-update epochs 88–119. Own-forward accuracy over the same weights/window and
final exact accuracy are secondary descriptions. This selected regime and the
source choices were informed by earlier exploratory studies: this is an exposed
task with fresh seed replication, not an untouched benchmark.

Controls are hold; three groups of four random ASTs sampled by the fixed
implementation (seed 20261119); and one four-candidate control group: capped extrapolation with gains -0.5,
-0.25 and 0.1, plus the mean of the last two synchronized states. The horizon
is capped at one observed interval. Random candidates are bounded perturbations
of hold, rather than arbitrary programs that can ignore observations. A separate
algebraically hold-equivalent A/A program checks the execution path. The linear group has the same development evaluation
capacity as one model call. It is not exhaustive optimization over the DSL.
Previously tested SNI and lag-one extrapolators are reported as historical
context only, not mixed with fresh-seed comparisons. Beating these narrow controls
does not establish superiority to all classical estimators.

## Selection and final evaluation

Run every candidate on development seeds 100–115. Select the maximum mean
primary accuracy within each call/group; break exact ties by candidate ID.
A candidate with any failed development seed is ineligible, but its failures
and successful records remain visible. Calls with no eligible candidate remain
explicit failures. Freeze selections and bind them to the inventory, complete
development results, code and data before loading final evaluation data.

Evaluate each frozen selection and hold on final seeds 200–231. Do not change
programs or select on final performance. Report each call separately, paired
candidate-minus-hold differences, seed variation and failures. Incomplete final
runs get lower/upper sensitivity bounds imputing accuracy 0 and 1; never promote
a conditional mean over surviving seeds. A positive label requires zero failed
final seeds, mean improvement above 0.5 percentage points and above twice its
paired standard error. This is an allocation rule, not a 95% confirmation test.
Report approximate paired-t intervals descriptively and do not turn multiple
call-level intervals into a family-wide claim.

After development, record paired seed SD and projected final SE (SD/sqrt(32))
for each pick. If projected SE exceeds 0.5 percentage points, label that pick
low precision; do not add seeds or alter the threshold. Final uncertainty and
call-to-call variation remain separate.

## Decisions and stopping

If nobody beats hold and the evaluated picks have adequate precision, close
this task for the engine at these settings. Low-precision or failed picks make
that outcome inconclusive, rather than evidence for closure. If random
or linear controls also improve, report easy DSL/control headroom without crediting
typed representation. At least two distinct positive selected ASTs in a source condition
and no positive random/control groups can motivate a separately designed replication
only if task-only does not also have two distinct positive picks;
it is not evidence of a significant difference between representations. Exact AST
duplicates are counted once for this allocation rule and reported across calls;
this does not detect all semantic equivalences. If task-only also succeeds, the
lead belongs to the task/DSL, with no representation attribution. Sources
performing worse than task-only is also a reportable outcome. Do not select the
most flattering interpretation after results arrive.

All successful, invalid and failed candidates stay in the inventory. A provider
failure stops additional calls and preserves completed responses and outstanding
budget reservations. Local checkpoints may resume only under identical frozen
inputs; failed jobs are retained and never silently retried. No more Cora
diagnostic grids or adaptive model calls belong to this campaign.

Freeze manifests must bind source views, representations, prompts, model settings,
code, data, controls, seeds and selection inputs. Public reports omit private raw
call logs and full source excerpts. Before execution, the real modules must pass
an end-to-end mocked integration test and source-parity review. A protocol file
alone is not evidence those gates have passed.


## Pending stage-0 amendment (not yet implemented or frozen)

Before paid preparation, execute exactly one free screen of hold, the four
controls, the A/A program and exact-detached G1 on development seeds 100–115.
This measures reference precision, not candidate precision or model-arm power.
Using only the paired seed SD of G1 minus hold on the primary metric
(mean exact-forward accuracy over pre-update epochs 88–119), propose N = 16 * ceil(max(32, ceil((SD / 0.005)^2)) / 16). Stop before API
spending if any hold/G1/A/A seed is missing or nonfinite, the exact A/A fails identity,
or N exceeds 128. Control failures and their paired SDs are descriptive and
do not determine N. Otherwise final seeds would
be 200 through 199+N. No additional reference arms, reruns or control changes
follow this screen. This proposal supersedes the fixed-32 precision design only
after the code, all manifests and analysis agree; the current audit plan remains
an immutable superseded design artifact, not an executable approval.

Candidate-minus-hold variation may differ from these references. Actual final
SE above 0.5 percentage points or incomplete results remain low precision and
cannot support closing the task. G1 is a reference, not an upper bound. No
claims follow from a stage-0 plan file without executed results.


### Precision interpretation and A/A scope

The N calculation uses a point estimate from 16 development seeds, not an upper
confidence bound or a power guarantee. At realised SE=0.5 percentage points,
the allocation rule requires an observed gain above 1.0 point. Under an
approximate normal model with known SE, power is roughly 50% for a true 1-point
gain and 84% for a 1.5-point gain; these are illustrations, not predictions for
generated candidates. A null does not rule out smaller useful gains.

The exact A/A checks deterministic execution equivalence only. It is not a chaos
floor. A tiny input perturbation would be a different sensitivity experiment and
is outside this one-screen design. Random controls instead have a known
pointwise bound: abs(perturbation) <= 0.1 * abs(last_sync). Actual changes may be
smaller, including zero entries; report duplicates and do not resample.

### Fixed preparation fidelity gate

Before generation, a fresh reviewer receives only source excerpts and prepared
records, without the task or experimental results. Review every substantive
statement, port, condition and summary against its own cited evidence, using
supported / unsupported / contradicted / indeterminate. Supported means the
quoted evidence entails the claim; unsupported means it does not state or
justify the claim; contradicted means it explicitly conflicts. Indeterminate
means malformed evidence or an unresolved referent prevents adjudication, not
reviewer hesitation. The reviewer is a fresh native agent, with no external API
charge. A small fixed calibration set is reviewed once before paid preparation;
its labels and all verdicts are retained. This small hand-built set cannot
estimate a reliable population false-flag rate. Unknowns are checked
as statements of missing information, not demanded as positive source claims.
Save all record-level verdicts and explanations with source/record hashes.

Apply identically to both representations: stop the single paid attempt if any
record is contradicted, or any record is unsupported/indeterminate. No repair,
resampling or replacement source is allowed. This strict gate estimates the
pipeline conditional on faithful preparation and may stop it; it does not
estimate unconditional representation performance. Reviewers may not reject
records for favouring any estimator. If a task-exposed reviewer participates,
disclose the exposure rather than claiming blindness.
