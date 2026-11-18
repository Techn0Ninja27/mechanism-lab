# Matched representation pilot

2026-11-18. Implementation review in progress; no generation or measurement
results yet. This supersedes the deferred open-loop design in
`reports/ml-usefulness-pilot-v1/protocol.md`. The user authorized paid work
within the shared $20 phase budget. This campaign has a $6 ceiling including
preparation, enforced by the existing shared ledger.

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
typed input/output ports. The soft output target is 18,000 UTF-8 bytes and the
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
implementation (seed 20261119); and one four-candidate linear extrapolation group
with gains 0, 0.25, 0.5 and 1. The linear group has the same development evaluation
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

If nobody beats hold, close this task for the engine at these settings. If random
or linear controls also improve, report easy DSL/control headroom without crediting
typed representation. At least two of three positive calls in a model condition
and no positive random groups can motivate a separately designed replication;
it is not evidence of a significant difference between representations. Sources
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
