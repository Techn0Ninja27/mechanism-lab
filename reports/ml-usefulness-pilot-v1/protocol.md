# ML usefulness pilot: source representation and executable mechanism selection

Status: **deferred design record; not preregistered, not run, and no paid
generation authorized** (2026-11-18).

This pilot asks whether the representation used by Mechanism Lab helps a model
produce a useful, executable mechanism for a small ML-systems problem. It is a
comparison of source representations, not a novelty claim and not a benchmark
of the model's general research ability. The initial proposal was revised after
adversarial review of its controls, source identity, replicate unit, and
headroom; this document does not describe a frozen preregistration.

## Question

On the Cora graph, can a model that receives the same literature through a typed
atom/interface representation select a causal embedding-update expression that
reduces stale remote-embedding error better than the same model receiving only
the task, or the same source material as ordinary excerpts?

The representation comparison is deferred. If it is later approved, the model
arms will use the same model, temperature, candidate count, output-token cap,
question, and executable schema:

1. **Task-only:** the task, data contract, and DSL specification; no literature.
2. **Raw-source:** the task plus excerpts from the five sources in the existing
   GNN source pack.
3. **Atom/interface:** the task plus typed atoms and interface records freshly
   derived from those same five sources. Each atom retains its source ID, exact
   quote, quote hash, inputs, outputs, assumptions, and evidence level. No
   additional source is available in this arm.
4. **Raw-plus-distillation:** the raw excerpts plus a separately frozen,
   task-blind distillation of those same excerpts. This is a representation
   confound control: an atom/interface gain must survive comparison with extra
   ordinary summarization, not only with raw text.

The raw-source, atom/interface, and raw-plus-distillation prompts must carry the
same canonical source URLs/titles and content hashes from the pack (IDs are
aliases only and must not be guessed). Their lengths may differ; actual prompt
and completion costs are logged per arm, and the study makes no claim that their monetary
costs are equal.

The existing source material is in
`results/gnn-useful-proposal-v1/pack.json`. It contains SNI-GNN full-method
text and abstracts for FreshGNN, Sancus, SAT, and Sift. The existing atom
catalogue `results/ml-graph-atoms-v1/extraction/atoms.jsonl` is **not usable for
this comparison**: a repository audit found 53 atoms from 20 source IDs and
zero exact source-quote matches (0/53) in the nine GNN-pack chunks. It must not be
silently treated as an atomization of this source pack. Before generation,
create a bounded fresh atom file from the five pack files, with exact quote
spans and source hashes; the atom count and extraction method must be fixed and
recorded before any task-specific generation. Any preparation call and its cost
is logged separately and requires an explicit budget decision; this protocol
makes no claim that it has already happened. The graph study in
`operator-studies/atom-graph/` is an undirected co-occurrence baseline only; it
does not establish a useful or novel edge.

A fifth, local control samples four expressions per independent trial from the
frozen DSL grammar with recorded seeds and the same development selection
rule. Its trial count must match the model-call count per condition. It is a random-DSL baseline, not an additional model arm. A generated
candidate is one member of an arm's output, not an independent replicate; the
generation unit is the complete arm-level request, and no candidate-level
retries are permitted.

## Executable candidate contract

Each independent generation call has candidate capacity `K=4`. The number of
calls per arm must be fixed after a cost and variability assessment; one call
per arm cannot support a representation-level comparison.
The output must be JSON with exactly four candidates. Every candidate contains:

```json
{
  "id": "cand-00",
  "expression": "mix(0.5, x_last, x_last)",
  "state": ["x_last"],
  "parameters": {"...": "..."},
  "rationale": "...",
  "falsifier": "..."
}
```

The expression is compiled by a small, deterministic evaluator implemented for
this pilot. It is an arithmetic DSL over *causally observed history*, rather
than a menu of named methods:

- state values are `x_last`, `slope_ewma`, `dt`, and the previous-sync
  observation; an observation becomes available only at a synchronization
  endpoint;
- permitted constructors include scalar arithmetic, `exp`, `clip`,
  `trend(x_last, slope_ewma, dt, decay)`, and `mix(weight, a, b)`;
- constants and parameters have bounded declared ranges fixed before calls;
- an expression may not read intermediate truth, future truth, labels, test
  seeds, or a downstream metric.

The evaluator rejects unknown names, undeclared state, out-of-range parameters,
non-finite output, malformed JSON, duplicate IDs, missing fields, and expressions
that cannot be compiled. Invalid candidates are retained as failures and never
manually repaired or replaced. An arm with no valid candidate is an arm-level
failure. If a model emits more or fewer than four candidates, the raw response
is retained and the arm fails; it is not silently truncated or retried.

This contract lets a generated molecule be a composition of observed operators
while keeping execution and leakage checks independent of the generator.

Before any model call, the parser will be tested with one neutral contract
example that is not supplied as a research candidate and is not scored, such as
`mix(0.5, x_last, x_last)`. This tests syntax, bounds, and causal-state
handling without presenting a worked solution family or implying that any
expression is scientifically promising.

## Data and split

The pilot reuses the LINQS Cora archive already prepared for the GNN screen:
2,708 nodes, 1,433 features, seven classes, and 5,278 unique undirected
non-self edges. The archive SHA-256 is
`4f67c1434d39c076036039597d5326edb32c9b03c91e007ad49831c01b794ba0`.
The existing single-process four-partition emulation and causal trajectory
implementation are in
`operator-studies/gnn-sync-gate/{screen.py,prepare_data.py}`. Their tests must
pass before this pilot is frozen.

Seeds 10 and 11 are now exposed baseline-development trajectories. The
proposed comparison reserves seeds 12,
13, 14, and 15 for held-out evaluation. The graph, partition seed, model
architecture, optimizer, sync interval, warmup, schedules, and boundary-pair
set are identical across arms and seeds. The current screen's seeds 0–3 are
already exposed results and must not be reused as held-out evidence for this
comparison.

Development labels are used only to select one candidate per arm and one fixed
control mixture. Held-out trajectories are generated once after selection is
frozen. No source prompt, candidate, parameter, or control choice may inspect
held-out values.

## Controls and evaluation

The local evaluator runs every valid generated candidate and these controls on
exactly the same trajectories:

- last-observed hold;
- the existing SNI-inspired damped trend predictor;
- a fixed convex mixture of hold and trend, with its weight selected from
  `{0, 0.25, 0.5, 0.75, 1}` on development seeds only.
- a causal autoregressive lag predictor and a causal ridge-linear predictor,
  fit only from observations available at each synchronization endpoint. Their
  regularization and lag order are frozen before held-out execution. These are
  strong predictive controls, not a theoretical upper bound on achievable
  held-out error.
- a jointly tuned finite lambda/beta damping grid and a pooled continuous
  global-mixture control. The tuning must use only development trajectories and
  must be shared across both schedules; it cannot be selected separately per
  held-out schedule.

The primary metric is stale-epoch relative L2 error, averaged over the fixed
directed boundary-pair set and stale epochs:

`R = mean(||x_used - h_true||_2 / (||h_true||_2 + 1e-8))`.

For each held-out seed, report the source-arm and raw-source-arm selected
candidate, the task-only selected candidate, all fixed controls, and the paired
difference on the same trajectory. The primary comparison is the mean
held-out difference between atom/interface and raw-source; secondary comparisons
are each selected arm against the best fixed control and against task-only.
Report every seed, schedule, invalid candidate, and arm cost. Four seeds are a
small pilot: no node-level or pair-level confidence intervals are reported as
if those observations were independent replicates. Boundary-pair summaries are
diagnostic strata; inference is at the seed/trajectory level.

No closed-loop downstream accuracy claim is made in this pilot. If a generated
candidate passes the stale-error screen, a separate reviewed experiment must
implement the full training loop and evaluate held-out node labels under a new
protocol. That closed-loop regime study is pending and is outside this
document.

The free development headroom screen is evidence for control calibration only:
`operator-studies/gnn-predictor-headroom/README.md` and its local result were
published with Mechanism Lab commit `c4d5176`. It found that lambda=0.4 and
beta=0.05 improved the original lambda=0.2/beta=0.2 predictor on seeds 10/11,
so the original 5% rule is obsolete as a representation criterion. The paid
arms must beat the jointly tuned controls. Claude's independent exploratory
review is retained at `reports/ml-usefulness-review-claude-v1/` (commit
`b310d49`); it is calibration/review evidence, not an independent replicate.

If this deferred study is later approved, its falsification rules must be
reviewed and frozen before any call. They are:

- atom/interface must improve the mean held-out stale-error metric by at least
  the raw-source and raw-plus-distillation arms under a predeclared effect
  threshold, and must not lose to task-only under that same rule;
- it must beat the best jointly tuned causal control on the same held-out
  schedules. The earlier 5% comparison to the original SNI setting is an
  allocation screen only and cannot support a representation or usefulness
  claim;
- a candidate with an invalid expression, leakage, or failed finite/causal test
  cannot count as a win.

The thresholds are routing criteria for this pilot, not evidence of scientific
importance. If the arms select the same expression, the result is a valid
tie/negative representation finding.

The eventual pilot will be limited by its call and trajectory counts. Candidates within one arm are not replicates;
the arm-level generation call is the experimental unit. More than one
independent call per arm would be needed to estimate call-to-call variability,
and this design makes no fixed-power or `n=5` guarantee. A fresh held-out
trajectory result does not substitute for a separate closed-loop training
result; the gap between those measurements must be reported before calling the
engine useful for downstream ML.

## Leakage and causal checks

Before any generation call, the implementation must pass:

- existing trajectory tests for future-prefix invariance, intermediate-truth
  invariance, endpoint-before-observation scoring, separate hold state, and
  group update counts;
- DSL AST checks that reject future truth, labels, held-out identifiers, and
  any state not listed in the candidate contract;
- a canary test that changes intermediate or future observations while holding
  all observations available at the current step fixed, and obtains identical
  candidate execution and control state;
- source-equivalence checks showing that raw-source, atom/interface, and
  raw-plus-distillation arms reference the same five canonical source URLs and
  content hashes;
- a prompt audit proving that development outcomes and prior run results do not
  enter any generation arm;
- the DSL grammar and parameter ranges must be frozen before task-blind atom
  extraction, so the atom/interface representation cannot be tailored to this
  task after seeing it;
- synthetic known expressions must be used to test parser capacity, bounds,
  causal state and malformed-output handling. Passing this capacity test is a
  contract check; it does not require generated expressions to beat a real
  development control.
- equal `K`, model, decoding settings, and maximum completion cap across arms;
- a fresh output directory, immutable prompt/config hashes, and a budget ledger
  entry before the first call.

The generator is not allowed to choose the held-out seed, graph partition,
metric, or stopping rule. The evaluator, not the model, owns all execution and
selection decisions.

## Cost and reproducibility

The deferred design requires four model conditions with multiple independent
calls per condition, plus matched local random-DSL trials. A provisional planning cap is `$6`
with four arm allocations whose sum is at most `$6` (a nominal `$1.50` each),
but final cost feasibility is unresolved because
same-model call-level replication and task-blind atom/distillation preparation
also consume budget. No paid call is authorized by this document. The actual
prompt/completion token counts, model, prices, request IDs, and failures must
be copied into the result manifest. No automatic retry is allowed; any explicit
rerun requires a new output directory and a new budget decision.

The run must pin the current git commit, Python/PyTorch versions, source-pack
hashes, atom-file hash, DSL/evaluator hash, prompt hashes, seeds, and split
parameters. A single local command must reproduce candidate parsing and all
trajectory evaluation from the saved responses. The final held-out command must
refuse to run if the development selection manifest or any source/config hash
has changed.

## Pre-run review gate

This protocol is not frozen yet. Before spending API budget, review and record:

1. the exact model and completion cap, with the four arm allocations summing
   to at most $6;
2. the final DSL grammar, parameter ranges, candidate capacity, and evaluator
   behavior on malformed and adversarial expressions;
3. hashes for Cora, source pack, atom file, trajectory code, and controls;
4. causal/leakage test output and a dry-run showing no network/model calls;
5. the development/held-out seed manifest and the one-time held-out guard;
6. agreement that no downstream accuracy or novelty claim follows from this
   pilot alone.

Only after this checklist is reviewed, the replication unit and cost are
feasible, and the closed-loop follow-up is specified may the protocol be marked
frozen and any generation calls be made.
