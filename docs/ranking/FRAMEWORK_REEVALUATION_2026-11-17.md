# Scoring reevaluation after the framework audit

2026-11-17. Requested priorities: useful ML research proposals, reconsidered
scoring, and lessons from Connected Papers, MiroFish and research frameworks.
This is an implementation/source audit and a proposed evaluation protocol. No
external framework was installed, no benchmark was reproduced, and no paid
model calls were made. Popularity is not evaluation evidence.

## What our current scores actually mean

`generation/rerank.py` combines standardized coherence and unavailability with a
beta weight, optionally enforcing coherence and pair-diversity constraints.
It does not measure task benefit, empirical feasibility or contribution-level
prior-art overlap. `local_runner/proposal_v2.py` previously used that ordering to pick
seeds despite stripping scores and disclaiming ranking validity. The later
[selection-route correction](../../reports/selection-route-audit-v1/README.md)
replaced this with source-backed uniform triplets; beta no longer influences
that selector. This does not repair the learned scorer. A larger search corpus has not repaired
that model or retrained it.

`generation/dual_ranker.py` is a useful selection scaffold: it separates axes,
preserves unknowns, limits gain comparisons to the same task/metric/baseline/
budget, and offers exploration. It consumes supplied values; it does not infer
or calibrate them. No caller was found in the inspected production generation
paths. `generation/utility_ranker.py` is a narrow compression predictor whose
held-out study failed against a fixed baseline, not a research-value model.
Recent grounded runs produce individual proposals and reviews, not a validated
ranking of alternative research directions.

Increasing beta or inventing new weights would therefore not address the main
gap. We need evidence-producing components and tests of their judgments.

## Framework findings

**Connected Papers — related-work expansion.** Its official documentation
uses co-citation and bibliographic coupling to form paper-similarity graphs;
direct citation is not required. Prior/derivative views help trace surrounding
work. Borrow the retrieval principle: seed paper -> references/citing work ->
related mechanisms and baselines. Graph distance is not a novelty or usefulness
score. The current index chiefly stores abstracts/metadata, not a complete
citation graph. The exact proprietary weighting was not reproduced, and no
public integration API was verified. [Official explanation](https://www.connectedpapers.com/about),
[features](https://www.connectedpapers.com/).

**MiroFish — simulation workflow and trace inspection.** The upstream pipeline
builds a graph from source material, generates personas/configuration, runs social
interaction simulations and produces reports. Its implementation uses Zep and
OASIS. Explicit configurations and logs are useful design references. Generated
social agents are not independent scientific reviewers; the audit found no
validated ML novelty/usefulness scorer. Do not use swarm agreement as confidence.
[README](https://github.com/666ghj/MiroFish),
[simulation configuration](https://github.com/666ghj/MiroFish/blob/7657031ac01184afe2cb220f5ee3545573b5e843/backend/app/services/simulation_config_generator.py),
[report agent](https://github.com/666ghj/MiroFish/blob/7657031ac01184afe2cb220f5ee3545573b5e843/backend/app/services/report_agent.py).

**ClaimNov — contribution-level overlap.** It extracts typed claims, retrieves
same-dimension prior claims, judges coverage, and gives the full profile to a
final novelty judge. This is closer to our need than a whole-proposal similarity
score: known components need not invalidate a distinct central contribution.
Its paper evaluates disguised, disputed, later-realized and hybrid proposals,
but uses bounded reference pools and partly model-derived labels. Reported
performance is not reproduced here. [Paper](https://arxiv.org/html/2610.36511v1),
[code](https://github.com/emmatliu/claimnov).

Code audit at commit `413f16c7f94b936fd98c2e554b20db868a436f6a` found two adoption
hazards: empty entailment lists give zero coverage in `aggregate.py`, which can
produce maximal numerical novelty; `profile_veto` in `gate.py` defaults malformed
JSON to a novel decision. Its unresolved flag checks missing claims, not all
missing comparisons. We would require explicit unknown/error states and minimum
coverage per central claim. These are inspected code paths, not measured error
rates or a claim that every caller mishandles them. No license file was found in
the inspected tree, so this audit proposes design lessons, not vendoring code.
[Aggregation](https://github.com/emmatliu/claimnov/blob/413f16c7f94b936fd98c2e554b20db868a436f6a/src/novelty/aggregate.py),
[gate](https://github.com/emmatliu/claimnov/blob/413f16c7f94b936fd98c2e554b20db868a436f6a/src/novelty/gate.py).

**DeepInstructor / ExpInstructor — evidence-backed criticism.** It retrieves
structured experience from peer reviews by evaluation dimension. Borrow
traceable concerns with their sources rather than unsupported numerical ratings.
The public code exposes separate novelty, significance and feasibility evaluation
paths. Reviewer agreement is not experimental utility, and review-derived
judgments can favor conventional work. Its graph and evaluation labels need
leakage checks before adaptation. [Paper](https://arxiv.org/abs/2609.22104),
[repository](https://github.com/Prongcan/ExpInstructor).

**ResearchAgent — problem, method and experiment as separate objects.** It
retrieves papers/references and entities, then iteratively generates and reviews
those objects. This is a useful baseline for testing whether our atom-based
approach adds anything. Improvements in reviewer ratings do not establish real
research success. [Official repository](https://github.com/JinheonBaek/ResearchAgent),
[paper](https://aclanthology.org/2025.naacl-long.342/).

**AIDE / AI Scientist-v2 — execution guided by measured outcomes.** AIDE searches
over code drafts, repairs and improvements against a specified metric. AI
Scientist-v2 builds on that search pattern for experiments. Borrow bounded search
and retained branches once a task, strong control and evaluation protocol are
fixed. Metric optimization demonstrates engineering performance, not necessarily
a scientific contribution. [AIDE](https://github.com/WecoAI/aideml),
[AI Scientist-v2](https://github.com/SakanaAI/AI-Scientist-v2).

**EvoScientist — scoped failure and implementation memory.** Separate ideation
and experimentation memories can avoid repeated mistakes. Our existing scoped
negative-result records are a start. Any future learned memory policy needs
reset-memory and cross-task controls; remembered outcomes must not leak into
held-out scoring. [Paper](https://arxiv.org/abs/2603.08127),
[repository](https://github.com/EvoScientist/EvoScientist).

## Proposed scoring contract

Keep evidence and unknowns visible before producing any ranking:

1. **Problem value:** who needs the result, which task/metric matters, strongest
   existing solution, and what practical or scientific question remains.
2. **Contribution overlap:** each central claim paired with the closest prior
   contribution and concrete difference. Distinguish proposed outcomes from
   observed results. Missing retrieval means unknown, not novel.
3. **Feasibility:** required data, hardware, observable inputs, implementation
   dependencies and smallest decisive experiment. A failed cheap test narrows
   the claim rather than discrediting an entire source pair.
4. **Expected research value:** plausible benefit over the strongest control,
   uncertainty, experiment cost, and what a negative result would teach. Compare
   numerical gains only within compatible tasks and budgets.

Do not immediately collapse these into a weighted sum. Route unsupported claims
to evidence gathering, clear duplicates to revision, invalid mechanisms to
rejection, and worthwhile uncertain claims to bounded testing. Exploration can
remain explicit without labeling it high confidence.

## First evaluation, before promoting a replacement scorer

Use a fixed development pool containing known duplicates, paraphrased duplicates,
partial-overlap proposals, coherent-but-useless ideas, useful incremental ideas,
and unresolved cases. Include our observed failures, but label that part exposed.
Use separate, untouched paper/task families for the eventual generalization test.

Compare (a) current available proxy signals, (b) a direct structured reviewer,
(c) claim-level retrieval, and (d) claim-level plus citation-neighborhood retrieval
under the same source/model/token budget. Do not let a new scorer see experiment
outcomes when predicting them. Count duplicate detection, erroneous rejection of
useful partial novelty, correct prior-art attribution, unsupported confidence,
rank stability under paraphrase/order, and realized benefit per experiment cost.
Human review is needed for disputed scientific judgments; model agreement alone
is not the ground truth. A cheap retrospective pilot cannot justify general
calibration claims.

The immediate priority is a bounded claim/evidence layer plus a strong-baseline
check. A full MiroFish-style simulator or large review-graph import would add
cost without first resolving those two demonstrated failure modes.

## First outcome-withheld usefulness probe

A [retrospective paired judgment probe](../../operator-studies/utility-judgment-pilot/README.md) has now been run on three completed local experiments. Six Haiku presentations (both orders) produced three abstentions and one correct preference among three answered presentations. Each pair changed between abstention and preference. Outcomes were withheld from the model, but the cases were selected after inspecting outcomes, so this is exposed development evidence. It does not support promoting those probabilities to usefulness scores or training labels; it also does not establish the accuracy of stronger models or evidence-enriched reviews. Cost/accounting and convex-mixture reasoning errors are recorded for a future prospectively fixed comparison.
