# World-diversity search: an evaluation-first protocol

This document adapts Henrik Westerberg's world-diversity and ontology-governed
search to the Alien scorer. It is a protocol, not evidence that the method
works. The released Ontology of the Alien study is one retirement-design
problem, with path-dependent traces; its own README says that usefulness,
external novelty, world coverage, and real-world validation were not measured.
The method should therefore be tested against matched controls before it is
used to mint new scorer atoms.

Primary references:

- [Orthogonal Insight Toolkit](https://github.com/emergent-wisdom/orthogonal-insight-toolkit)
- [Ontology of the Alien](https://github.com/emergent-wisdom/ontology-of-the-alien)
- [Meaning Model loneliness record](https://github.com/emergent-wisdom/meaning-model/tree/main/examples/alien-search-loneliness)
- [Fractal Intelligence](https://github.com/emergent-wisdom/fractal-intelligence)
- [Sema](https://github.com/emergent-wisdom/sema)

## Question

Does changing the causal premises before solution generation produce more
distinct, prior-art-resistant and testable mechanisms than direct generation
or random inspiration at the same model-call budget?

The unit of analysis is a **mechanism**, not a differently worded proposal.
Each candidate must state the changed intervention, the causal pathway, the
expected observable outcome, and the cheapest experiment that could falsify it.

## Small pilot: 24 proposals

Use two problems with executable or otherwise concrete tests. One should be an
LLM-research problem; the second can be a public Kaggle task after its data and
metric are pinned. Avoid a task where the answer is only a taste judgment.
For each problem, run four independently sampled seeds in each arm:

| Arm | What the generator sees | Purpose |
|---|---|---|
| Direct | problem, constraints, and output schema | ordinary baseline |
| Random seed | problem plus a seed word; no persistent ontology | separates random framing from world construction |
| World detour | solver sees the problem and an invented world, but not why it was made | tests altered causal premises |

This gives 2 problems × 4 seeds × 3 arms = 24 final candidates. Use the same
model family, temperature policy, output schema, and maximum token budget in all
arms. Count builder, solver, compiler, and judge tokens separately. Do not call
the world arm “matched” if it uses three model calls and the direct arm uses one:
either give every arm the same total budget or report a cost-adjusted curve.

The first pilot should cap each final candidate at 1,200 output tokens and each
world/solver/compiler role at 1,500. A cheap local or API model can produce
drafts; a stronger model is reserved for the blinded review set. Keep four
seeds per arm rather than treating 24 outputs as independent replications.

## Blind role separation

The separation is the causal intervention. Preserve each role's input and
output as immutable JSON with a hash.

1. **Builder** receives only a seed and a world schema. It does not see the
   target problem, candidate atoms, prior-art retrieval, or evaluation metric.
   It writes 3–5 explicit laws, their scope, and at least one consequence that
   differs from ordinary reality. Reject worlds that are merely metaphors.
2. **Solver** receives the target problem, constraints, and one world. It does
   not receive the seed, the fact that the world is experimental, the ontology,
   or prior candidates. It proposes an intervention that follows from the
   world's laws and names the causal steps.
3. **Compiler** receives only the world, solver output, and target-domain
   interface. It extracts a mechanism and translates it to reality. It must
   list each mapping, each disanalogy, and which world assumption is discarded.
4. **Reviewer** receives standardized mechanisms in randomized order, with the
   arm and seed removed. Retrieval is run on the mechanism's exact claims. The
   reviewer sees source IDs and abstracts, never generator metadata.

The builder/solver boundary must be checked automatically: fail a run if the
builder prompt contains target terms or the solver prompt contains the seed.
The compiler may see both because its role is translation, not generation.

## Persistent ontology condition

The ontology is a separate fourth arm only in a later experiment. Do not add it
to this 24-candidate pilot or the world effect becomes unidentifiable. In the
follow-up, randomize whether the Taxonomist receives prior candidates and use
the same archive size and call budget in the tabu control.

Store each accepted and rejected mechanism as a content-addressed record:

```json
{
  "id": "sha256:...",
  "problem_id": "...",
  "mechanism": "...",
  "causal_steps": ["..."],
  "predicted_outcome": "...",
  "falsification_test": "...",
  "prior_art_ids": ["..."],
  "status": "unreviewed|covered|incremental|plausibly_distinct|rejected",
  "human_adjudicated": false
}
```

Hashing gives a stable identity and audit trail. It does not prove that two
records are semantically equivalent or correct; those remain review questions.
This is the useful part of Sema to borrow without treating a content hash as a
novelty oracle.

## Evaluation

Run the same bounded retrieval for every arm. Require a reviewer to quote the
source abstract supporting each overlap and distinguish:

- **covered**: the proposed causal mechanism is already present;
- **incremental**: the mechanism is known but the intervention or boundary is
  a precise, testable extension;
- **plausibly distinct**: no retrieved source establishes the same mechanism;
- **insufficient evidence**: retrieval is too weak to decide.

The last category is essential. “Not found” is not “novel.” Human adjudication
is required before any record can become a training label or a new atom.

Primary outcome: the proportion of candidates reaching `incremental` or
`plausibly_distinct` **and** passing a blinded feasibility review. Secondary
outcomes are mechanism diversity, retrieval overlap, falsification-test quality,
and cost per surviving candidate. Report Wilson intervals or bootstrap intervals
over seeds; do not claim significance from 24 correlated outputs.

Measure diversity at the causal-step level, not lexical distance. Have a blind
reviewer normalize each mechanism into operator, object, intervention,
mediator, and outcome. Compare exact normalized tuples, graph edit distance, and
embedding distance. Inspect examples because every metric can reward verbosity.

For the executable test, run a cheap phase-zero experiment before any expensive
training or deployment. A candidate fails if its intervention cannot be
implemented from the stated inputs, its control is missing, or its predicted
direction does not beat the pre-registered baseline. World-model simulations are
diagnostic: they show what follows from invented laws, not what works in reality.

## Kaggle extension

Use Kaggle only as a testbed for executable mechanisms, not as evidence of
scientific novelty. Pin a competition snapshot, train/validation split, metric,
time limit, and allowed data. Ask each arm for a complete change to a baseline
pipeline: feature construction, architecture, loss, sampling, or validation
design. Require runnable code and a predicted metric change before revealing
the public leaderboard.

Evaluate on a private holdout or repeated local splits. The final test must be
locked before generation to prevent leaderboard overfitting. A world-generated
idea counts only if its named intervention improves the matched baseline under
the same compute budget and survives removal of each claimed component.

For a first Kaggle pilot, use 12 candidates (four per arm) on one tabular
competition. This is cheaper than 24 full submissions and exposes whether the
method produces executable, non-collage changes. Only expand to two problems
after the LLM-research pilot has a working provenance and review pipeline.

## Failure modes to record

- World rules are decorative and do not alter the solver's causal reasoning.
- Solver outputs are ordinary ideas with world nouns substituted.
- Compiler adds familiar components and calls the bundle new.
- Ontology categories steer later agents toward the curator's vocabulary.
- A diversity metric rewards long or fragmented text.
- Simulated outcomes are mistaken for real-world validation.
- Retrieval misses the exact prior art or the reviewer self-confirms novelty.
- Extra role calls give the world arm more search budget.

The cheapest kill test is to compare world and random-seed outputs after
mechanism normalization and blind prior-art review. If the world arm's apparent
advantage disappears after cost matching, component ablations, or expert review,
stop before expanding the atom vocabulary.
