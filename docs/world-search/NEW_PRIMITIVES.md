# Beyond a fixed atom vocabulary

## The problem

The released Alien Science scorer assigns every candidate to a fixed set of
literature-derived atom symbols. It can search for unusual combinations of
those symbols, but it cannot give a new primitive a learned coherence or
availability score. A decoder can coin a new label; that is not evidence that
the label names a new mechanism. The system therefore needs a separate
**primitive-discovery loop** rather than simply adding more atom IDs.

## What prior work actually offers

**DreamCoder** learns a symbolic library while solving a stream of program
induction tasks. Its wake phase solves tasks and its sleep phase refactors
solutions into reusable abstractions, then trains a search policy over the
expanded language. The abstractions are evaluated by compression and by
generalization to held-out tasks, not by their names. It is a useful model for
alternating discovery and reuse, but its domains have an executable program
semantics that research prose does not automatically provide.

Primary source: [Ellis et al., DreamCoder (2020)](https://arxiv.org/abs/2006.08381);
published version: [PLDI 2021](https://doi.org/10.1145/3453483.3454080).

**Stitch** performs corpus-guided top-down library learning. It searches for
abstractions that compress a corpus of programs, and reports comparable or
better library quality than deductive DreamCoder at much lower cost. This is a
strong engineering pattern for turning repeated solution structure into a
candidate primitive. Compression alone is insufficient for science: a
frequent artifact or an overly broad macro can compress well without making a
new causal prediction.

Primary source: [Bowers et al., Top-Down Synthesis for Library Learning
(2022)](https://arxiv.org/abs/2211.16605).

**Novelty search and quality-diversity** separate exploration from a single
fitness optimum. Novelty search rewards behavioral difference, while
MAP-Elites and related quality-diversity methods retain high-quality solutions
across behavior niches. This supplies a useful archive design for our
mechanism ledger: retain candidates by causal behavior, not just by text or
embedding distance. It does not tell us whether a niche is scientifically new.

Primary sources: [Lehman and Stanley, Abandoning Objectives (2011)](https://doi.org/10.1162/ARTL_a_00057)
and [Mouret and Clune, Illuminating search spaces by mapping elites (2015)](https://arxiv.org/abs/1504.04909).

**Sema** is relevant as an identity layer rather than a novelty oracle. A
content hash can prove that two agents refer to the same exact definition and
version. It cannot prove that the definition is correct, useful, or absent
from the literature. That separation is exactly what a primitive ledger needs.

Primary source: [Sema repository](https://github.com/emergent-wisdom/sema).

## What counts as a candidate new primitive

Use “primitive” operationally: a typed, reusable mechanism definition that
shortens or clarifies multiple independently generated solutions and changes
their predicted intervention behavior. It need not be metaphysically
irreducible. A mechanism may be composed from known parts and still deserve a
new abstraction if its causal interface is reusable and its predictions are
not captured by the existing vocabulary.

The useful unit may be a **molecule** rather than an atom. An unordered
triplet of IDs is only a list of ingredients. A molecule records a typed causal
graph: which component observes what, which transformation acts, which state
is carried forward, and which outcome is expected. A recurring molecule can be
promoted to a macro-atom while retaining its full decomposition and
provenance. Promotion is a compression and interface decision, not a claim
that its components were unknown to science.

Reject a candidate if it is only:

- a synonym, metaphor, or renamed known atom;
- a one-off conjunction that never recurs in independently generated tasks;
- a prose summary with no intervention, assumptions, or falsifiable consequence;
- a compression artifact that saves tokens but does not improve held-out task
  solving or explanation;
- a world-specific rule that cannot be translated into a real target domain.

## Admission test for world-generated mechanisms

For each proposed primitive, retain the raw world, solver trace, compiler
translation, and source hash. Then require all of the following before adding a
vocabulary entry:

1. **Definition and identity.** Store a typed definition, causal operator,
   inputs, outputs, assumptions, disanalogies in the target domain, and a
   content hash. Keep rejected and superseded versions. Hash equality proves
   identity only; semantic equivalence still needs an independent comparison.
2. **Existing-vocabulary comparison.** Ask a blinded reviewer to construct the
   shortest expression using current atoms. The candidate must either reduce
   description/program cost at comparable meaning, or make a prediction that
   the shortest existing expression does not make. Report uncertainty rather
   than forcing a new atom.
3. **Compression and reuse.** The candidate molecule must recur in at least
   three independently seeded problems, with a held-out occurrence. Measure a
   description-length or implementation-cost reduction after charging for the
   typed graph definition itself, analogous to DreamCoder/Stitch library
   compression. Compare against the best unordered and ordered compositions
   of existing atoms; promote only when the graph interface reduces cost or
   enables a transfer that those compositions cannot express.
4. **Counterfactual separation.** Remove or replace the candidate operator in
   an executable toy model or controlled benchmark. It must change a
   pre-registered outcome in the predicted direction, while matched controls
   using the existing atoms do not. A different story with the same outcome is
   not enough.
5. **Transfer.** Apply the definition to a new problem and a new model or
   dataset. The primitive must produce a usable intervention or diagnostic
   without carrying over the invented world's impossible laws.
6. **Prior-art audit.** Search the exact mechanism, its causal operator,
   aliases, and the nearest existing atom definitions. Label the result
   `covered`, `incremental`, `plausibly_distinct`, or `insufficient_evidence`.
   Automated labels remain unadjudicated until a human checks the cited
   sources.
7. **Quality-diversity placement.** Record the candidate's causal behavior
   coordinates (what variable it changes, through which interface, on what
   timescale, and under what failure mode). Retain it only if it occupies a
   behaviorally distinct, useful niche; lexical novelty and embedding distance
   are diagnostics, not admission criteria.

The cheapest kill test is step 4 on a toy simulator. If deleting the alleged
primitive does not change the predicted outcome, or an existing-atom control
matches it, do not expand the vocabulary. This test should run before a large
literature search or weight update.

## How this changes our engine

The current pipeline should become:

`worlds / direct controls → typed causal molecule → existing-atom decomposition → macro-atom candidate ledger → executable counterfactual → prior-art review → held-out transfer → optional scorer update`

The scorer update is last. New primitives should initially be represented as
content-addressed records with no learned score. Once a candidate passes the
admission test, create training examples for its definition and behavior,
retrain on a time- and task-family-held-out split, and compare against the old
scorer. This avoids giving a fresh name an artificially high novelty score.

Promotion must preserve the graph: every macro-atom points to its constituent
atoms, edge types, source traces, failed variants, and the experiments that
justified its interface. If later evidence shows that the molecule is merely a
known pattern, demote it without destroying the history.

The first pilot should not claim that any generated atom is revolutionary. It
should estimate the fraction of candidates that are (a) covered by existing
atoms, (b) genuinely incremental, (c) unresolved because retrieval is weak,
and (d) falsified by the executable test. That is the evidence needed before
scaling the search.
