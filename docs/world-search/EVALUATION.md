# World-search evaluation contract

The pilot compares direct generation, random semantic seeds, and counterfactual
world generation on two problems. Every output must be converted to the same
mechanism record before judging:

`title`, `intervention`, `mechanism`, `assumptions`, `baseline`, `metric`,
`falsification`, `implementation`, and `queries`.

The pilot may emit `assumptions` as one string; the evaluator normalizes it to
a one-item list. `queries` remains a list of two targeted search strings.

`blind_cases_with_mapping(records, seed)` validates this schema, shuffles with
a recorded seed, and removes `arm`, source IDs, model names, world seeds, and
generation metadata. Keep the source-to-blind mapping outside the judge input.
The judge
must see the same problem statement and the same output format for all arms.

The comparison is descriptive unless an independent evaluator confirms the
mechanism. `summarize` reports invalid/missing cases and exact normalized
pooled and pairwise cross-arm mechanism duplicates; titles and implementation
prose are not part of that signature. These numbers do not measure semantic novelty,
external literature novelty, usefulness, or feasibility.

Primitive triage keeps three decisions separate: whether the current atom
representation has a gap, what bounded literature review supports, and what
an empirical test has shown. A proposal may be marked `propose_for_review`,
but there is no automatic new-atom admission. A new vocabulary item requires
an exact source record, a human decision, and a reproducible empirical test.

Prior-art review is a second stage. Retrieve evidence independently of the
generation arm, require source IDs and verbatim abstract spans, and keep every
automated label `human_adjudicated=false` and `train_eligible=false`. A model
must not be allowed to certify its own novelty. Human reviewers should inspect
the proposal, strongest matches, causal difference, and cheapest falsification
test blind to arm.

The minimum 24-case design is two problems × four rows × three
arms. Use a fresh builder that never sees the problem, a solver that sees the
problem and world rules but not the seed or study purpose, and a compiler that
sees only the in-world solution and emits the common mechanism schema. Direct
cases currently do not consume their nominal `seed_word`, so they are
stochastic replicate controls unless an API seed is recorded; random-seed and
world cases receive the word. Record actual input/output tokens, reserved cost,
and failures; do not call model-rated originality a
ground-truth metric. The effective unit is the problem–seed branch, not each
sentence or each judge call.

Pre-register: (1) schema-completion rate; (2) exact duplicate rate; (3) blind
human-rated mechanism distinctness; (4) prior-art overlap and evidence quality;
and (5) pass rate on the proposed falsification test. Report all failures and
model refusals. Do not expand the Alien atom vocabulary from a proposal until
its mechanism has an exact source record, a clear difference from retrieved
work, and an independently reproducible test.

Run offline checks with:

```sh
python -m unittest tests/test_world_evaluation.py
```

