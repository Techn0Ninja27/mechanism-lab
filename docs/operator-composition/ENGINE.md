# Typed executable operator composition

`generation.operator_composition` is a deliberately small execution engine
for testing atom/molecule mechanisms.  An `Operator` names a reversible encode
and decode callable, input and output semantic types, version, source,
provenance, assumptions, and an explicit `tested`/`untested` contract status.
Operators are installed in an `OperatorRegistry`; composition resolves only
IDs already in that registry.  There is no `eval`, dynamic import, or automatic
novelty promotion.

The built-in semantic validators are `signed_ints` (a `list[int]`, excluding
booleans), `unsigned_ints`, `bytes`, `string`, and `strings`.  A registry may
add validators.  Validation runs before execution and after every operator;
decode validates the output first and then applies codecs in reverse order.
This checks exact codec round trips, not semantic equivalence or scientific
meaning.

`enumerate_chains` searches only chains (not a general DAG), in deterministic
operator-ID order, with `max_depth` in `[1, 8]` and `max_candidates` in
`[1, 4096]`.  An optional `output_type` filters returned complete chains while
still allowing traversal through intermediate partial chains.  It can restrict
IDs with `allowed_registry_ids` and defaults to no repeated operator IDs.
Explicit repeated IDs in `compose_chain` are supported: molecule occurrences
become distinct nodes such as `shift#0` and `shift#1`, while retaining the
underlying operator content ID.  Distinct same-type operators remain distinct
because both operator and chain content IDs include registry identity, version,
source, and implementation digest.  If callable source cannot be inspected,
an explicit implementation digest is required.

`OperatorChain.to_molecule()` uses the existing `atom_record` and
`molecule_record` constructors.  It records each operator as an atom and each
adjacency as `composes`, with status `combination` and
`scientific_novelty_claim: false`.  Promotion, if ever warranted, remains an
explicit evidence-gated operation outside this engine.
