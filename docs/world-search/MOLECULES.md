# Atoms, molecules, and promoted abstractions

The released atom vocabulary is a useful search coordinate system, but an
unordered triplet of atom IDs is not yet a scientific mechanism. A molecule in
this harness is a content-addressed directed acyclic graph: it stores the atom
definitions and revisions, typed relations such as `enables` or `inhibits`, an
operational mechanism, assumptions, a falsifiable prediction, and provenance.
Changing an edge or an atom definition changes the molecule ID. Reordering the
same graph does not.

The status field separates `known`, `combination`, and `candidate_primitive`.
The last status is a hypothesis that the composition captures a useful
distinction; it is not a novelty verdict. A composition may be valuable because
it is reusable even when every component is known.

Promotion creates a macro-atom that retains its full decomposition. It requires
positive validation, an independent held-out positive result, a component
ablation, and a prior-art review. Promotion records reusable evidence; it does
not claim that the abstraction is scientifically novel. The strict evidence
gate is intentionally stronger than a model saying that a proposal “looks
new.”

```python
from generation.molecules import atom_record, molecule_record, promote

a = atom_record("uncertainty", "estimate predictive uncertainty", [{"source": "paper-1"}])
b = atom_record("early_exit", "stop computation conditionally", [{"source": "paper-2"}])
m = molecule_record(
    [a, b], [{"source": "uncertainty", "target": "early_exit", "type": "controls"}],
    mechanism="calibrated uncertainty controls extra computation",
    assumptions=["uncertainty is calibrated on the deployment shift"],
    falsifiable_prediction="at fixed average compute, selective depth improves stressed-set accuracy",
    status="combination",
)
```

The cheapest promotion test is a matched-budget held-out experiment: compare
the molecule against the best single-atom baselines, remove each component in
turn, and test on a second task. If the joint effect disappears under ablation,
fails to transfer, or has no measurable outcome, retain it as a generated
combination rather than promoting it.
