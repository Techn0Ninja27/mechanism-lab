# Graph-guided search: primary-source notes — 2026-11-16

[HyGRAIL](https://arxiv.org/abs/2609.02056) combines heterogeneous graph-model
triage with LLM review of missing typed links. Review receives node associations
and multi-hop paths. Its abstract reports that compact evidence from both ends
of a candidate link is useful; the reported experiment is on MatKG. This is
precedent for evidence-guided link proposals, not proof that our co-occurrence
pairs are novel mechanisms.

[Graph-PRefLexOR](https://arxiv.org/abs/2607.00924) trains explicit graph-based
reasoning phases using GRPO. Its reported graph expansion increases long-range
recombination within a bounded semantic space. This supports testing graph
operations while distinguishing recombination from introducing a new primitive.

[Temporal literature-based discovery](https://arxiv.org/abs/1907.09395) uses
keyword co-occurrence networks and temporal features to predict emerging links.
Temporal evaluation is a useful next comparison, but the current catalog was
constructed from the full corpus. Our random grouped split cannot support a
historical discovery claim, even if paper dates are available.

The present baseline tests only undirected co-occurrence structure. A mechanism
search should retain this as one layer and introduce a separate typed layer:
inputs, outputs, assumptions, operators, and measured effects. Candidate edits
could replace an interface-compatible operator or transfer a small subgraph to
a different task. Such an edit needs source evidence and a falsifying test;
closeness in vector space is insufficient.

A proposed new node can represent a reusable multi-edge mechanism. Naming that
node does not establish a new scientific primitive. Its value should be tested
by reuse and component-removal experiments, preserving the existing molecule
evidence requirements.
