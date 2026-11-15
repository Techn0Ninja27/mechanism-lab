# Lacuna: use the upstream production map

The original Alien Space of Science README explicitly identifies Lacuna as its production version. The map is built by Tiptree Systems with overlapping authors, including Alejandro H. Artiles. It already provides papers, concept elements, research directions and Alien-generated hypotheses at production scale. Mechanism Lab should use this infrastructure for research navigation rather than presenting independent large-scale ingestion as a new contribution.

Primary references:
- [Original README](https://github.com/alejandrohdez00/alien-space-of-science)
- [Lacuna MCP README](https://github.com/tiptreesystems/lacuna-research-mcp)
- [Lacuna paper](https://arxiv.org/abs/2606.26246)
- [Construction](https://lacuna.tiptreesystems.com/how-lacuna-is-made)
- [Service terms](https://tiptreesystems.com/terms)

The MCP repository is MIT-licensed adapter code; that is not a bulk license to the hosted corpus. The documented API supports bounded research lookups. We use it live for navigation and follow primary-paper links, without mirroring the service or treating generated hypotheses as facts. The separate local study corpus contains independently obtained arXiv metadata and the released research dataset, with their own provenance.

Example live lookups (no key or model API spend):

```sh
python scripts/lacuna_lookup.py search 'missingness shifts' --type paper --limit 3
python scripts/lacuna_lookup.py search 'library learning abstraction discovery' --type hypothesis --limit 3
python scripts/lacuna_lookup.py context paper art_1cabe8a62b4445a0832e66e98794aec8
```

The official MCP supports richer paper/full-text, concept, direction and linked-code views; see its README for installation. This client uses the documented API directly and does not modify the user's global Codex configuration.

Our contribution under test is the downstream loop: source-grounded atoms → explicit relationships → implementations → component ablations and held-out tests → evidence-gated reusable abstractions. We have not established that atomization creates scientifically novel mechanisms or that our loop outperforms Lacuna.
