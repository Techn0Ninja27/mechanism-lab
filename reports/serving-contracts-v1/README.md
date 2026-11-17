# Source-linked contracts to molecule hypotheses

2026-11-17. This connects the existing source-contract validator to the existing
atom/molecule representation. It is a small curated integration, not bulk
atomization or an executable-compatibility proof.

Two real full-method substeps are represented in [contracts.json](contracts.json):

* Chen et al.'s cancellation update produces an intermediate certified lower
  bound on response length, from observed tokens and retained bound state.
* DMRS computes peak decode demand from known response length and prompt size,
  under its non-preemptive phase assumptions.

The exact quotes, chunk IDs and source hashes validate against the saved
five-paper serving pack. [Chen](https://arxiv.org/html/2508.14544v2) and
[DMRS](https://arxiv.org/html/2610.15579v1) remain the primary sources; the port
labels and conditions are curated interpretations, not author-provided types.
Both cards are also registered in the engine's source-contract registry and
are included only when their exact cited chunks are present.

The attempted connection emits two source-linked atom records and **no
molecule**. Its [assessment](result.json) reports lower-bound versus exact-length
representation, cancellation versus before-admission phase, missing exact-length
knowledge, a separate prompt-size input, and intermediate-value exposure.

For intuition, two requests with prompt4, length lower bound3, and actual
response6 each reserve14 units together using their bounds, but their joint
peak is20. Both individually fit a capacity14 pool. A non-preemptive consumer
cannot treat the lower bounds as exact peak demands; eviction or another rule
must handle later growth. This does not refute Chen's algorithm or the proposed
adaptation, which may explicitly supply such a rule.

`generation.contract_molecules.contract_connection` validates source evidence,
creates provenance-bound atoms, and creates a hypothesis molecule only when
all declared ports/conditions/inputs match. Such a molecule remains a
`combination`, never an automatic scientific novelty or executable-correctness
claim. Tests cover a matching declaration, bound/exact mismatch, missing
auxiliary input and invalid evidence. Lexical similarity is not used here.

Reproduce the real-source assessment with a fresh output path:

```sh
../.venv/bin/python reports/serving-contracts-v1/check.py --pack results/serving-problem-v2/pack.json --output /tmp/serving-contract-check.json
../.venv/bin/python -m unittest tests.test_contract_molecules tests.test_interface_contracts -v
```

This remains two manually curated substeps. The large corpus is not fully
atomized, and no legacy scorer vocabulary or weights were changed. The saved
abstract atoms lack these typed operational fields; automatically assigning
them without full-method evidence would overstate what the sources provide.

## Atomizer budget repair

`scripts/atomize_budgeted.py` now expects the authorized $20 shared phase cap,
while retaining its $0.50 per-run and32-paper bounds. Historical $10 plans are
not rewritten and require a new output directory. A real24-paper offline plan
was checked at `results/atomization-cap20-plan-v1/`; no extraction ran.

The old `scripts/literature_atomization.py run` direct-network path is retired
and points to the shared-ledger runner. Its offline planning, parser, validation
and historical-artifact helpers remain available. Tests verify the retired
path exits before reading inputs or accessing credentials/network. No paid API
calls were made for this work.
