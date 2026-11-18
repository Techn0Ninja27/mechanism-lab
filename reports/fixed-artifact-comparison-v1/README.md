# Fixed-artifact comparison: offline implementation

2026-11-19. This comparison has not generated candidates or run training.
Campaign 2 remains stopped. The new question is whether its unchanged atom
artifact helps two decoders produce useful programs, compared with raw sources,
one prose artifact, and task-only prompts. The prospective design is in
[protocol.md](protocol.md).

Implemented: an offline plan builder with input and prompt hashes, a seeded
48-call schedule, conservative standard-price reservations, a descriptive
fidelity table, and per-decoder decision rules. The planner itself has no paid execution authority; the separate audited
lifecycle is described in [execution-status.md](execution-status.md). Missing or invalid prose is preserved as unavailable; the protocol
stops before generation in that case.

[atoms-fidelity.json](atoms-fidelity.json) reproduces 23 valid reviews out of 24,
six fields flagged by either reviewer and two by both. Unknown-information fields
are reported separately. These are reviewer measurements, not truth labels. A compound prose summary and
a short typed field are different denominators: do not compare field-flag rates
across representations as quality estimates.
[The separate model adjudication](../generation-comparison-campaign2/adjudication-summary.json)
classifies the flags without changing the failed run.

[offline-check.json](offline-check.json) records the conservative reservation
including a 5% BYOK-fee contingency: $24.8005002 new, $29.9422518 phase total. No API request
was made and the live ledger was byte-identical. [provider-check.md](provider-check.md) documents public prices, advertised model
settings, and fee assumptions. The lifecycle and HTTP adapter are implemented. An independently audited
execution freeze and the actual run are still pending; account routing is not
claimed from metadata alone.

The [request client](client-status.md) is integrated with the lifecycle and an
explicit HTTP adapter. Tests inject fake providers and never read real credentials.

Validation (97 tests):

```sh
../.venv/bin/python -m unittest tests.test_fixed_artifact_client tests.test_fixed_artifact_runner tests.test_fixed_artifact_fidelity tests.test_fixed_artifact_decisions tests.test_research_ledger tests.test_fixed_artifact_transport tests.test_fixed_artifact_analysis tests.test_fixed_artifact_execution tests.test_fixed_artifact_review_launch tests.test_fixed_artifact_command
```

Recompute the fidelity table without altering any input:

```sh
../.venv/bin/python -m generation.fixed_artifact_fidelity results/generation-comparison-campaign2-run/prep-draws/atoms/00/reference-slots
```

No usefulness, novelty, or representation advantage has been established by
this preparation work.
