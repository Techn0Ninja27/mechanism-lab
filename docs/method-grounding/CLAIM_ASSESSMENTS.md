# Contribution-level evidence assessment

2026-11-17. Grounded research protocol 6 adds a bounded claim/evidence layer,
informed by the [framework audit](../ranking/FRAMEWORK_REEVALUATION_2026-11-17.md).
This is an original small contract/validation implementation, not copied external
framework code or a calibrated novelty scorer.

Proposals now declare one to six contribution claims with stable IDs, a central
or supporting role, and a kind: problem, method, resource, or expected result.
At least one is central. Reviews assess every ID exactly once, naming examined
source chunks and reporting `covered`, `partial`, `not_found`, or `unknown`.
Overlap requires exact quoted evidence. Missing chunks cannot establish overlap
or a completed search. No-match is only relative to the examined evidence.

The deterministic summary distinguishes unresolved central evidence, reported
central overlap, no match in examined evidence, and mixed/partial overlap. Known
supporting ingredients do not automatically disqualify a distinct central claim.
No status sets a numeric novelty estimate, establishes novelty, promotes training
eligibility, or automatically rejects a proposal. Declared roles and coverage
remain model judgments: a generator can misidentify what is central, and a valid
quote need not entail its asserted interpretation. Expected outcomes remain
hypotheses even when similar results exist in prior work.

The webapp shows the profile separately from the overall reviewer decision and
any executed experiment. Missing historical profiles show `not_recorded`;
malformed profiles show `invalid`. Original results remain unchanged. Protocol 6
and dependency hashes invalidate earlier paid plans; use fresh output directories.
Historical runs require their recorded code revision, not the latest validator.

## Validation

Four module tests cover empty evidence, no-match abstention, peripheral overlap,
exact attribution, duplicate/missing claims and preservation of result-hypothesis
status. Mocked full generation/review tests check the protocol integration. The
engine suite ran 263 tests with 7 skips and no failures; 38 webapp tests passed.
These are software-behavior checks, not evidence of better scientific ranking.

The [single-proposal pilot](../../reports/claim-assessment-v1/README.md) exercises
the paid assessment path, preserves an ambiguous application-level label, and
keeps an independently measured negative experiment separate. It is not a
held-out evaluation or an end-to-end paid protocol-6 generation run.
