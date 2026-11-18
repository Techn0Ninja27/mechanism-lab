# Proposal-readiness reference audit

Date: 2026-11-18.

This is a development consistency audit for a proposed readiness classifier. It
replays four existing result files against their recorded structure and review
fields. It is not a blind scientific-quality benchmark, and it does not decide
whether any proposal is actually useful or novel.

## Expected disposition rules

The proposed gate should derive a machine-readable allocation status from the
saved result, while preserving the underlying model judgments:

- `incomplete`: failed/pending/claim-only records, malformed metadata, or missing
  fields needed to describe a reproducible test.
- `needs_revision`: a reviewed proposal with unresolved mechanism, weak control,
  unsupported assumption, unobservable input, insufficient evidence, or a
  reviewer decision other than a clean `test`.
- `not_recommended`: an explicit reject/insufficient-evidence decision or an
  explicitly redundant mechanism. This is a workflow disposition based on the
  recorded review, not a scientific proof of impossibility.
- `source_only_diagnostic`: structurally complete and reviewed, but no retrieved
  prior-art context is recorded; it may be inspected, not treated as ready for
  an external test.
- `test_design_candidate`: structurally complete, reviewed as `test`, with a
  nontrivial or unresolved mechanism explicitly handled, a measurable metric,
  matched control and kill threshold, and recorded retrieval context. This means
  “eligible for human test-design review,” never “run automatically.”

Coverage labels such as `covered`, `partial`, `not_found` and `unknown` describe
which supplied chunks were examined. They are not entailment, novelty or
usefulness judgments. In particular, `not_found` must not be converted into a
novelty claim, and `partial` coverage must not automatically veto a proposal.

## Replay of existing records

### Serving portfolio decode

File: `results/serving-portfolio-v1/decode/result.json`.

The result is structurally `reviewed`; the selected candidate is `cand-00`. Its
portfolio review says `test` / `none_found`, but its mechanism check remains
`unresolved`. The candidate has a baseline, operator delta, matched control,
data paths, experiment metric and kill threshold. The recorded claim coverage
is partial/unknown in places and correctly says that no-match does not establish
novelty.

Disposition: `needs_revision`, because the transition is unresolved even though
the reviewer decision is `test`. The paired candidates should remain visible:

- `cand-00`: `needs_revision` for unresolved memory/transfer/restart semantics.
- `cand-01`: `needs_revision` for an unsupported assumption and shared retention
  mechanism.
- `cand-02`: `not_recommended` because the reviewer explicitly reports prior-art
  overlap and marks the mechanism redundant.

The selected status must not be promoted to experiment-ready merely because the
portfolio selected it.

### Serving single-proposal run

File: `results/improvement-phase-2026-11-16/grounded-serving-v2/result.json`.

The result is `reviewed`; the proposal has complete structural fields, but the
review is `test` with `failure_type=weak_control` and an unresolved mechanism
check. The review and factual audit also preserve that the candidate is an
ordering change with strong prior-art overlap questions and a weak-control gate.

Disposition: `needs_revision`. This is exactly the kind of false “ready” result
the gate is meant to prevent.

### Claim-only replay

File: `results/improvement-phase-2026-11-16/grounded-claim-pilot-v1/result.json`.

The status is `claim_only_replay_not_new_proposal`; it has no ordinary reviewer
record and explicitly describes a known composition of frequency ranking and
varint coding. It should not enter a proposal test queue.

Disposition: `incomplete` / `source_only_diagnostic`, with the former preferred
because it lacks a full review and is explicitly not a new proposal. Its
`nontrivial` mechanism field is a model assertion, not a readiness override.

### Failed optimizer run

File: `results/improvement-phase-2026-11-16/grounded-optimizer-v1/result.json`.

The status is `failed`, the review is null, and no complete reviewed proposal
record exists. The proposal text contains a baseline and a test hypothesis, but
that cannot substitute for a completed review.

Disposition: `incomplete`. The gate should preserve the failure and must not
derive readiness from proposal fields alone.

## What this audit demonstrates

The current artifacts already contain enough structured signals to prevent two
common allocation errors:

1. treating JSON/schema validity or reviewer `test` as permission to run; and
2. treating no-match/partial source coverage as proof of novelty or rejection.

The gate remains a conservative routing mechanism. It does not solve novelty,
usefulness or scientific validation. Later factual audits can find issues that
are invisible to this structural replay; the gate must therefore expose its
reasons and limitations in the webapp and retain the original result unchanged.

No paid calls, proposal regeneration or experiment launches were used for this
audit.
