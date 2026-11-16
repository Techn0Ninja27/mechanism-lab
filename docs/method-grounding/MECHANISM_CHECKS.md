# Target specialization checks

2026-11-17. Grounded research protocol 5 requires both the proposer and reviewer
to return a `mechanism_check` with source/target assumption scope, the operation
after substitution, its remaining effect, the simplest baseline, and a
distinguishing test. Status is `nontrivial`, `redundant`, or `unresolved`.

This makes an existing cancellation-review instruction inspectable. The
[full-method triage](../../reports/operational-notes-v1/full-method-triage.md)
motivated the change: a fixed-kernel residual covariance mapping eliminates the
nonlinear term of the suggested splitting method. Compatible matrix types alone
would not reveal that the extra stage contributes nothing.

The validator checks structure, allowed statuses and bounded nonempty text. It
does **not** verify algebra or scientific correctness. A claimed nontrivial
effect grants no novelty, training eligibility, empirical result, or automatic
experiment approval. An unresolved derivation is valid; the prompt explicitly
distinguishes an excluded source regime from a contradiction. Systems proposals
may use concrete algorithm steps instead of invented equations.

Both assessments are displayed separately in the webapp with a model-assessment
label. Older results show `not_recorded`, malformed display objects show `invalid`,
and neither receives a retroactive assessment. Original results are unchanged.
The new protocol and script hash invalidate previously frozen paid plans; create
a fresh directory to use the updated generator.

## Validation and remaining evaluation

- Engine suite: 252 tests run, 7 skipped, no failures. Includes unresolved-status
  acceptance, missing/empty/oversized field rejection, source-quote validation,
  frozen-plan checks, and mocked two-stage generation preserving both checks.
- Webapp suite: 36 tests passed. Includes missing historical checks and malformed
  assessment handling. JavaScript syntax checked with `node --check`.
- Live browser: ten historical proposal/reviewer sections showed `not_recorded`;
  an unresolved rendering fixture preserved the operation text and escaped HTML.
- Offline development plan: `results/mechanism-check-plan-v1`, using the existing
  interface-proposal-v2 pack. No paid calls or new scientific results.

These checks establish software behavior only. We have not measured whether
protocol 5 yields better proposals or more accurate judgments. A future quality
comparison needs fixed source packets, matched model budgets, independently
adjudicated cases (including nontrivial and uncertain cases), and checks for
false rejection as well as missed redundancy. The motivating exposed case
cannot serve as held-out evidence for that claim.
