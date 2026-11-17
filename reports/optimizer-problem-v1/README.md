# Problem-led optimizer proposal: low-rank Newton–Muon preconditioning

2026-11-17. **Proposal generated; model review failed evidence-ID validation;
mathematical corrections required before profiling.** No optimizer-training
benefit, speedup, memory saving or novelty is established.

Unlike the earlier GNN gate run, the question did not prescribe an intervention.
It asked the engine to identify a documented optimizer bottleneck, select a
minimal change, account for strong competitors and state a cheap stopping rule.
The engine selected the cost of Newton–Muon's activation-covariance
preconditioner and proposed a rank-r eigenbasis plus isotropic remainder.

The proposed target is at least 3% lower synchronized time to a fixed held-out
language-model loss versus full-inverse Newton–Muon and simpler controls. The
proposal recommends revision: at tiny scale, the overhead may be too small to
justify the extra approximation. Its early test requires overhead above 5% and
adequate operator fidelity before a real-text training sweep. These are proposed
criteria, not measured thresholds supported by this run.

A subsequent [real-text MPS baseline cost screen](../../operator-studies/newton-muon-cost-screen/README.md) measured about 5–9% higher per-step time for full Cholesky Newton–Muon versus Muon at widths 256/512. The necessary cost screen survived; low-rank benefit and time-to-loss remain untested.

## Evidence and closest existing methods

Seven HTML snapshots totaled 4,868,528 bytes. The bounded pack supplied 55,198
characters in 21 chunks: selected Newton–Muon methods/experiments/appendices,
finite-Newton–Schulz setting and limitations, and five primary abstracts.
Ten local retrieval records added further competing optimizers. Full paper/code
coverage remains incomplete. Two initial section selections hit table-of-contents
entries; inspection corrected them before paid execution. Exact input hashes
and final selections are preserved.

- [Newton–Muon](https://arxiv.org/html/2604.01472v1) already uses damping, moving
  averages, periodic refresh and block structure. Appendix B includes polynomial
  inversion; Appendix F reports a weak diagonal update-statistics variant. A
  source-named low-rank direction is not proof that no one has implemented it.
- [Finite Newton–Schulz analysis](https://arxiv.org/html/2608.26288v1) warns that
  approximation and smoothing are not interchangeable with an exact polar map.
  Its theoretical assumptions differ from deployed Muon normalization and
  coefficients; its guarantees cannot simply be transferred to this proposal.
- [Muon²](https://arxiv.org/abs/2604.09967),
  [spectrum-adaptive NS](https://arxiv.org/abs/2609.33047),
  [Polar Switch](https://arxiv.org/abs/2610.26742),
  [CacheMuon](https://arxiv.org/abs/2606.16371) and
  [HiMuon](https://arxiv.org/abs/2606.27216) cover nearby second-moment,
  polynomial, temporal-reuse and tiling approaches. Their abstract-level presence
  prevents claiming these broad mechanisms as new but does not supply complete
  runnable baselines.

A native source review also considered Huawei P/D-Serve and HeteGen. We did not
choose a synthetic-only serving simulator or transfer-cost experiment whose
premise changes on unified-memory hardware. The optimizer question admits a
real local training test, if its preliminary cost/definition checks survive.

## Failure and independent audit

Both Opus calls finished normally with `finish_reason=stop`. The review included
`2610.24839 (prior_art:b80616e44f0e58922f92)` as a checked ID, which is not an exact
supplied chunk ID. Its embedded hash refers to a statistical-whitening record;
the prefixed identifier is not an established attribution. We did not silently
rewrite it or retry. The original failed result and raw response are preserved.
The public `unvalidated-review.json` keeps this distinction explicit. All nine
quoted evidence spans matched their supplied chunks, but that does not repair
the invalid checked ID or establish correct reasoning.

The model reviewer suggested a profiling test but missed two algebraic details
that an independent native agent and the root identified:

1. For `r=n`, the expression defining the residual mean divides by zero. The
   implementation needs a separate full-inverse branch.
2. For `r=0`, the preconditioner is scalar at one step, but it is applied before
   momentum. Changing that scalar across refreshes changes the momentum history;
   it is not generally equivalent to ordinary Muon over training.

Other required corrections: the proposal's `at most 3% gain` claim conflicts
with its `at least 3%` target; full `eigh` is still cubic; iterative eigenspace
estimation costs dense matrix products; storing/updating full K remains
quadratic. A low-rank inverse representation alone does not remove that memory.
Use the source's polynomial inverse and larger refresh intervals as controls.
The [algebra preflight](../../operator-studies/newton-muon-preflight/README.md)
tests the identities only; it cannot validate usefulness.

The first polynomial control is recorded in the [matrix-fidelity study](../../operator-studies/newton-muon-polynomial/README.md).
On captured real training covariances, 18/56 blocks had an eligible published
plan under a conservative spectral bound, and every covered block met the 1%
inverse-residual check. The other 38 blocks remain uncovered by that policy;
this is a coverage limitation, not evidence that the published method fails.
The check is untimed and does not establish optimizer quality, time-to-loss,
memory savings, or novelty. The [evidence ledger](../../docs/method-grounding/experiment-feedback.json)
keeps the result hash-bound and explicitly ineligible for training claims.

## Reproduction and cost

Local `results/optimizer-problem-v1/` retains HTML, selected text, source pack,
retrieval, original calls and failed result. Public artifacts preserve hashes,
method descriptions and call receipts; exact source quotes are hashed in JSON.
Historical `prepare.py` and `select.py` reconstruct the selection from the saved
HTML. Re-fetching unversioned provider content may change bytes. The paid plan
requires its recorded code/dependency hashes; later validator/budget changes
intentionally invalidate it. The current $20 shared cap does not rewrite this
historical $10 plan.

Two calls used **82,535 input + 6,564 output tokens**, costing **$0.46142** under
conservative upstream accounting. Phase spending reached **$3.2803365**. The
user then explicitly increased the total cap from $10 to **$20**, with a locked
migration preserving all 36 reconciled reservations and receipts. No calls were
pending or unknown. The per-run $2 cap remains unchanged.

The webapp shows the original failed run, its proposal and a separate factual
audit. It does not present the unvalidated model review as an accepted review.
The next step is a corrected, cost-accounted baseline profile, not an assertion
that the engine has discovered a new optimizer.

## Additional controls, 2026-11-17

The [matched refresh control](../../operator-studies/newton-muon-refresh/README.md)
found refresh64 versus32 was 0.24% costlier at width256 and 1.57% cheaper at
width512. This short, fixed-order screen provides no consistent improvement
or quality claim. The [spectral-bound diagnostic](../../operator-studies/newton-muon-bound-diagnostic/README.md)
found cheap alternative bounds add no polynomial-plan coverage; a CPU
eigensolver makes 28/56 blocks eligible, with no runtime claim.

These results do not justify promoting the generated low-rank proposal. Before
more approximation engineering, the unresolved decision is whether full
Newton-Muon improves held-out loss per wall-clock time over Muon on this
workload at all. A matched held-out comparison can falsify this local use case;
if it fails, close this candidate here and use the recorded evidence in the
next problem-led generation rather than optimizing an unhelpful baseline.
