# A concrete application proposal: when should a GNN stop extrapolating?

2026-11-17. **Generated and reviewed; requires revision; no novelty or empirical
benefit established.** The updated engine completed both paid stages normally
without truncation, quote repair, or a retry. This is a full protocol-6 run,
including contribution-level assessment, not the earlier claim-only pilot.

**Follow-up:** the corrected [Cora screening experiment](../../operator-studies/gnn-sync-gate/README.md) failed: the gate had 2.086× / 2.014× predictor-only error under ordinary / LR-drop schedules. No closed-loop or hardware follow-up is justified by this configuration.

The practical question is whether remote-embedding prediction sometimes hurts
GNN training enough that choosing between prediction and the last synchronized
value improves the accuracy/communication tradeoff. Synchronization timing and
prediction budgets remain fixed. Learning which choice is safer must use only
errors revealed when a real synchronization arrives; current remote embeddings
cannot be supplied to the gate before its decision.

## Mechanism and useful test

The generated mechanism is ordinary two-expert Fixed Share/Hedge weighting at
partition-pair level. It mixes a self-fed damped trend predictor and an observed
hold cache. This is a known algorithm applied to a specific systems question.
It is not a new research primitive or novel expert algorithm. A useful result
would require an advantage over a development-tuned constant mixture and both
fixed operators, followed by a closed-loop accuracy comparison at matched costs.

The reviewer found two substantive defects: the proposal conflates predictor
state with last observed hold, and one claim requires per-offset losses that
synchronization does not reveal. The original output is preserved. A separately
corrected [screening protocol](../../operator-studies/gnn-sync-gate/protocol.json)
keeps independent states, freezes predictions before receiving observations, and
learns only from endpoint feedback. Shorter-horizon agreement is measured only
by the evaluator. The original proposal's 80% agreement assumption is an exposed
diagnostic, not a guarantee.

The screen uses real Cora training trajectories, not synthetic node labels.
It deliberately starts with a small CPU experiment rather than attempting a
SmartNIC reproduction. Its custom splits, random partitions, development seeds
and screening seeds are recorded explicitly; this differs from the model's
larger public-split/METIS-like plan. Go/no-go requires at least 5% stale-error
reduction against strong fixed controls in both ordinary and prespecified
learning-rate-drop schedules. Passing only permits further investigation;
closed-loop training and untouched graphs remain required.

## Evidence and uncertainty

The decoder received selected [SNI-GNN Section IV](https://arxiv.org/html/2608.06441v1)
(16,359 characters, equations retained, figures omitted), four provider abstracts
for SAT, Sancus-Q, FreshGNN and Sift, and bounded local retrieval. Those abstracts
are not full implementations. The selected source is a methods excerpt, not the
complete paper or author code. The initial text selector mistakenly picked the
table of contents; inspection caught this and the selector was corrected before
the paid run. `source-receipt.json` and the pack hash identify the actual input.

Local retrieval surfaced delayed-feedback expert weighting (arXiv:1902.10433).
Both model stages recognized that the proposed gate is a known expert rule.
The reviewer marked the application claim as partial overlap and expected
benefit as unknown. It found a methodological contradiction rather than merely
assigning a low novelty number. This is useful behavior in one exposed case,
not a validated accuracy or calibration result for the new scorer.

Root corrections to the review are also recorded: known-algorithm overlap is
not an algebraic no-op relative to a fixed predictor; and closed-loop errors
can in principle be audited per arm with fresh evaluation-only computations.
Neither point supplies evidence of benefit. SAT's full implementation remains
an uncovered stronger baseline. The SNI theoretical sketch was not used as a
correctness guarantee for its damped, self-fed recurrence.

## Artifacts and cost

Original inputs, raw calls and output are in local
`results/gnn-useful-proposal-v1/`. Public JSON preserves the generated proposal,
review and claim profile with exact source quotes replaced by hashes; the
redacted copy is not valid generation input. Historical `prepare.py`, manifest,
question, search configuration, source receipt, plan and call receipts are
included. Preparation requires the recorded local citation snapshot. Hosted
model outputs are not deterministic; use the recorded revision for historical
validation and a fresh output directory for a new paid execution.

Two Opus calls: **44,487 input + 6,174 output tokens**, both `finish_reason=stop`.
Conservative upstream charge **$0.301428**. Shared phase total
**$2.8172673 / $10**. Source HTML and the small Cora archive were the only new
research/data downloads; no models or large corpora were downloaded. The app
shows this proposal, its `revise` review, claim profile and separate root audit.
