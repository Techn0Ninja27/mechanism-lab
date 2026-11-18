# Proposal → outcome table (v1)

Date: 2026-11-18. Author: Claude. Read-only pass over stored artifacts: no
paid calls, no reruns, no edits to any result. Every row was re-checked
against the raw result files by a separate agent, and Codex reviewed the
draft; corrections from both are applied. Machine-readable copy:
[table.json](table.json).

## What this is

One row per distinct grounded proposal (deduplicated by `result.json`
SHA-256), with the labels the proposer and reviewer assigned before any
experiment, the readiness status `generation/proposal_readiness.py` derives
today, and what the linked experiment found. It answers one question: on
existing data, can the review labels or the readiness gate be validated
against outcomes?

**Answer: no.** Seven proposal operators were directly tested, including two with corrected
or newly specified protocols; none passed its relevant strong-control screen. With no positives, any rule that never says "ready" scores
perfectly, so the table cannot distinguish a good gate from a constant one.

## How rows were built

- **Execution type** separates three cases:
  - *direct*: the experiment tests the proposal's own operator, sometimes
    under a corrected or newly specified protocol (noted per row);
  - *related follow-up*: an experiment inspired by the run, not an
    implementation of the proposal;
  - *not run*.
- A link is listed only where the study README or the proposal's report names
  the other. I did not infer links from topic similarity.
- Outcomes are quoted from the study README or frozen result file.
- Timestamps are file modification times. No result file carries an internal
  timestamp, so ordering is weak evidence and a copy can change it.
- Readiness was computed by calling `evaluate_result` on the stored file.

## Table

Labels are `proposal decision / review decision / review failure_type`.

| # | Proposal (result file) | Labels before experiment | Readiness today | Execution type and study | Outcome | Later audit contradicted proposal or review? |
|---|---|---|---|---|---|---|
| 1 | Normalization axis, StableMiss (`grounded-01`) | revise / reject / algebraic_noop | incomplete (old schema) | direct: `normalization-axis` | Negative. Proposed mask feature is constant on binary columns; all-ones column gives NaN | No. The review predicted both |
| 2 | Shared-matrix polynomial vs NeuMiss (`grounded-02`) | revise / **test** / weak_control | incomplete (old schema) | direct: `neumiss-polynomial` | Negative. Loses to Richardson and Chebyshev at every depth (`results-strong-controls.json`: depth 3 validation error 0.0603 vs 0.0162 and 0.0062). Matrix sharing already published by the authors in 2021 (`reports/improvement-2026-11-16/README.md`) | Partly. The review's weak-control points were not the controls that killed it, and the specific 2021 shared-weight overlap was not identified |
| 3 | NeuMiss Jacobi-style recurrence (`grounded-03`) | revise / revise / algebraic_noop | incomplete (old schema) | direct: `jacobi-fixed-point` | Negative. Literal formula converges to the wrong fixed point (relative error 0.297–0.409) | No. Consistent with the review |
| 4 | FIER → WindowQuant 1-bit ranking (`grounded-interface-v2`) | revise / **test** / **none_found** | incomplete (old schema) | direct: `trace-window-ablation` (linked through `reports/interface-proposal`); also `window-rank-contract`, `smolvlm-trace` | Negative. Failed its frozen development gate on two images | Yes, at README level (no `factual-audit.json`): "the proposal's causal counterexample is wrong"; a review cost claim is unverified |
| 5 | FIER/WindowQuant follow-up, self-rejected (`grounded-contract-v1`) | reject / reject / weak_control | incomplete (old schema) | related follow-up: `trace-aggregation-ablation` tests the reviewer's max-aggregation suggestion, post hoc on the same exposed images | Mixed, and not a test of the proposal. One-bit max beat one-bit mean on both images (the study's one declared rule passed) but still lost to exact mass | Yes. Five findings; three contradict the review, one the proposal |
| 6 | Frequency-rank varint codec for graph walks (`grounded-systems-v1`) | investigate / **test** / weak_control | incomplete (status string `reviewed_after_manual_quote_relink`) | direct: `graph-walk-codecs` | Negative. Varint/packed byte ratio 1.25 and 1.21 with the mapping table free; 0.97 and 0.94 against FaBC, missing the 0.85 gate | Yes. Four findings in `proposal-audit.json` against the proposal. The stronger packed controls came from the model review, not the audit |
| 7 | Sync-revealed gate for stale GNN embeddings (`grounded-gnn-gate-v1`) | revise / revise / unsupported_assumption | not_recommended | direct, corrected protocol: `gnn-sync-gate` | Negative. Gate error 2.09× and 2.01× the best control; both 5% gates failed. The reviewer's named assumption was checked and was not the cause | Yes. Five findings; protocol corrected before running |
| 8 | Low-rank Newton–Muon refresh (`grounded-optimizer-v1`) | revise / no valid review (run `failed` on an invalid evidence ID) | incomplete (not reviewed) | **not run as proposed.** Prerequisite screens only: `newton-muon-preflight`, `-cost-screen`, plus baseline studies `-heldout`, `-refresh`, `-polynomial`, `-bound-diagnostic`, `-kernel-probe` | No outcome for the proposal. One proposal claim (rank 0 equals Muon) falsified in preflight. Baseline prerequisites mixed: cost screen met, held-out allocation rule failed. Candidate closed without evaluation | Yes. Five findings |
| 9 | Largest-prompt-first decode admission (`grounded-serving-v2`) | revise / **test** / weak_control | needs_revision | direct, specified variant: `serving-admission-preflight`, `serving-admission-screen` (linked through `reports/serving-problem-v2`) | Negative. Failed all six primary groups (60 of the 180 traces; no secondary group passed either) | Yes. Three findings; the paid run retrieved zero records |
| 10 | Serving portfolio, cand-00 selected (`grounded-serving-portfolio-v1`) | investigate / **test** / **none_found** | needs_revision | not run | none | Yes. Three findings; selected candidate underspecified |
| 10b | Portfolio cand-01 | revise / revise / unsupported_assumption | needs_revision | not run | none | Same audit |
| 10c | Portfolio cand-02 | reject / reject / prior_art_overlap | not_recommended | not run | none | Same audit |

Counts: 7 direct (rows 1, 2, 3, 4, 6, 7, 9), 1 related follow-up (row 5),
4 not run (rows 8, 10, 10b, 10c).

Not proposals, listed for completeness: `grounded-claim-pilot-v1` (claim-only
replay of row 6), `interface-proposal-v1/decode` and
`systems-proposal-v1/decode-v2` (failed runs with no scored proposal).

Identical copies: rows 4, 5, 7, 8, 9, 10 also exist under
`results/<name>/decode/`; row 6 under
`results/systems-proposal-v1/review-after-relink/`.

## Not in this table

- **`reconstruction-studies/`**: six Opus proposals, each implemented on one
  missingness-shift fixture; all six kept a preregistered `no_gain` verdict.
  They come from a different pipeline and I did not check whether they carry
  review labels in this schema. They are six more executed negatives.
- **`candidate-studies/`**: 21 world-search studies. Its audit lists 7 clean
  negatives and treats no `supported_pilot` label as clean evidence.
- **`integer-composition`**: enumerated search, not a generated proposal.
- **`reports/proposals-v3-2026-11-16`**: five proposals, none executed.

## Findings

1. **Zero positive outcomes in seven directly tested proposals.** The gate
   cannot be validated on existing data. This is a property of the data, not
   a verdict on the gate. Two sub-gates did pass and are disclosed above:
   row 5's one declared rule and row 8's cost screen. Neither is a positive
   result for a proposal.
2. **Reviewer `test` did not predict success.** Four of the seven directly
   tested proposals had a review decision of `test` (rows 2, 4, 6, 9). All
   four were negative. Row 4 is the cleanest case: `test` with `none_found`,
   then a failed frozen gate.
3. **The reviewer's named objection matched the experiment in four of seven
   direct rows** (1, 3, 6, 9). It was weak in row 2 (the flagged control was
   not the one that mattered) and wrong in row 7 (the named assumption was
   tested and was not the cause). Seven cases, read after the fact: a lead,
   not a rate.
4. **Every audited row was contradicted.** All five rows with a
   `factual-audit.json` (5, 7, 8, 9, 10) and row 6's `proposal-audit.json`
   contain findings against the stored proposal or review; row 4 has
   README-level corrections. The gate reads none of these.
5. **The gate's pass branch has never fired.** No stored record derives
   `test_design_candidate`. Seven of ten rows return `incomplete`: five only
   because they predate the current schema (rows 1–5), one because of its
   status string (row 6), one because the run failed (row 8). The gate says
   nothing about most of the history.
6. **No successful rejected proposal has been observed.** Rows 1, 3 and 7
   were labelled reject or revise, tested, and were negative. With no
   positives anywhere, no false-negative rate can be estimated.

## Leakage and ordering

- Rows 1–3: the study README is dated 1.6, 4.9 and 3.3 minutes after the
  result file. Row 1's `results.json` and `experiment.py` are dated about 15
  minutes after the result. Review labels precede outcomes on this evidence.
- Row 5 was generated with the row 4 negative in its context
  (`experiment_feedback` record `fier-windowquant-onebit-trace-v1`). Its
  self-rejection is not an independent prediction.
- Row 10 was generated with two feedback records in context:
  `serving-admission-preflight-v2` and `serving-largest-prompt-screen-v1`,
  the latter being the row 9 failure. Same caveat.
- Rows 7–9 carry an `experiment_feedback` block with zero records. Rows 1–4
  and 6 have no such block.
- Proposer and reviewer are recorded as the same model in the workflow notes
  (`docs/GROUNDED_RESEARCH.md`); I did not verify this per call receipt.
  Agreement between them is not independent evidence.

## Two things to check (not resolved here)

- `reports/serving-portfolio-v1/factual-audit.json` and
  `reports/serving-problem-v2/factual-audit.json` bind to the hash of the
  redacted export (`3608f21f…`, `b801100c…`), not the local original the
  readiness replay hashes (`6bd12ab8…`, `ade47a2a…`). A join of audits to
  readiness records by hash will miss these two. The other three audits bind
  to local hashes.
- Row 6's status string is `reviewed_after_manual_quote_relink`, which the
  gate treats as not reviewed. Decide whether that is intended.

## Limits

Links and outcomes rest on what each study and report says about itself. No
experiment was rerun and no outcome number was re-derived. Seven negatives
from one model, mostly single development screens on small or synthetic
data, say little about what a better evaluator would find.
