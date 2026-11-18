# Audit of the completed generation stage (fixed-artifact-comparison-v1)

Date: 2026-11-19. Author: Claude, from an independent agent's read-only audit.
No network, no evaluation; development and final evaluation outputs were not
opened. Live ledger hash identical at start and end.

**No discrepancies with Codex's reported outcome.** Every count, total and
hash recomputed matches. Two facts in the outputs matter for interpretation.

## Receipts and ledger

- 49 receipts: 48 generation plus the preparation call; names equal the
  frozen schedule.
- All 48 requests equal the frozen requests (request and prompt hashes,
  model, 12,288 token cap, low effort, no temperature).
- Each decoder answered its own 24 calls. Provider and finish reason are the
  same on all 48: `stop`. All 48 ran on the user's own key.
- Upstream cost equals tokens × frozen price on all 48. Generation
  $7.043956; with the preparation call $7.149892, as reported.
- All 49 receipt checksums match their ledger anchors. Campaign charge
  $23.2796550 at full reserves; phase charge $28.4214066 of $30.

| Decoder | Completion tokens | Reasoning share | Largest call (of 12,288) | Upstream USD |
|---|---:|---:|---:|---:|
| Opus 5.5 | 51,573 | 24% | 3,688 | 1.56 |
| Fable 5.2 | 110,392 | 56% | 7,678 | 5.48 |

Nothing was truncated. Opus used no reasoning tokens on its six task-only
calls.

## Validity

Re-parsing all 48 raw responses with the committed parser and compiler
reproduces every one of the 192 inventory rows.

| Decoder | task-only | raw | atoms | prose | Total valid |
|---|---:|---:|---:|---:|---:|
| Opus 5.5 | 7 of 24 | 12 of 24 | 14 of 24 | 18 of 24 | 51 of 96 |
| Fable 5.2 | 23 of 24 | 19 of 24 | 23 of 24 | 23 of 24 | 88 of 96 |

| Recorded error | Slots | Notes |
|---|---:|---|
| "malformed or too-deep AST" | 23 | depth is the only violation in all 23 (21 Opus, 2 Fable) |
| "constant is outside [-4, 4]" | 18 | constants 6, 8, 16, mostly as clip bounds (12 Opus, 6 Fable) |
| Not JSON (whole response) | 12 | three Opus task-only calls, each with one extra closing brace |

Every recorded error is a true property of the emitted text. No response was
repaired, leniently parsed or re-requested. No valid program violates the
stated grammar.

## Two facts for interpretation

**1. The depth rule is stricter than one natural reading of the prompt.** The
prompt says "at most 32 nodes and depth 6". The compiler counts the root as
depth 1, so six levels of nodes is the limit. Nineteen of the 23 too-deep
programs are exactly seven levels, which is six edges, with no other
violation: 17 Opus, 2 Fable. Read as six edges they would be valid, and one
all-invalid Opus call would have a valid program. 94 of the 139 valid
programs sit exactly at six levels. This choice was frozen before the calls;
it is a stated limitation, not something to revisit.

**2. Four Opus calls have no valid program:** three task-only (the malformed
responses) and one raw. Under the frozen decision table a call with no
eligible candidate makes that decoder's outcome incomplete, and task-only is
the arm every attribution rule depends on.

| Valid programs in the call | 0 | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|---:|
| Opus calls | 4 | 3 | 6 | 8 | 3 |
| Fable calls | 0 | 0 | 2 | 4 | 18 |

Opus picks are best-of-fewer and not comparable in selection strength to
Fable's. The validity gap is mostly format compliance at the depth limit and
constant range. It says nothing about which decoder proposes better
estimators.

## Distinct programs

- Opus: 51 valid, 48 distinct. Fable: 88 valid, 71 distinct. 115 distinct
  overall; 4 shared between decoders.
- No duplicates within a call. The most repeated program appears in 7 Fable
  slots across all four arms.
- No valid program equals hold, the frozen comparator, or any fixed or random
  control, by hash or by output on random synthetic states.
- 135 of 139 valid programs mention a mean or average: the candidates are
  mostly variations on the comparator's ingredients, which the task text
  names.

## Exports and steering

Both 124-row exports are complete, with invalid slots carried as failures. No
response cites an accuracy figure or mentions fidelity, reviewers or the
adjudication.
