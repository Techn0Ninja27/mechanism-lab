# Corrected-source proposal baseline — 2026-11-16

Six seed triplets were attempted with Fable5.2, two concurrent calls, maximum16384 output tokens each. These are **unranked hypotheses**, not validated scorer-ranked discoveries. Source support was rebuilt using original clusters.json assignments. The 50,000-draw search is preserved, but its canonical semantic interpretation failed provenance checks; no repaired scorer is claimed.

| Candidate | Decode status | Abstract-only review | Output |
|---|---|---|---|
| 01-29-85-254 | complete | insufficient_evidence | [proposal](01-29-85-254.md) |
| 02-28-59-175 | complete | insufficient_evidence | [proposal](02-28-59-175.md) |
| 03-63-93-168 | complete | insufficient_evidence | [proposal](03-63-93-168.md) |
| 04-178-227-232 | complete | incremental | [proposal](04-178-227-232.md) |
| 05-125-194-250 | complete | insufficient_evidence | [proposal](05-125-194-250.md) |
| 06-3-63-86 | content_filter | not reviewed | — |

All literature verdicts are bounded automated triage, not human validation. A failed novelty verdict does not prove no contribution, and a positive verdict would not establish novelty. Every exported row has human_adjudicated=false and train_eligible=false. Six previous duplicate-paper controls were recognized as covered, but those trivial controls do not calibrate novel proposals. Candidate06 received a provider refusal; no fallback or retry was launched.

The publication contains proposals and audit results, not secrets, model weights, full corpus, prompts, or raw model responses. The local handoff retains the complete provenance and receipts.

## Accounting for this corrected batch

{
  "prompt_tokens": 125781,
  "completion_tokens": 60627,
  "upstream_inference_cost": 2.3655784999999994
}

Upstream costs are from persisted receipts and may use BYOK credits. They are not an account balance or total session bill. The earlier contaminated batch cost about$2.00 in saved receipts, excluding unknown aborted duplicate-call charges.

## Remaining scientific work

Rebuild a coherent assignment/name corpus and train or validate scorers against it before claiming a restored Alien Science ranking. Expand targeted retrieval around the specific proposed mechanisms, independently adjudicate overlap, and run the cheapest falsification experiments. No generator was trained on these model judgments.
