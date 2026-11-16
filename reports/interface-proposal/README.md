# FIER → WindowQuant: a graph candidate under review

2026-11-16. Status: narrow ablation candidate; novelty and usefulness unestablished.
The completed Opus proposal/review is saved in `proposal-and-review.json`. These
are model judgments, not human adjudication or experimental findings.

Selection: the highest-ranked method-to-method pair from the bounded interface
graph. The highest overall pair instead confused model-weight quantization with
cache-key quantization. This exposes a lexical ranking failure, not typed
compatibility. The selected pair is FIER (systems retrieval label) → WindowQuant
(ML retrieval label); both concern ML inference. Corpus labels are not adjudicated
discipline boundaries.

The proposal replaces WindowQuant's encoder-space window relevance ranking with
per-layer prompt-query/key scores. It preserves each clip and layer's baseline
FP16/INT4/INT2 window counts and first-window FP16 rule. The 1-bit scoring arm
must compete against full-precision logits, softmax attention mass, 2-bit scoring,
and position/random controls. Metadata, transient buffers, and allocation costs
must be counted separately from packed KV payload.

## What the review changes

- Mean logits and mean softmax attention mass are different rankings. The full
  precision logit arm is a comparator, not a mathematical upper bound on utility.
- The proposal's causal counterexample is wrong: full prefill produces prompt
  queries before the described post-prefill quantization. Prompt order changes
  causal relevance, not merely whether the query can exist.
- Exact attention is computed during prefill, but the review's claim that window
  mass can be collected at essentially no cost is unverified. Fused attention
  kernels need not materialize the full matrix; kernel modification, reduction,
  and memory costs must be measured. Do not treat this claim as an established
  reason to reject the low-bit arm.
- The pinned FIER file visibly allocates `past_key` with `torch.empty` and passes
  it to 2-bit quantization without a visible fill in that branch. This is a local
  code discrepancy, not a replication or a conclusion about all FIER results.
- The existing generic method-lineage registry covers two missing-data families,
  not KV quantization. Its lack of a match is no evidence of novelty.

## Additional prior art found after decoding

These primary abstract checks were not supplied to the completed decoder. They
are a required expansion of its baseline coverage before investment:

- [MiKV (2024)](https://arxiv.org/abs/2402.18096): preserve important KV pairs at
  higher precision while retaining less important pairs at lower precision.
- [MixKVQ (2025)](https://arxiv.org/abs/2512.19206): query-aware key-channel
  precision selection combines quantization difficulty with query relevance.
- [Attention-Aware Transform Coding (2026)](https://arxiv.org/abs/2608.14191):
  allocate bits using attention-output distortion under a quantization model.
- [ReadKV (2026)](https://arxiv.org/abs/2610.11245): progressive stored codes allow
  per-query precision selection with separate retained-bit and fetched-bit budgets.

These overlaps invalidate a broad novelty claim for query-aware mixed precision.
They do not by themselves settle the narrow window-ranking ablation. Full methods
and implementation comparisons remain required. The local ten-result retrieval
included WindowQuant, FIER, KVTuner, KIVI and CoMixKV, but omitted the four works
above; corpus size alone does not guarantee adequate prior-art recall. An exact-title
SQLite audit found MiKV present but omitted by retrieval; MixKVQ, ReadKV and
Attention-Aware Transform Coding were absent from this active snapshot. Those
three therefore require corpus additions, while MiKV requires retrieval coverage
improvement. This audit does not claim exhaustive alias matching.

## Execution record and next test

The initial partial-source run stopped at exact-quotation validation (one altered
quote); it was not output truncation. Its OpenRouter usage cost was $0.19703772 ($0.199028 charged to the
conservative shared budget using upstream inference cost). A fresh
run added WindowQuant method pages 7–12, generated and reviewed one proposal, and
both calls passed validation with `finish_reason=stop`. All three calls remain
accounted for. Total phase budget-accounted spend is $1.6850121 against the $10 cap; see receipts.
These three calls reported $0.70651548 in usage cost and $0.713652 in upstream
inference cost; the ledger uses the latter, including any BYOK cost.

The CPU synthetic harness in `operator-studies/window-rank-contract` checks count,
byte-accounting and permutation contracts. Its corrected v3 result reproduces
exactly in an independent root run: three seeds, maximum correctly permuted
attention error 1.11e-16; failing to permute a nonuniform mask changes outputs
by at least 0.105 in these cases. Initial quantizer/accounting bugs are preserved
as explicitly invalid diagnostics. It cannot demonstrate VLM usefulness.
The next empirical gate requires real prompt/decode traces, excluding the first
decode step from the main comparison, frozen development choices, and untouched
transfer clips/model. Task quality and measured GPU latency are later gates.
No model weights were downloaded and no GPU benchmark has been run for this proposal.

Follow-up: the [primary-source corpus repair](../corpus-ml-grounded-v4/README.md)
adds the three missing works. A [full-method audit](../../docs/method-grounding/kv-window-ranking-prior-art.md)
now distinguishes overlapping mechanisms from the narrow proposed ablation.
The original paid outputs above remain unchanged.

Real-model follow-up: [SmolVLM MPS attention capture](../../operator-studies/smolvlm-trace/README.md)
now supplies Q/K/V and attention tensors from two exposed development images.
Independent NumPy replay verifies extraction consistency. This is not yet an
evaluation of the proposed precision-ranking intervention.

Completed intervention: the [fixed-budget trace ablation](../../operator-studies/trace-window-ablation/README.md)
found no support for the 1-bit ranking against stronger controls on the two
exposed images. Its scoped negative result is now available to future grounded
proposal runs through the local experiment-feedback registry. The original
proposal/reviewer outputs remain immutable, and the entire source pair is not
rejected by this narrow result.
