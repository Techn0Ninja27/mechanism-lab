# Prior-art audit: FIER → WindowQuant window ranking

Audited 2026-11-16 from the primary arXiv HTML versions. The proposed transfer
is: replace WindowQuant's encoder-space window score with per-layer prompt-query
and key scores, while keeping each clip's WindowQuant FP16/INT4/INT2 counts and
first-window FP16 rule fixed. The intended test compares WindowQuant's score,
exact attention-derived scores, full-precision logits, 1-bit and 2-bit key
scores, and random/position controls on held-out VLM traces.

The audit does not establish novelty. Four papers already cover the broad design
space of query-aware, attention-aware or importance-aware KV precision. The
remaining distinction is narrower: per-layer *window* ranking at fixed counts in
a VLM, with the baseline allocation counts frozen before the alternative score
is applied. I found no evidence here that this exact ablation has been tested;
that is a search result, not a novelty claim.

## What overlaps

**MixKVQ** ([arXiv:2512.19206](https://arxiv.org/abs/2512.19206)) is the closest
mechanistic overlap. It defines key-channel salience as query magnitude times
quantization sensitivity, assigns BF16/UINT4/UINT2 by thresholds, updates query
statistics periodically, and keeps values at per-token 2-bit precision. Its
objective is explicitly attention-score error, not cache reconstruction alone.
The proposed window score is therefore not a new general principle of using
queries to allocate precision. It differs in granularity (windows versus key
channels), modality (VLM visual windows versus text LLM cache), and fixed-count
ranking protocol.

**ReadKV** ([arXiv:2610.11245](https://arxiv.org/abs/2610.11245)) makes query
adaptivity a cache-access protocol: a current query chooses key prefixes, then
the resulting attention chooses value prefixes. The stored progressive code is
unchanged, and it evaluates separate retained-bit and fetched-bit budgets.
This directly weakens any claim that query-aware precision decisions are absent
from KV-cache research. It is not an exact duplicate: the proposed test assigns
resident precision once at prefill and does not read progressively at each
decode query.

**Attention-Aware Transform Coding (AATC)** ([arXiv:2608.14191](https://arxiv.org/abs/2608.14191)) derives attention-aware key/value distortion and allocates bits with a calibration set, whitening/SVD transforms, and reverse water-filling. Its allocation is per transformed channel and token, not VLM windows, and is calibration-based rather than a prompt-conditioned ranking. It nevertheless covers the stronger idea of allocating KV bits using attention-output consequences.

**MiKV** ([arXiv:2402.18096](https://arxiv.org/abs/2402.18096)) preserves important cache pairs in higher precision and retains evicted pairs at low precision. Its outlier-aware prefill balancer uses both query and key channel maxima before low-bit quantization. It is token/pair-oriented rather than window/layer-oriented, but it covers importance-aware mixed precision and query/key-aware quantization.

**NOVA-KV** ([arXiv:2608.04074](https://arxiv.org/abs/2608.04074)) is another
close 2026 source. Its full method derives key and value transforms from
attention-product distortion, then uses fixed-width vector quantization and
equal-volume grouping. The key objective is explicitly attention-product error
(Sections 3-4, Eqs. 3-5), and fixed-rate grouping is designed for serving
(Theorem 2 and Section 5). It does not rank VLM visual windows or preserve
WindowQuant's per-layer count vector, but it further closes the space around
attention-aware bit allocation. Include NOVA-KV as a representation/quantizer
baseline if the work proceeds.

WindowQuant itself ([arXiv:2605.02262](https://arxiv.org/abs/2605.02262)) already uses visual-window/text-prompt similarity to assign FP16, INT4 and INT2 and reorders windows by bit-width. FIER ([arXiv:2508.08256](https://arxiv.org/abs/2508.08256)) supplies query-dependent key scoring for retrieval, but its retrieval-time and quantization-time objectives differ. Transferring FIER's signal is therefore a testable engineering ablation, not a new category.

The method pages were checked at exact HTML records: [MixKVQ](https://arxiv.org/html/2512.19206v1)
Sections 3.2, 4.1 and 4.2 with Eqs. 4-8; [ReadKV](https://arxiv.org/html/2610.11245v1)
Sections 3-3.2 and Appendix B.2-B.3; [AATC](https://arxiv.org/html/2608.14191v1)
Sections II-B, IV and V-A with Eqs. 6-7; [MiKV](https://arxiv.org/html/2402.18096v1)
Sections 3.1-3.2 with Eqs. 1-4; and [NOVA-KV](https://arxiv.org/html/2608.04074v1)
Sections 3-4 with Eqs. 3-5 plus Theorem 2. [WindowQuant](https://arxiv.org/html/2605.02262v1)
Sections 3.1-3.3 and Eqs. 8-11 provide the baseline, while [FIER](https://arxiv.org/html/2508.08256v2)
provides the adjacent retrieval method. These exact scopes are also recorded in
the machine-readable audit.

## Exact versus partial overlap

| Claim | Evidence status | Reason |
|---|---|---|
| Query information can guide KV precision | Already covered | MixKVQ, ReadKV and MiKV explicitly use query/attention information. |
| Attention-aware distortion can guide bit allocation | Already covered | AATC derives and optimizes an attention-aware distortion objective. |
| Fixed-rate attention-product-aware vector quantization | Already covered | NOVA-KV derives attention-preserving transforms and fixed-width grouped VQ. |
| A VLM's visual windows can be ranked per layer by prompt-attention evidence while preserving WindowQuant's baseline counts | Narrow, unverified | The retrieved primary descriptions cover channels, tokens, progressive reads or calibration; they do not establish this exact fixed-count VLM window ablation. |
| 1-bit quantized keys are a useful prefill scoring signal | Unverified and likely weak | The proposal must compare against exact logits and attention mass; quantization may only add noise because prefill already has full-precision keys. |
| The transfer is novel | Not established | No exhaustive search or implementation comparison was performed. |

## Investment gate

Do not invest in a broad “query-aware mixed-precision KV” claim. That premise is
well populated and would require a much stronger contribution. A small trace
ablation is still justified if it can answer the narrower question cheaply:

1. Freeze WindowQuant's per-clip, per-layer precision counts from its own score;
   do not let the candidate score change the byte budget.
2. Compare Eq. 8, exact prompt softmax mass, exact mean logits, 1-bit and 2-bit
   grouped-key scores, and random/position controls.
3. Include prompt-before-video and prompt-after-video causal layouts, decode-step
   stratification after the first token, and both QA and caption tasks.
4. Count packed payload, scale/zero metadata, permutation/repacking, score
   buffers and peak prefill memory separately. Equal nominal bits are not equal
   system cost.
5. Require an untouched VLM/model or video-source transfer set, task quality
   against FP16 and WindowQuant, and measured prefill/repacking/decode latency.

Kill the candidate if it does not beat WindowQuant at equal bytes, if gains vanish
after the exact-attention controls, or if scoring/repacking overhead erases the
memory benefit. If it survives, the defensible contribution would be a narrow
systems result about when prefill attention-derived **window** ranking helps a
VLM's fixed mixed-precision allocation—not a new KV-quantization principle.

The local proposal and review are at
`reports/interface-proposal/proposal-and-review.json`. The review itself calls
for roughly 50 fresh clips and the controls above. This note expands its prior
art after decoding; it does not modify that completed paid result.
