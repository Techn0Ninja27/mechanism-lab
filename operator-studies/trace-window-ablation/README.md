# Fixed-budget ranking on real SmolVLM attention traces

2026-11-16. The graph-generated FIER → WindowQuant candidate's 1-bit ranking
failed its frozen development gate. It has no demonstrated fidelity advantage
over exact prompt-attention ranking in this two-image pilot. This is a local
attention intervention, not an end-to-end VLM benchmark or a novelty result.

## Result

Mean relative L2 attention-output error, aggregated over 30 layers and decoder
steps 1–3 for each image; lower is better:

| Ranking / reference | Candy | Statue |
|---|---:|---:|
| Aligned-input cosine proxy | 0.03285 | 0.01848 |
| Exact prompt softmax mass | **0.01510** | 0.01510 |
| Exact mean QK logit | 0.01972 | 0.02173 |
| 1-bit mean QK logit | 0.02518 | 0.02229 |
| 2-bit mean QK logit | 0.01804 | 0.02195 |
| Recency | 0.01630 | 0.01890 |
| Position | 0.03713 | 0.01918 |
| Fixed interleaved pattern | 0.02709 | **0.01462** |
| All visual KV at FP16, larger budget | 0.0000766 | 0.0000591 |

Three seeded random controls are also in `summary.json`. Exact attention mass
beats the embedding proxy on both images, but a fixed pattern does better on
the statue. The 1-bit arm is worse than exact mass on both; it also loses to
the embedding proxy on the statue. The observed result does not depend on the
gate's particular 5% margin. No inference statistics are warranted: these are
two exposed images, and layer/step measurements are correlated.

## Frozen method

`protocol.json` was written before the intervention scores were computed.
`freeze.json` binds the code, protocol and trace manifest. The source traces
come from the [verified SmolVLM capture](../smolvlm-trace/README.md).
Each image has 64 visual tokens, grouped into eight contiguous chunks of eight.
Every ranking receives exactly two FP16, three INT4 and three INT2 chunks, with
the first chunk fixed at FP16. The allocation is shared across KV heads in each
layer. It is an arbitrary frozen budget, not calibrated WindowQuant thresholds.

Rankings use only prefill evidence. Prompt queries are the capture's non-special
text positions after the visual span, including any ordinary template text.
Exact mass is normalized against all keys; mean raw logits are a distinct
observable. The embedding comparator uses aligned LLM inputs and is explicitly
a proxy. One-/two-bit scoring uses an affine key surrogate with 32-token groups;
resident KV quantization uses separate eight-token groups. These are different
operators, neither a faithful FIER codec nor a WindowQuant reproduction.

For resident quantization, min/scale are computed over the eight tokens per KV
head/channel. Metadata is float32, with 2 or 4 bits per code. High-precision
chunks undergo an FP16 roundtrip. Nonvisual keys/values and newly generated KV
remain native float32. All arms use the same quantizer. Nine query heads share
three KV heads. We evaluate output before `o_proj`, using the captured native
queries and teacher-forced new KV. Error does not propagate across layers or
generation steps. The FP16 reference has small rounding error because the
original model ran float32; it is not an exactly zero-error intervention.

The primary gate required the 1-bit arm to reduce relative L2 by at least 5%
against every deterministic nonrandom ranking control, without lowering cosine,
on both images. It failed. All four decoder steps and all layers remain in
`result.json`; `summary.json` includes step means and layer/step distributions.
The same-code deterministic rerun produced a byte-identical result. This is
reproduction on one machine, not external replication.

## Cost and interpretation

Each equal-budget arm models 19,200 visual payload bytes plus 18,432 metadata
bytes and 32 permutation bytes per layer: **37,664 bytes**. All-FP16 visual KV
uses 49,152 bytes; native FP32 uses 98,304. Nonvisual storage is unchanged and
reported separately. Bit packing is not implemented; these are analytical
storage counts, not NumPy allocation or measured GPU memory. Allocator padding,
kernel workspace, transient scoring and dequantization buffers are excluded.

There is no measured speed or memory-traffic advantage. Lower-cost scoring
could still justify some fidelity loss in a different systems experiment; this
pilot does not test that tradeoff. Its narrower finding is that the proposed
1-bit signal does not outperform stronger prefill controls at fixed resident
precision counts on these fixtures. Broad query-aware quantization is already
covered by the [prior-art audit](../../docs/method-grounding/kv-window-ranking-prior-art.md).
No primitive, ranker or novelty claim is promoted from this result.

## Reproduce

With the archived `results/smolvlm-trace-v3` inputs present, from the engine root:

```sh
sh operator-studies/trace-window-ablation/run.sh results/trace-ablation-fresh
```

The runner freezes and evaluates once into a fresh directory. NumPy 2.5.4 and
Python 3.12.14 were used; no model/API calls are needed. Source NPZ hashes are
recorded in the protocol. Quantizer-level, constant-input, fixed-count, tie,
nonfinite-input and GQA permutation checks are in `test_experiment.py`.
