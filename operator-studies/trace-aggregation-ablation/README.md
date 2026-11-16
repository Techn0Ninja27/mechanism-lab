# Max aggregation follow-up on exposed attention traces

2026-11-17. A fresh model review suggested that averaging token logits caused
the earlier ranking failure. We tested window maxima without changing the
resident precision budget or replay engine. This is a posthoc exploratory check
on the same two exposed images, not a held-out test or novel-method result.

For each prefill text query and query head, the new arms take the maximum raw
QK logit within each eight-token window, then average those maxima over queries
and heads. One-bit, two-bit and unquantized scoring keys are compared. Resident
KV quantization, counts and evaluation are unchanged. Decode queries never enter
the ranking. The old replay source is hash-pinned and unmodified; the driver
injects a scoring callback in memory. Every original baseline summary reproduces
exactly before the new results are accepted.

Mean relative L2 output error over layers and decoder steps 1–3:

| Scorer | Candy | Statue |
|---|---:|---:|
| One-bit mean | 0.0251777 | 0.0222857 |
| One-bit max | 0.0249168 | 0.0211241 |
| Two-bit mean | 0.0180379 | 0.0219475 |
| Two-bit max | 0.0179206 | 0.0185683 |
| Exact-key max | 0.0206906 | 0.0180212 |
| Exact softmax mass | **0.0151038** | **0.0151007** |

One-bit max lowers relative L2 slightly on both fixtures, satisfying only the
protocol's descriptive comparison against one-bit mean. It still loses to exact
mass on both, and its cosine slightly worsens on candy (0.9992398 versus
0.9992563). Thus it does not dominate even its mean counterpart on both metrics.
The test does not establish that aggregation alone explains the original gap.
No speed, traffic, task-quality, error-propagation or novelty gain was measured.
The `one_bit_max_improves_both_images` boolean in the result refers only to the
relative-L2 comparison specified in the protocol, not to general usefulness.

The source traces, limitations, quantizer and analytical cache accounting are
described in [the original pilot](../trace-window-ablation/README.md). The new
protocol and driver were frozen before this follow-up's scores were computed.
Original results are not rewritten. The result retains only new-arm detailed
rows; complete original rows are linked by their frozen result hash.

Reproduce from the engine root with the archived traces present:

```sh
sh operator-studies/trace-aggregation-ablation/run.sh results/aggregation-fresh
```

Python 3.12.14 and NumPy 2.5.4 were used. No generation calls, new model inference
or network access are needed for replay.
