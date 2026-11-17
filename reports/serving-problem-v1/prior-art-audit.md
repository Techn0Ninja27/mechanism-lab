# Prior-art audit for the serving pack

Reviewed 2026-11-17 against the primary arXiv records cited by DMRS. This
document is a boundary check, not a novelty claim.

## Feng et al., 2026a: general non-clairvoyant KV-cache scheduling

[General Non-Clairvoyant KV-Cache Scheduling via Regime-Aware Routing,
arXiv:2607.09248](https://arxiv.org/abs/2607.09248) studies batched inference
with a hard KV-cache budget. Prompt length is known, response length is
unknown, and the response component grows during decoding. Eviction discards
accumulated cache computation. Its regime-aware routing framework time-shares
the memory budget among specialized sub-schedulers and routes jobs as their
behavior is revealed. The abstract claims constant competitive guarantees for
completion time, makespan, and online arrivals.

This directly covers the most obvious proposal of adding unknown output
lengths to a single-memory version of the problem. A new proposal cannot claim
that non-clairvoyant scheduling is absent or that a generic prediction-free
regime router is new. The remaining mismatch is structural: DMRS has two
separate memory pools and retained prompt KV while a request waits for decode;
Feng et al.'s abstract describes one hard KV budget and does not establish the
same dual-pool retained-cache model. That is a scoped interface difference,
not evidence that a dual-pool online algorithm would be novel.

## Feng et al., 2026b: geometric slicing and batching

[Competitive Non-Clairvoyant KV-Cache Scheduling for LLM Inference,
arXiv:2601.22996](https://arxiv.org/abs/2601.22996) proposes the Geometric
Slicing Algorithm for an offline batch with unknown response lengths. It
periodically restarts jobs to bound memory exposure and uses a staggered
pipeline. The paper reports competitive ratios 61.92 generally and 32 in a
large-memory regime; its clairvoyant Geometric Batching Algorithm reports
10.67 and 6.75 in the corresponding regimes, with trace experiments.

This rules out presenting geometric phases, restart-based eviction, or a
clairvoyant/non-clairvoyant pair as a fresh central mechanism. It still uses a
single KV-cache budget in the abstracted problem, so it does not by itself
resolve DMRS's retained-prefill-memory plus decode-memory coupling.

## Chen, Ye, and Zhou, 2025: uncertainty intervals

[Adaptively Robust LLM Inference Optimization under Prediction Uncertainty,
arXiv:2508.14544](https://arxiv.org/abs/2508.14544) assumes an interval
prediction for each output length. Its conservative `A_max` uses the upper
bound and degrades with overestimation; `A_min` starts from the lower bound and
refines the estimate online, with a log-scale competitive guarantee. The paper
uses prediction intervals rather than zero-information non-clairvoyance.

This rules out a proposal that merely swaps an upper-bound admission rule for a
lower-bound rule or claims adaptive refinement under interval uncertainty is
unexplored. The exact interaction between interval uncertainty, two separate
memory capacities, retained prompt cache, transfer delay, and tail-latency
objectives is not established by the abstract. Any follow-up must compare
against both the lower-bound refinement policy and a no-prediction baseline,
and must state whether output-length intervals are available online.

## What survives the comparison

DMRS's actual model is an offline batch released at time zero with known
`(m_i, p_i, d_i)`, separate prefill/decode capacities, non-preemptive phases,
and instantaneous theoretical KV transfer. Its GPU study fixes output lengths
and uses an external proxy; the measured experiments include transfer overhead.
The paper itself names online and non-clairvoyant variants as future work.
Those statements define an unresolved interface, not a novelty result.

The strongest defensible next question is whether an already-known
non-clairvoyant or interval-refinement policy can be adapted to *two* memory
pools with retained prompt caches while preserving both completion-time and
tail-latency behavior. But the supplied sources do not yet show a new
operator, a formal guarantee, or a deployment win for that bridge. Before any
paid generation, add the full texts of Feng 2607.09248, Feng 2601.22996 and
Chen 2508.14544 to the source pack and compare their algorithms line by line.
If that comparison finds the dual-pool bridge already covered, the correct
output is `reject` or `insufficient_evidence`, not a forced proposal.

The DMRS results themselves are mixed: its algorithms beat the P/D baseline
on the reported table, but colocated serving is faster on conversation rows
and the appendix reports higher P99 completion times in several settings.
Any experiment must retain colocated, P/D, and the relevant non-clairvoyant or
interval baseline, and report TCT together with P95/P99 latency.
