# Factual audit of the serving proposal

Reviewed 2026-11-17 against `results/serving-problem-v2/decode/result.json`,
the supplied DMRS/Arrow/Feng/Chen excerpts, and the review embedded in that
result. This is an audit of coherence and evidence scope. It is not a novelty
judgment and does not refute the underlying papers.

## Main assessment

The proposal has a real interface mismatch to study: DMRS couples retained
prefill KV with decode admission under two memory pools, while the supplied
Feng and Chen works primarily study single-pool uncertainty. The proposed
largest-prompt-first permutation is a plausible simulator intervention, but it
is currently a weak heuristic rather than a demonstrated mechanism. The model
does not yet specify enough state or event semantics to support a safe
implementation or a fair comparison.

## Issues that must be fixed before implementation

1. **The lower-bound state is underspecified.** `q_tilde_i` is called a
   certified lower bound, but the proposal does not say how it is obtained at
   arrival, what the initial value is, or what information is available without
   reading the future output. If it is initialized to one token, this is nearly
   a work-conserving reservation policy; if it is predicted, the predictor and
   calibration become part of the method. After eviction, the previous
   observed prefix can remain a lower bound on total response length, but the
   simulator must distinguish that persistent bound from the reset current
   prefix after re-prefill.

2. **The claimed memory invariant is incomplete.** The algorithm checks the
   decode pool before a token grows, then evicts jobs if the projected demand
   exceeds capacity. A single active request whose prompt plus observed output
   exceeds `M_D` can be evicted and restarted forever; no feasibility rule,
   spill policy, or admission rejection is specified. The implementation must
   define what happens when `p_i + d_i > M_D`, even though `d_i` is unknown to
   the scheduler. It must also define whether KV transfer is instantaneous or
   consumes time and temporary memory. DMRS assumes instantaneous transfer in
   theory but includes transfer overhead in its GPU measurements.

3. **Slot ordering is ambiguous.** The text says overflow is checked before
   decode growth, but it does not specify whether completions, prefill
   completions, decode admission, token generation, and memory accounting occur
   in that exact order. The invariant and completion times depend on this
   ordering. State transitions should be written as an executable transition
   table, including the current prefix, persistent lower bound, and memory
   charge before and after one decode token.

4. **The control set is too weak.** FIFO alone cannot isolate the proposed
   effect. The review correctly identifies at least smallest-reservation-first
   (`p_i + q_tilde_i`), smallest-prompt-first, DMRS's known-length ordering
   surrogate, and a work-conserving/backpressure policy as controls. The
   central claim should survive the best of these controls, with identical
   information, capacities, arrival traces, and seeds.

5. **The hypothesis and kill rule disagree.** The hypothesis asks for at least
   5% lower TCT, while the kill threshold accepts a TCT ratio whose upper
   confidence bound is below 0.97, i.e. roughly a 3% improvement. Choose one
   pre-registered threshold or make the result descriptive. With 20 synthetic
   seeds, bootstrap the appropriate request-level or trace-level unit; do not
   treat individual requests as independent across the same generated trace.

6. **The simulator does not establish serving performance.** The proposed
   uniform slot costs omit hardware scheduling, transfer time, batching
   contention, and model-specific token costs. This is acceptable for a causal
   scheduling screen only if the report labels it simulation evidence and
   retains total completion time, P95/P99, eviction/re-prefill count, and
   resource utilization. Any hardware claim would need a separate test.

7. **The evidence and retrieval coverage are bounded.** The paid run's local
   retrieval returned zero records because its conjunction queries included
   author names. The proposal therefore has no retrieved-corpus support beyond
   the supplied pack. The supplied adjacent papers already cover generic
   non-clairvoyant scheduling, geometric restarts, and interval refinement;
   the remaining two-pool interface difference is not evidence of novelty.

## Cheapest falsification

Use a two-request deterministic trace before building a simulator. Let the
prefill pool hold both requests (`M_P = 10`), let decode capacity be
`M_D = 10`, and let both requests initially have certified lower bound one.
Request A has prompt 8 and eventual output 2; request B has prompt 2 and
eventual output 1. Each full request fits alone (`8+2=10`, `2+1=3`), but the
initial reservations do not fit together (`8+1 + 2+1 = 12`). With unit-cost
decode slots, largest-prompt-first runs A then B, giving completion times 2 and
3 and total completion time 5. Smallest-prompt-first runs B then A, giving
completion times 1 and 3 and total 4. This is not evidence about real serving;
it is a cheap counterexample showing that the proposed ordering can harm total
completion time even when every request is feasible. The implementation should
reproduce this trace before any stochastic sweep and also include the symmetric
regime in which freeing a large retained prompt helps, otherwise the causal
story is one-sided.

## Required disposition

The proposal's `revise` decision is appropriate; the separate reviewer recommended a small test with stronger controls. Do not call the
proposal novel or pay for implementation until the lower-bound source,
transition semantics, feasibility behavior, comparator set, and threshold are
fixed. If the two-request trace and best-control sweep show no robust regime
where largest-first helps without a P99 penalty, reject this operator and keep
the negative result.

