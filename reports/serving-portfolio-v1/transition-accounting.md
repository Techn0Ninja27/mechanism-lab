# Retained-prompt transition accounting

2026-11-17. Analytic bookkeeping for the selected candidate, not a measured
result. Let p be prompt-KV storage, a generated-response storage, m prefill
recomputation time, and tau(p) prompt transfer time. Neither m nor tau is
assumed constant or favorable. Other requests occupy P and D in the pools.

| Choice | Before decode eviction | Eviction transition | Required check / cost |
|---|---|---|---|
| Discard then re-prefill | D includes p+a; P has no copy | Release D; return to unassigned, later prefill | Later P+p must fit; recompute prefill cost m and replay discarded response progress |
| Move prompt back | D includes p+a; P has no copy | Reserve p in P, transfer prompt, acknowledge, release D | P+p must fit **before** transfer. Conservative peak holds copies in both pools until acknowledgement; cost tau(p), then replay discarded response progress |
| Keep duplicate throughout decode | P includes p; D includes p+a | Release D and reuse already-held prompt in P | P occupancy persists even for requests never evicted; count both copies from decode admission. Replay discarded response progress still required |

If move-back has no destination capacity, the policy must explicitly wait,
discard, or choose another eviction. It cannot allocate an uncounted copy.
If the implementation can stream/reclaim progressively, that is a separate
specified transfer mechanism requiring its own peak-memory accounting.
Keeping a duplicate also forfeits the earlier release of prefill memory that
motivated the failed queue-ordering candidate.

All comparisons must use the same observed lower-bound state and response
replay rules. Retaining prompt KV saves prefill reconstruction only; it does
not restore discarded generated-token KV. A lower-bound admission improvement
is not automatically attributable to retention. Isolate admission and retention
as separate factors before crediting either. These distinctions expose missing
definitions; they do not prove the policy succeeds or fails.
