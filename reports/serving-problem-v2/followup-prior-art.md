# Follow-up prior-art audit for the prompt-release ordering candidate

Reviewed 2026-11-17 against newer local-corpus leads and their primary arXiv full texts. This note audits overlap; it does not establish external novelty.

## What the candidate claims

The generated candidate changes only the order in which requests waiting for decode admission are scanned: largest retained prompt cache first. Its stated mechanism is that admitting a large prompt releases more prefill-pool memory. The proposed controls were initially too weak: FIFO was included, but smallest-reservation-first and backpressure were not. The corrected toy preflight found a two-request trace where largest-first has flow time 7 and smallest-reservation-first has 6.

## Direct overlap

**Online Scheduling for LLM Inference with KV Cache Constraints** ([Jaillet et al., arXiv:2502.07115](https://arxiv.org/html/2502.07115)) already defines an online KV-memory scheduling problem and a Memory-Constrained Shortest-First (MC-SF) policy. Its algorithm scans waiting requests in ascending predicted output length and checks memory at the future completion times of active and newly added requests (source §4, Algorithm 1). This is a directly relevant control for any proposed waiting-set permutation. The paper also states that arbitrary arrivals prevent a deterministic online algorithm from having a constant competitive ratio in general, under its model; this is not a bound transferred to the dual-memory model or later frameworks. A small simulator result cannot establish a general guarantee.

**LLM Serving Optimization with Variable Prefill and Decode Lengths** ([Wang, Ye, and Zhou, arXiv:2508.06133](https://arxiv.org/html/2508.06133v5)) studies heterogeneous prompt and response lengths under a KV budget. It explicitly reports that FCFS, shortest-output-first, and total-size priority can have unbounded approximation ratios in its offline model, and proposes Sorted-F, whose batch metric balances batch size against downstream decode cost (source §3). Its online extension is receding-horizon and the paper states that formal online guarantees remain out of scope (source §8.2). A largest-prompt permutation is therefore not an unexplored general answer to variable prompt/decode scheduling; it must beat a stronger F-metric or receding-horizon comparator in the exact target interface.

**Taming Request Imbalance: SLO-Aware Scheduling for Disaggregated LLM Inference** ([Wang, arXiv:2605.02329](https://arxiv.org/html/2605.02329)) is an implementation-level disaggregated serving baseline. Its prefill algorithm predicts finish times and ranks requests by normalized TTFT urgency; its decode algorithm uses TPOT slack to adaptively batch short requests (source §3). The paper identifies FCFS prefill head-of-line blocking and short-request starvation under naive shortest-job-first. This directly rules out treating FIFO versus a prompt-size ordering as the complete practical baseline. A proper test must include urgency/SLO and starvation controls, or explicitly state that it studies a different objective without SLOs.

**HAServe (v2, revised 13 October 2026; the local corpus retained the earlier PRISM title)** ([Qu et al., arXiv:2605.08581](https://arxiv.org/html/2605.08581v2)) is adjacent systems work on cache reuse, rather than a matching baseline for retained-prefill memory alone. It jointly selects admission and KV retention from a visible request horizon, using exact prefix paths and visible consumers; its scheduler commits both decisions together (source §4). It evaluates P50/P90/P95/P99 TTFT, throughput, cache hit rate, and answer quality on repeated-prompt workloads. The candidate's prompt-size ordering has no prefix-consumer model, retention decision, or visible-horizon latency objective, so it is a materially simpler interface. That simplicity does not make it novel: it means HAServe is a required comparator whenever prefix reuse or retention is part of the workload.

## Consequence for the candidate

The candidate survives only as a narrow, implementation-neutral question about one ordering inside a dual-memory state machine. The adjacent papers already cover shortest-first, F-metric batch selection, urgency/SLO scheduling, starvation behavior, and joint admission/retention. The two-pool retained prefill-cache interface from DMRS remains a scoped mismatch, but a mismatch is not evidence that largest-prompt-first is new or useful.

The smallest defensible next test, if retained at all, must compare at least:

* FIFO/work-conserving;
* smallest-reservation-first or MC-SF-style projected-feasibility order;
* largest-prompt-first;
* a backpressure policy;
* Sorted-F or a clearly documented approximation of its relevant batch rule;
* urgency/SLO scheduling when tail latency is part of the objective.

All policies must receive the same observable state and no true future output lengths. Report total completion time, P95/P99 or an explicitly different tail metric, evictions/re-prefills, and starvation or maximum waiting time. Keep offline backlog and online arrivals as separate regimes. Do not carry over DMRS's offline approximation guarantees, or the headline numbers of any system paper, into this simulator.

The corrected toy counterexample is enough to reject uniform dominance of the candidate. Because the remaining comparison set is already substantial and the proposed operator is only a queue permutation, the evidence currently supports closing the candidate unless a pre-registered regime-specific result beats the strongest controls without a tail-latency or starvation penalty.

## Retrieval caveat

The paid run's author-name conjunctions returned zero local-corpus records. The later diagnostic found the sources above with shorter mechanism queries, plus PRISM/HAServe, FlowKV, output-length prediction, and other leads. Those leads were not supplied to the paid model retroactively. They are retrieval leads, not entailment or novelty evidence; full-text checks are still required before any efficacy study.


The bounded local screen subsequently failed against simpler matched controls;
therefore we did not implement these larger literature systems merely to
continue an already unsuccessful candidate. Their different information and
resource assumptions remain explicit coverage gaps, not defeated baselines.
