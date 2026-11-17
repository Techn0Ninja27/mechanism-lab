# Portfolio audit: retained-prompt eviction candidate

Reviewed 2026-11-17 from the local portfolio result
`results/serving-portfolio-v1/decode/result.json`. This note is a code and
semantics audit, not a novelty judgment and not permission to run the proposed
experiment.

## What the portfolio actually found

The three generated candidates are not three independent mechanisms.

* `cand-00` replaces peak reservation with lower-bound admission and proposes
  routing an evicted request to a prefill-held copy when possible.
* `cand-01` changes the trigger from lower-bound overflow to a geometric phase
  kill, but sends the killed request to the same retained prefill state. The
  reviewer correctly calls these the same retained-target transition with a
  different trigger. The structural diagnostics report no near-duplicate pair,
  so the current fingerprint is too literal to detect this equivalence.
* `cand-02` changes the pool-flip signal and the reviewer marks it redundant
  with Arrow-style decode-load flipping.

The portfolio review selects `cand-00` for its cheap ablation, not because its
novelty or usefulness is established. Its central claim is only partially
covered, its mechanism is unresolved, and its expected 5% gain is unknown.

## The selected transition is underspecified

The phrase “retained prefill copy” does not identify a valid memory state. In
DMRS-like accounting, a request in `W` owns prompt KV in prefill memory. When it
is admitted to decode, that KV is transferred to the decode pool and released
from prefill memory. After that transition there are three materially different
implementations:

1. **Move:** transfer the only prompt KV copy from decode back to prefill on an
   eviction. Prefill occupancy increases by `p_i`, decode occupancy decreases
   by the transferred KV, and the transfer has a latency/bandwidth cost.
2. **Duplicate:** keep a prefill copy while the request is decoding. Decode
   admission then adds `p_i + lower_i` while prefill continues to count `p_i`.
   This can consume roughly an additional `p_i` memory units compared with the
   move policy and requires an explicit copy operation.
3. **Discard:** release the decode copy and return the request to `U`; a later
   prefill recomputes the prompt KV. This is the existing re-prefill control.

The generated candidate does not choose among these states. Its claimed
memory-safety invariant therefore cannot be checked: `M_P` and `M_D` depend on
whether the copy is moved or duplicated, and the cost comparison depends on
transfer latency, prefill compute time, and whether a copy can be retained
without violating `M_P`. “Instantaneous transfer” in DMRS does not decide the
ownership or duplicate-copy question for this new transition.

The candidate also leaves the initial lower bound, lower-bound update after a
restart, behavior when `p_i + d_i > M_D`, and event ordering around decode
growth and EOS unspecified. The local preflight used lower bound 1 and a
single-copy move/discard state, but that is a development assumption and does
not validate `cand-00`'s retained-copy branch.

## Cheapest analytic check before simulation

Do not start with a stochastic sweep. Freeze one request with prompt size
`p`, observed decode count `a`, lower bound `ell`, and prefill recompute cost
`m`, plus a second request that forces one decode eviction. For the single
eviction event, write the accounting table for all three transitions:

| Transition | Change in `M_P` | Change in `M_D` | Extra work/latency |
|---|---:|---:|---|
| move back to prefill | `+p` | `-(p+a)` | KV transfer back |
| duplicate/retain | `+p` while decode remains | `0` until later release | KV copy plus retained memory |
| discard and re-prefill | `0` immediately | `-(p+a)` | recompute cost `m` later |

The simulator should reject the trace unless every intermediate state, including
the state before EOS frees memory, satisfies both capacities. The policy may
observe only arrival, `p`, `a`, lower-bound state, and EOS; true `d` belongs
only to the fixture oracle. If the transition is move, model transfer cost and
ownership explicitly. If it is duplicate, count both copies. If neither can be
specified, stop and classify the proposal as unresolved rather than inventing a
favorable implementation.

Only after this table is executable should `cand-00` be compared against two
matched controls: the same lower-bound admission with discard/re-prefill, and a
peak-reservation policy using the same declared upper-bound information. The
analytic test should first compute whether the maximum possible saved work
(`evictions × m`, less transfer cost) can reach 5% of the control's measured
flow-time budget. If it cannot, the candidate is killed without a larger sweep.

The current result supports a revised semantic specification and this cheap
accounting test. It does not support a claim that the portfolio produced three
distinct research directions or that `cand-00` is ready for implementation.
