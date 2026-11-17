# Largest-prompt-first admission: bounded control screen

Protocol fixed 2026-11-17 before inspecting the sweep. This is a synthetic
**development allocation test**, not a serving benchmark or novelty result.
The generated proposal's conflicting3%/5% thresholds are resolved here to a
5% mean-flow-time reduction. This is a specified variant, not retrospective
validation of its ambiguous original experiment.

180 traces: 10 seeds × two arrival patterns × three prompt/output association
patterns × three prefill capacities. Each trace has 24 requests, prompt sizes 1–8,
output lengths 1–12, prefill duration ceil(prompt/2), decode capacity 24, and
prefill capacity 8/16/32. Staggered interarrival increments use a fixed discrete
distribution; bursts contain8 requests every12 slots. Output/prompt association
is independent, positive or negative by the fixed generator in run.py, not a
claim of any exact correlation coefficient. Paired capacities share traces.

Six arms see identical online observations: FIFO, largest-prompt-first,
smallest-reservation-first, smallest-prompt-first, smallest-prefill-first, and
FIFO with prefill backpressure. The latter pauses new prefills when retained
waiting-prompt memory is at least half of prefill capacity, evaluated after
decode admissions. No threshold tuning is performed. All arms retain the same
projected-growth memory checks, eviction, lowerbound1 and restart rules.
The simulator is copied from the frozen corrected preflight; the only changes
are smallest-prompt ordering and the explicit backpressure gate.

The primary groups are the six arrival/association groups with prefill
capacity 8. Within each trace compare mean flow time (completion minus arrival)
and P99 flow time with the best of the five controls *separately per metric*.
This is a conservative comparison envelope, not an executable oracle policy.
A group passes if its median mean-flow ratio is <=0.95 and median P99 ratio
<=1.05; continuation requires every primary group to pass. Other capacities
are descriptive sensitivity checks and cannot rescue a failed primary gate.
Any incomplete arm invalidates its group's continuation gate; no completed-only
selection. The frozen horizon is 1000slots. Invariant errors stop the run.

Request-level P95/P99 use nearest ranks. With 24 requests, P99 is the maximum;
we report this limitation rather than interpreting it as a production-tail
estimate. Aggregation is across paired traces, not pooled request samples.
Absolute completion-time sums, restart counts and slots with a capacity-blocked
prefill queue are also recorded. There are no confidence-interval or population
claims. The model ignores hardware contention and transfer latency, permits
parallel prefills subject only to memory, and filters individually infeasible
fixtures with an oracle whose lengths are not exposed to policy code.

Run with Python3.12.14, standard library only, using a fresh output directory:

```sh
python operator-studies/serving-admission-screen/run.py --output /tmp/serving-admission-screen
python -m unittest discover -s operator-studies/serving-admission-screen -p 'test_*.py' -v
```

No network or paid APIs. The code and case digest are frozen before the sweep.
A pass would justify a broader independent test; a failure closes this ordering
candidate on the frozen model/settings without rejecting all two-pool policies.

## Result

All 1,080 arms completed across 180 traces; none was dropped or censored. The
candidate failed all six primary groups. Ratios below are medians across the
10 paired traces in each group, lower is better:

| Arrivals | Association | Mean flow / best-control envelope | P99 / best-control envelope | Mean flow / FIFO |
|---|---|---:|---:|---:|
| Staggered | Independent | 1.0222 | 1.0618 | 0.9995 |
| Staggered | Positive | 1.1122 | 1.0630 | 1.0000 |
| Staggered | Negative | 1.0830 | 1.1052 | 1.0074 |
| Bursts | Independent | 1.0182 | 1.0051 | 1.0000 |
| Bursts | Positive | 1.0000 | 1.0000 | 1.0000 |
| Bursts | Negative | 1.1038 | 1.1524 | 1.0171 |

The envelope chooses the best control separately on each trace and metric;
it is intentionally conservative and is not an executable scheduler. The
[post-run fixed-control decomposition](fixed-control-descriptive.json) also
shows no primary group achieving a 5% median improvement over FIFO alone.
Some comparisons against other fixed controls favor the candidate, so this
is not a claim that one policy dominates every trace. No group at the two
secondary capacities met both criteria either.

Decision: close this largest-prompt-first candidate on the frozen toy model.
Do not tune its thresholds on these exposed traces and call the outcome held
out. This does not reject dual-memory admission generally. Stronger literature
implementations and hardware models were not tested; the candidate already
failed the cheaper simple-control screen.

[Raw metrics and trace hashes](raw.json), [generated cases](cases.json),
[result](result.json), [pre-run protocol](protocol.md), and [freeze](freeze.json)
are committed. A fresh deterministic replay reproduced the same outputs.
No paid API calls or network downloads were used by the experiment.
