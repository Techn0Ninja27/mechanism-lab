# Largest-prompt-first admission: bounded control screen

Protocol fixed 2026-11-17 before inspecting the sweep. This is a synthetic
**development allocation test**, not a serving benchmark or novelty result.
The generated proposal's conflicting3%/5% thresholds are resolved here to a
5% mean-flow-time reduction. This is a specified variant, not retrospective
validation of its ambiguous original experiment.

180 traces: 10 seeds × two arrival patterns × three prompt/output association
patterns × three prefill capacities. Each trace has24 requests, prompt sizes1–8,
output lengths1–12, prefill duration ceil(prompt/2), decode capacity24, and
prefill capacity8/16/32. Staggered interarrival increments use a fixed discrete
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
capacity8. Within each trace compare mean flow time (completion minus arrival)
and P99 flow time with the best of the five controls *separately per metric*.
This is a conservative comparison envelope, not an executable oracle policy.
A group passes if its median mean-flow ratio is <=0.95 and median P99 ratio
<=1.05; continuation requires every primary group to pass. Other capacities
are descriptive sensitivity checks and cannot rescue a failed primary gate.
Any incomplete arm invalidates its group's continuation gate; no completed-only
selection. The frozen horizon is1000slots. Invariant errors stop the run.

Request-level P95/P99 use nearest ranks. With24 requests, P99 is the maximum;
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
