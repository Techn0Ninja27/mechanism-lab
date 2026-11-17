# Serving admission preflight

2026-11-17. Corrected development diagnostic for the generated prompt-release
ordering proposal. **No efficacy or novelty claim.** Four policies share one
state machine: FIFO, largest prompt first, smallest reservation first, and
smallest prefill time first.

The first agent implementation admitted against current decode occupancy,
not projected next-token occupancy. Its after-EOS memory check could hide a
transient overflow. That v1 code/output is preserved locally under
`results/serving-admission-preflight-v1/` and is not valid invariant evidence.
The current implementation reserves projected growth during admission and
checks the peak before any EOS releases memory. The hidden true length is
stored separately from the policy-visible request data.

Explicit toy assumptions: initial length lower bound1; eviction resets decoded
progress and requires prefill again while retaining the maximum certified
lower bound; instantaneous transfer; unit decode slots; parallel prefills
limited only by memory. An oracle checks individual request feasibility in the
fixture before simulation; the policy does not receive true lengths. These
choices resolve underspecification for the toy, not for a deployable system.

| Diagnostic | FIFO | Largest prompt | Smallest reservation | Smallest prefill |
|---|---:|---:|---:|---:|
| Equal prompts | 9 | 9 | 9 | 9 |
| Capacity edge/restart | 11 | 11 | 11 | 11 |
| Original case with no ordering effect | 9 | 9 | 9 | 9 |
| Two-request counterexample | 7 | 7 | 6 | 7 |
| Projected-growth admission | 9 | 9 | 9 | 9 |

Values are summed flow time (completion minus arrival) in simulator slots.
The counterexample uses prompt sizes8 and2, outputs2 and1, unit prefills,
and capacities10/10. Largest-first loses to smallest-first. This disproves
uniform dominance, not a possible regime-specific benefit or the generated
stochastic hypothesis. The equal-prompt case checks the expected no-effect
boundary; the restart case exercises re-prefill; the growth case catches the
old admission bug. All current traces satisfy the peak checks.

[Raw traces](result.json), [summary](summary.json) and [development freeze](freeze.json)
are retained. The cases and code were inspected during debugging before the
batch run; they are not untouched validation data. The generated hypothesis's
5% threshold and its3% kill rule remain inconsistent, and no efficacy gate
is evaluated here. Backpressure and literature baselines remain outstanding.

Reproduce using Python3.12.14 (standard library only):

```sh
python operator-studies/serving-admission-preflight/run_cases.py --cases operator-studies/serving-admission-preflight/cases.json --output /tmp/serving-preflight-result.json
python -m unittest discover -s operator-studies/serving-admission-preflight -p 'test_*.py' -v
```

Choose a fresh output path. Six tests include projected-growth safety, the
counterexample, restarts, infeasible fixtures, and policy observation isolation.
