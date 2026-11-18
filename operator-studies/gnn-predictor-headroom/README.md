# GNN predictor headroom screen

Status: development-only implementation, 2026-11-18. No paid calls are made.

This study asks whether cheap causal predictors have measurable headroom over
the current self-fed SNI-inspired predictor on the same Cora trajectory task.
It uses only seeds 10 and 11, under the ordinary and prespecified learning-rate
drop schedules. Seeds 12–15 remain reserved for a later held-out experiment.

The runner reuses `gnn-sync-gate/screen.py` for Cora loading and GNN trajectory
training. It checks that the imported `replay_gate` predictor and the local
implementation of the lambda=0.2, beta=0.2 damped predictor produce the same
stale-error metric within a `1e-6` floating-point tolerance. The local controls are:

- last-observed hold;
- self-fed damped trend with frozen lambda (slope-memory) and beta (forecast)
  grids;
- fixed hold/trend mixtures;
- a deterministic bounded continuous global mixture approximation;
- global ridge autoregressive predictors fit only from sync observations, with
  fixed lag and regularization grids.

For the self-fed predictor, a synchronization endpoint is evaluated before its
new observation is consumed. The primary metric includes stale epochs only;
the endpoint truth then becomes available for the next interval. During stale
epochs, the predictor receives only its own previous prediction. Intermediate
truth is used for scoring only. Ridge/AR controls forecast the next sync from
equally spaced sync history and linearly interpolate within the stale interval;
they do not train on the endpoint being predicted.

This is a finite development screen. It is not an exhaustive search over the
future arithmetic DSL, a least-squares upper bound, a held-out result, a
closed-loop training result, or evidence of downstream GNN accuracy.

The initial design fixed lambda at 0.2. Before this screen was run, an
independent calibration artifact showed that changing both the slope-memory
decay and forecast damping could matter. The design was therefore revised
openly to include the predeclared lambda and beta grids; no result from that
calibration is treated as this study's held-out evidence. The later paid pilot
must beat the best jointly tuned control, rather than the original SNI setting
alone.

Run from the repository root after the Cora files have been prepared:

```sh
../.venv/bin/python -m unittest discover -s operator-studies/gnn-predictor-headroom -p 'test_*.py'
../.venv/bin/python operator-studies/gnn-predictor-headroom/run.py \
  --data results/gnn-sync-gate-v1/data \
  --output results/gnn-predictor-headroom-v1
```

The run writes `freeze.json` before loading observations and then writes
`result.json` with all development rows, control parameters, source hashes,
and the exact-baseline comparison. Existing output directories are never
overwritten. A meaningful allocation rule for a later paid pilot is that an
alternative must reduce stale error by at least 5% relative to the original
self-fed baseline in both schedules. This is necessary for allocation only,
not sufficient evidence of usefulness or generalization; after this screen it
must beat the jointly tuned controls as well.

## Development result (2026-11-18)

The four requested trajectories completed with no API calls. The local
implementation matched the imported predictor metric within `2.97e-10` at
worst (the acceptance tolerance is `1e-6`). A single configuration selected
pooled over both seeds and both schedules was `lambda=0.4`, `beta=0.05`,
`w=1.0` (the damped trend predictor with no hold mixture). Relative to the
original `lambda=0.2`, `beta=0.2` baseline, its stale-error reduction was
29.9% on the constant schedule and 33.6% on the learning-rate-drop schedule.
That clears the 5% allocation screen in both schedules.

This is a calibration finding, not evidence that the research engine is useful:
the original control was under-tuned. The best pooled ridge/AR setting
(`lag=3`, ridge `0.001`) reduced error 24.5% on the constant schedule but only
3.6% after the learning-rate drop. The pooled continuous-mixture control chose
weight 1.0 and did not improve over the original trend. The held-out seeds were
not opened, and no closed-loop accuracy result was produced. A later paid
representation pilot must beat this jointly tuned control and include the
raw-source-plus-distillation confound control before it can support any
usefulness claim.


Seven tests pass, including the separate forecast-horizon regression. The first
attempt was interrupted to fix excessive in-memory trace retention; its original
freeze remains locally at `results/gnn-predictor-headroom-v1-interrupted-2026-11-18/`.
The successful [result](result.json) and [freeze](freeze.json) are published here.
The freeze binds the runner and actual Cora input bytes; input data stays local.
Use the pinned requirements and a fresh output path to rerun.
