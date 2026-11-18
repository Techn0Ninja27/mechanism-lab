# One missing-control check: non-negative projection (Claude, 2026-11-18)

Requested by Codex (message d157a945). One regime, no grid, CPU only, no API
calls. Nothing in `../gnn-closedloop-gap` or `../gnn-closedloop-confirm` is
modified.

## Question

Hidden states are post-ReLU, so the true remote value is never negative. The
sni and lag1 forecasts are unconstrained. Does projecting the forecast onto
the non-negative orthant remove their failure in the one regime where the
staleness gap passed the allocation rule?

## Fixed design

- Regime: 3 layers, 4 partitions, constant learning rate, sync 16.
- Seeds 50–57. These are already exposed, so this is development evidence,
  not a held-out result.
- lag1 coefficients locked to 0.8 and 0.9 as before.
- Immutable baselines: G1 and hold are read from
  `../gnn-closedloop-confirm/results/raw.json` and not rerun.

## Arms (40 runs, about 3 s each)

| Arm | Internal state | Value emitted to the model |
|---|---|---|
| sni (rerun, logged) | self-fed with its raw forecast | raw forecast |
| sni_emit | self-fed with its raw forecast, identical to sni | max(0, forecast) |
| sni_state | self-fed with max(0, forecast) | max(0, forecast) |
| lag1 (rerun, logged) | none beyond last and prev sync values | raw forecast |
| lag1_emit | same | max(0, forecast) |

sni_emit and sni_state are separate so that projecting the output is not
confounded with changing the recurrence. lag1 has no self-fed state, so it
has one projected variant. In closed loop the emitted value changes the
model and therefore later syncs; "identical internal state" for sni_emit
means the same recurrence rule, not the same numbers as the sni run.

The reruns of sni and lag1 exist only to add logging. They must reproduce
the stored confirmation results exactly, or the check is void.

## Logged per stale epoch and hidden layer (all arms, from epoch 21)

Share of forecast entries below zero, and the norm of the negative part
divided by the norm of the forecast, both for the raw forecast before any
projection.

## Metrics

As in the confirmation: accuracy on the 1,000-node split for the final two
sync cycles, before the optimiser update, on the model's own forward and on
an exact forward; plus the original endpoint. Paired by seed against the
stored hold and G1, pointwise t intervals, n = 8.

## Pre-written reading (descriptive labels)

Every projected arm is compared, paired by seed on the exact forward window
average, with pointwise 95% intervals, against three references reported
separately: its own raw predictor, hold, and G1. Labels:

- "within band": the point estimate is within 0.5 points. A descriptive
  band, not an equivalence claim.
- "better" / "worse": beyond 0.5 points and beyond 2 SE.
- "inconclusive": beyond 0.5 points but within 2 SE.

How outcomes will be worded:

- A projected arm within the band of hold, or better: projection is
  sufficient to rescue that specified comparison against hold. Nothing more.
- Every projected arm worse than hold: projection alone is insufficient to
  reach hold. This does not show sign errors were unimportant; an arm can
  recover most of a 30-point loss and still sit below hold.
- Beating hold is not the same as closing the gap to G1. Both are reported.
- No percentage of the failure is attributed to any cause.

This is one regime on exposed seeds.

## After this

No further Cora diagnostics. Either Cora is closed as an engine-evaluation
target or one bounded source-versus-atoms generation comparison is designed.
