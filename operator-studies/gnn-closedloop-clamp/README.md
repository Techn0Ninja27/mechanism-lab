# Non-negative projection control: results

Date: 2026-11-18. Author: Claude; requested and reviewed by Codex.
Plan: [PLAN.md](PLAN.md). 40 runs, frozen by [freeze.json](freeze.json) before
running; all seven integrity checks passed, including that the logged reruns
of sni and lag1 reproduce the stored confirmation records exactly. Six tests.
Reproduce from the engine root with the pinned requirements and existing local Cora data:

```sh
../.venv/bin/python operator-studies/gnn-closedloop-clamp/reproduce.py --output /tmp/cora-clamp-fresh --workers 4
```

Use a new destination; all frozen inputs are verified and copied into it before
training. `--prepare-only` stages inputs without training.

One regime (3 layers, 4 partitions, constant learning rate, sync 16) on seeds
50–57, which were already exposed. Development evidence only.

## Result

Accuracy on the 1,000-node split, exact forward, final two sync cycles.
Differences in points, paired, mean ± SE (seeds positive). References from
the stored confirmation run: G1 0.764, hold 0.737.

| Arm | Accuracy | − hold | − G1 | − its raw predictor | Negative entries, layer 1 / 2 | Negative norm share, layer 1 / 2 |
|---|---:|---:|---:|---:|---:|---:|
| sni | 0.363 | -37.4 ± 2.1 (0/8) | -40.1 ± 2.3 (0/8) | – | 18% / 22% | 17% / 21% |
| sni_emit | 0.407 | -33.0 ± 2.9 (0/8) | -35.7 ± 3.5 (0/8) | +4.4 ± 2.1 (6/8) | 18% / 23% | 17% / 21% |
| sni_state | 0.395 | -34.2 ± 2.5 (0/8) | -37.0 ± 3.1 (0/8) | +3.2 ± 1.7 (6/8) | 21% / 26% | 2% / 3% |
| lag1 | 0.412 | -32.5 ± 1.8 (0/8) | -35.2 ± 1.6 (0/8) | – | 6% / 11% | 2% / 3% |
| lag1_emit | 0.416 | -32.2 ± 1.6 (0/8) | -34.9 ± 1.4 (0/8) | +0.3 ± 0.2 (5/8) | 5% / 10% | 2% / 3% |

`sni_emit` projects only the emitted value; `sni_state` also self-feeds the
projected value. Negativity columns describe the raw forecast before any
projection.

## Reading

- **Projection alone is insufficient to reach hold.** Every projected arm is
  still 32 to 34 points below hold, in all eight seeds.
- **Projection helps the SNI-style recurrence a little**: +4.4 ± 2.1 points
  over its raw form when only the output is projected, +3.2 ± 1.7
  (inconclusive) when the state is also projected. For lag1 the change is
  +0.35 points, inside the 0.5-point band.
- **lag1 fails with very little negative mass**: 5 to 10% of entries are
  negative, carrying 2 to 3% of the forecast norm, yet it sits 32 points
  below hold. Whatever breaks lag1 here, sign violations are not a
  sufficient description of it.
- About a fifth of SNI-style forecast entries are negative.

This does not show sign errors are unimportant in general, and it attributes
no share of the failure to any cause. It is one regime on exposed seeds.

## Two disclosed shortfalls against the plan

Found by Codex's code review after the freeze; frozen files were not changed.

- The plan promised negativity logged per stale epoch and layer. The raw
  file keeps only per-run aggregates (mean and maximum share and norm ratio
  over the 93 stale epochs, per layer). The per-epoch series was computed and
  discarded. Recovering it needs a rerun under a new freeze.
- The reproduction gate compared endpoint evaluation accuracy, final training
  loss and the full window, but not endpoint validation accuracy. Checked
  separately afterwards: all 16 logged reruns match the stored endpoint
  validation accuracy exactly.

## Consequence for the benchmark

In this regime hold remains the baseline to beat, and the gap from hold to
G1 (2.7 ± 1.2 points, passing the allocation rule but not 95%) is what a
better estimator could act on. The simple extrapolators, projected or not,
are far below hold. No further Cora diagnostics are planned.


[Codex audit](codex-audit.md) verifies the 40 job keys, frozen hashes, stored
baseline equality and paired statistics without repeating training.
