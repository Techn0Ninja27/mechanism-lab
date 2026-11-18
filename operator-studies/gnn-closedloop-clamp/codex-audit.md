# Codex audit of the non-negative projection control

Date: 2026-11-18. Read-only audit; no clamp training was rerun and no stored
result was modified.

## Integrity and arithmetic

The raw result has 40 rows: exactly 8 exposed seeds (50–57) × 5 arms, with
unique and complete `(seed, arm)` keys. Every frozen input hash matches the
current files, including the confirmation raw result and freeze. The raw freeze
hash and summary bindings match. All required metrics are finite, and all raw
forecast aggregates report 93 stale epochs per hidden layer.

I independently compared the stored unprojected `sni` and `lag1` rows against
the corresponding confirmation rows. For every seed, the full window,
post-update exact validation and evaluation accuracies, and final loss match
exactly. Thus the baseline reproduction gate is stronger than just comparing
the endpoint evaluation field; the README's separately checked validation
shortfall is also borne out by the raw records.

Recomputing the paired exact-forward window means and standard errors gives the
summary values exactly. With 8 paired seeds, the pointwise 95% Student-t
quantile is 2.3646243. The exact-forward differences versus hold are:

| arm | mean points | SE | 95% interval | seeds positive |
|---|---:|---:|---|---:|
| sni | -37.41 | 2.10 | [-42.37, -32.45] | 0/8 |
| sni_emit | -33.01 | 2.92 | [-39.90, -26.11] | 0/8 |
| sni_state | -34.24 | 2.49 | [-40.14, -28.34] | 0/8 |
| lag1 | -32.51 | 1.80 | [-36.75, -28.26] | 0/8 |
| lag1_emit | -32.16 | 1.58 | [-35.90, -28.42] | 0/8 |

The projection variants therefore remain far below hold in this exposed
regime. The positive sni_emit-versus-raw comparison is a descriptive control
contrast; it does not establish that negative values caused the original
failure.

## Semantics and limits

The tests check that projected emission and projected recurrence state differ,
that `sni_emit` preserves the raw recurrence state, that exact values are not
read between syncs, and that logging describes raw forecasts rather than
projected emissions. The raw file intentionally stores only aggregate
negativity statistics over the 93 stale epochs, not the per-epoch series
promised by the plan. This is disclosed and means that a per-step temporal
analysis cannot be reconstructed without a new run.

The two projection modes are well separated: `sni_emit` clamps only what the
model receives, while `sni_state` feeds the clamped value back into the SNI
state. `lag1_emit` clamps only emitted values. These are valid controls for the
specified one-regime question. They do not make a general statement about
constrained forecasting.

All seeds were already exposed by the confirmation study. The partitioning,
single graph, detached remote-gradient emulation, three-layer initialization,
and exact-forward evaluation remain the confirmation study's limitations.
This is not held-out evidence and does not validate an engine-generated method.

## Disposition

The clamp study is internally consistent and supports a narrow negative result:
non-negative projection alone does not rescue the SNI/lag1 implementations to
the hold baseline. The stored aggregate logging shortfall is material but
explicit. No sign-causality claim, downstream generalization claim, or paid
proposal-generation decision should be based on this study alone.
