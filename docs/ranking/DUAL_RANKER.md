# Dual-ranker scaffold

`generation.dual_ranker.rank_and_select` separates discovery from execution.
Discovery contains a nullable novelty estimate, evidence status, uncertainty,
and source consistency. Execution contains nullable feasibility and expected
gain plus the metric and baseline that make gain meaningful.

Novelty, uncertainty, and feasibility are validated as finite values in
`[0, 1]`; expected gain may be any finite real value. A row is exploit-ready
only when all three ranking axes, source consistency, metric, baseline, and
budget are present. An evidence-status label is metadata, not a substitute
for a novelty estimate.

Known-invalid and source-inconsistent rows are gated. A feasibility threshold
only gates a known value; missing feasibility is not silently turned into zero.
Pareto dominance uses novelty, feasibility, and expected gain, but expected
gain participates only for the same problem family/metric/baseline/budget.
Rows from different utility contexts are incomparable rather than being
ranked by an invented common scale.

The selector is deterministic for a seed, independent of input order, bounded
by `count`, and returns a rationale. An explicit exploration quota first covers
distinct problem families. Rows with unknown axes may be selected for that
exploration and are marked `exploration=True`, with
`promoted_confidence=False`; missingness never improves a confidence score.

This module is intentionally not a novelty scorer, source verifier, model
repairer, or training utility. Those upstream components must provide the row
fields and their provenance. Run the focused checks with:

```sh
pytest -q tests/test_dual_ranker.py
```
