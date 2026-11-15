# Mechanism preflight checks

`generation.mechanism_preflight` runs cheap algebraic and observability checks
before an experiment spends training compute. It is motivated by recurring
failure patterns in recent mechanism studies:

- positive constant feature scaling disappears after a train-fitted
  standardizer, while zero scales still mask a feature;
- a mask-clustering mechanism cannot identify `k` groups when fewer than `k`
  distinct mask rows occur;
- anchor/volatile reconstruction needs observed predictors distinct from the missing target; disjoint feature sets are valid, and predictive relationships still need data;
- removing an explicit mask does not remove the representation of missingness if native
  NaN routing or mask-gated selection still observes it. This does not prove the fitted predictor actually uses that information.

Each finding includes an evidence explanation, severity, whether the proposed
mechanism is unsupported by the declared setup, and the smallest follow-up
test. These checks do not prove causality, novelty, or usefulness. A passing
preflight only means that this narrow contradiction was not detected.

Run them before training and retain the returned JSON beside the experiment
configuration. Treat an `error` finding as a design repair, not as a negative
result about the underlying scientific idea.

The first draft incorrectly required anchor and target feature sets to overlap; review caught this before use. A numeric regression counterexample now tests that disjoint predictors can reconstruct another feature. These checks must not become an unvalidated automatic rejection oracle.
