# C006: stochastic constrained directions → gradient-aware compression

2026-11-16. Both blinded reviewers suggested a bridge between a stochastic
equality-constrained optimizer and an error-bounded FL gradient compressor.
This note is a bounded primary abstract check, not a full-method review or a
claim of novelty. No implementation investment has been made in the pairing.

The optimizer in [Cui et al.](https://arxiv.org/abs/2606.28230) decomposes its
stochastic search direction and uses Fletcher's augmented Lagrangian for step
selection. Its analysis explicitly handles stochastic Jacobian nondegeneracy
and direction perturbations. That makes the statistical character of any added
compression error relevant; tensor shape alone is not the contract.

The compressor in [Ye et al.](https://arxiv.org/abs/2511.05770) exploits
cross-round gradient magnitude history and convolutional-kernel sign structure.
The paper motivates this design by the poor fit of generic scientific-data
predictors to gradients. A generic constrained solver's direction need not have
those statistics or a convolutional layout. The reviewers' suggested reshape
would change representation but would not establish the needed correlations.

Compressed optimization is already an established family:

- [Communication Compression for Distributed Nonconvex Optimization](https://arxiv.org/abs/2201.03930)
  studies primal-dual algorithms under relative and absolute compressor-error
  classes, with convergence guarantees under stated assumptions.
- [TiCoPD](https://arxiv.org/abs/2501.05701) uses compressed difference terms and
  two-timescale primal-dual/consensus updates for decentralized optimization.
- [Q-GADMM / Q-SGADMM](https://arxiv.org/abs/1910.10453) combines quantized
  communicated differences with ADMM-style learning, including a stochastic
  neural-network variant.
- [Distributed constrained online nonconvex optimization with compressed communication](https://arxiv.org/abs/2503.22410)
  treats time-varying inequality constraints and a compressed primal-dual method.

These do not prove the exact Fletcher-plus-EBLC pairing already exists. They do
rule out treating the broad combination of constrained optimization and
compression as a new principle. A defensible next investigation would need to
specify a distributed target problem, what crosses a communication boundary,
the compressor's state and tensor layout, and the error assumptions preserved
by the proposed bridge. It would then need strong compressed-optimization
baselines. None of that follows from the current abstract-level edge, so the
pair remains unpromoted.
