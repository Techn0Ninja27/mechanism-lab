# Executable candidate groundwork

2026-11-18. This is an experimental execution target for research proposals,
not evidence that the generator produces useful or novel methods. No model
calls were used to build or test these modules.

`generation/causal_program.py` accepts a JSON program with `predict` and `update`
expressions. The allowed variables contain only observations and local state:
`last_sync`, `previous_sync`, `memory`, `age`, and `observed_interval`.
Arithmetic, safe division, ReLU, tanh, clipping and per-row norm/mean can be
composed; Python evaluation, indexing, external tools and hidden truth cannot.
Each expression is limited to 32 nodes and depth six. Compiled trees are
immutable. Both expressions read the same pre-update memory snapshot.

`observe(value, t, program)` receives actual observations only at scheduled
sync/warmup times. `predict(t, program)` advances local state without a hidden
value argument. Failed updates leave state unchanged. Tensor shape, time,
finite-value and autograd-detachment checks are enforced by the runtime.

`scripts/evaluate_causal_candidates.py` connects these programs to the actual
three-layer GCN training loop. Exact remote values are exposed only at sync or
warmup; every hidden layer has its own state. The evaluator measures own-forward
and exact-forward accuracy on the same weights before optimizer updates.
Non-finite training failures retain their candidate, call and arm metadata;
they receive no invented accuracy.

The hold program was checked against the existing confirmation implementation
for a complete 120-step run on non-measurement seed 98: final loss, final exact
accuracy and every recorded window accuracy match exactly. This verifies the
adapter for that control, not arbitrary generated programs.

## Current boundary

Only offline planning and a seed-98 smoke command are enabled. The development
selection API deliberately raises `NotImplementedError` until a complete,
hash-bound campaign manifest exists. Final evaluation is disabled. An initial
prototype accidentally minimized accuracy after switching from an error metric;
review removed that selection path before any generated candidate was evaluated.
A later selector must aggregate the declared accuracy across complete development
seed sets, maximize it, and freeze selections before opening final seeds.

`scripts/prepare_matched_representations.py` separately builds task-blind
extraction prompts from the existing five-source packet. It validates a fixed
12-atom representation and five freeform source distillations against exact
source/chunk quotes, preserving canonical source identities and unknowns.
It does not call a model. Quote matching does not validate the interpretation.
The packet has one full-method excerpt and four abstract-level excerpts.

Run the focused offline checks from the engine root:

```sh
../.venv/bin/python -m unittest tests.test_causal_program tests.test_causal_candidate_evaluator tests.test_matched_representations
```

Planned comparison: task-only, raw sources, typed atoms, freeform distillation,
and a local random-program control. Repeated requests and source preparation
must share one bounded campaign budget. The proposed campaign ceiling is $6
including preparation, within the existing $20 phase cap. No paid execution is
enabled by these modules, and the campaign is not yet frozen.
