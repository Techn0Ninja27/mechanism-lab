# Pre-extraction source audit — 2026-11-17

Inspected the local primary-source excerpts before any model extraction. This note is not supplied to the extractor or reviewer. It is a development audit, not a blinded gold standard or representative benchmark.

- GradLite, `2510.22467v1:chunk:0001`: Algorithm 1 estimates a residual but does not define the estimator. The equations use the exact gradient difference. Basis construction/adaptation, accumulator initialization, and activation-cache representation are underspecified. A faithful draft must preserve this distinction and must not infer executable memory savings or unbiasedness from the paper claiming them.
- PacTrain, `2505.18563v1:chunk:0001–0002`: the mask-stability branch communicates `model_update`; do not silently substitute a gradient-only interface. Preserve stable compression versus full AllReduce and unknown tracker stability/packing details. Pruned-model prerequisites and the dense collective representation need attention.
- PERA, `2604.11841v1:chunk:0001–0002`: B uses the ordinary second-order expansion; A uses a learned-coefficient expansion. Preserve this asymmetry, dimensions, coefficients and relevant initialization assumptions. Do not describe both expansions as identical elementwise squaring. The third chunk adds an expanded sum and limitations.

The named tasks were corrected before extraction to avoid introducing these mistakes in the request. Audit details remain separate from the supplied task names. Agreement with these checks would demonstrate limited source fidelity only, not novelty, usefulness or executable compatibility.
