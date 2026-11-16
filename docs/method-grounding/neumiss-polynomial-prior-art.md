# Prior-art check: grounded-02 NeuMiss shared operator

Date: 2026-11-16

The proposal in `results/improvement-phase-2026-11-16/grounded-02/result.json` asks whether NeuMiss's untied masked recurrence can be replaced by one shared matrix plus learned scalar coefficients, at equal matrix-vector cost. The proposal itself already flags substantial overlap. This review checked the full primary paper and the authors' later official implementation before treating the question as a research candidate.

## Direct overlap found

The strongest result is direct and changes the status of the proposal. The NeurIPS 2021 paper *What's a good imputation to predict with missing values?* says that the authors made a minor improvement over the original NeuMiss architecture: the theory points to shared weights and residual connections, while the 2020 paper used neither. The paper's architecture diagram and displayed recurrence use repeated applications of one shared matrix, with residual additions. The official `marineLM/NeuMiss_sota` repository states that its implementation is the latest version “with shared weights” from that second paper.

The pinned implementation inspected at commit `7902b8dbe7114e8dc3010b5e8b132c35806a9d74` makes this explicit in `neumiss/NeuMissBlock.py:52-66`: there is one `Linear(n_features, n_features)` object, and `Sequential(*(layer*self.depth))` repeats that same linear layer, mask nonlinearity, and skip connection. Thus “replace untied NeuMiss matrices with one shared matrix” is already published and implemented. It should not be presented as a new architecture or as an untested gap in NeuMiss.

The 2021 paper is also a primary source for the conceptual reason: the shared matrix is the learned analogue of the Neumann iteration, and the residual connection restores the additive series structure. This matters for grounded-02 because the proposal's suggested recurrence initially added a residual term at step 1, while the 2020 untied implementation does not. The reviewer correctly required `c_1=0` (or an explicitly separate first step) for a tying-only comparison.

## What remains distinct, but weakly

The exact new element in grounded-02 is not sharing itself. It is adding learned scalar coefficients `c_k` to the repeated shared operator and testing a single global polynomial approximation against the exact mask-specific inverse. I did not find this exact scalar-coefficient ablation in the NeuMiss papers or the official repositories inspected. That absence is not evidence of novelty. It is a narrow engineering comparison inside an already known shared-weight NeuMiss block.

There is broad prior art for the surrounding idea. Deep unfolding literature explicitly distinguishes tied parameters across iterations from untied parameters and discusses untying as a capacity/optimization choice. Learned iterative-recovery work analyzes intermediate degrees of weight sharing, including recurrent/tied and feedforward/untied endpoints. Krylov, Neumann, Chebyshev, and learned polynomial iterations all make polynomial approximation of linear solves a standard construction. Therefore the scalar-coefficient variant should be labelled “unverified operator ablation” unless a substantially new theorem, constraint, or transfer result emerges.

## Exact comparison contract

If the team runs this at all, the comparison must be framed as a diagnostic, not a novelty search:

1. Reproduce the 2021 shared-weight NeuMiss block with its residual recurrence as the main published baseline.
2. Compare a scalar-coefficient shared variant against that baseline, the original 2020 untied implementation, and a fixed-coefficient tied recurrence.
3. Keep the first step structurally matched. If the published recurrence adds a residual at every step, the scalar variant must use the same first-step convention; if comparing to the 2020 code, set `c_1=0` or report the first-step change as a second operator.
4. Fit one coefficient vector across masks in the oracle check. Per-mask polynomial fits would overstate what the learned model can represent.
5. Report the parameter-count difference and the non-identifiability of a common scale: a global scale of all `c_k` can be absorbed by the output mixing matrix `Wc`.
6. Use fresh development covariance draws and separate untouched transfer draws. The prior Mechanism Lab three-seed fixture is not valid confirmatory data for this question.

The cheapest kill test is algebraic: use fresh Gaussian covariance draws, compare the best global scalar polynomial's solve error to the fixed-coefficient tied recurrence, and stop if the improvement is under the preregistered 10% gate. Only if that gate passes should training be attempted. A positive result would support a resource/approximation tradeoff study, not a new NeuMiss family.

## Primary sources checked

- Le Morvan et al., *NeuMiss networks* (NeurIPS 2020), [paper](https://arxiv.org/abs/2007.01627) and [original code](https://github.com/marineLM/NeuMiss).
- Le Morvan et al., *What's a good imputation to predict with missing values?* (NeurIPS 2021), [paper](https://arxiv.org/abs/2106.00311), [official implementation](https://github.com/marineLM/NeuMiss_sota), and [official source file](https://github.com/marineLM/NeuMiss_sota/blob/7902b8dbe7114e8dc3010b5e8b132c35806a9d74/neumiss/NeuMissBlock.py).
- Hershey, Le Roux & Weninger, *Deep Unfolding: Model-Based Inspiration of Novel Deep Architectures* (2014), [arXiv](https://arxiv.org/abs/1409.2574), for the tied/untied unfolding distinction.
- Joukovsky et al., *Generalization Error Bounds for Iterative Recovery Algorithms Unfolded as Neural Networks* (2021), [PMLR](https://proceedings.mlr.press/v161/joukovsky21a.html), for analysis spanning multiple degrees of weight sharing.

Decision: **the sharing premise is covered prior art; the scalar-coefficient diagnostic remains a narrow, unverified ablation with no novelty established.**
