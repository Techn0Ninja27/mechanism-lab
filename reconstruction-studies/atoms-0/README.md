# atoms-0: mask-channel gating with conditional multiple imputation

Status: completed parent-reviewed pilot evaluation; no novelty claim.

## Mechanism under study

The proposal combines (a) a value pathway where missing feature values are filled by draws from a Gaussian conditional model fit using training rows only and conditioned on every observed input (including always-observed columns 0 and 1), and (b) an explicit binary missingness-indicator pathway. A HistGradientBoostingClassifier scores each indicator by the increase in log loss when that indicator is permuted, separately under four corruption regimes. A feature's mask channel is kept only when its estimated contribution is positive and its coefficient of variation across regimes is below a fixed threshold. The classifier is refit with gated mask channels. Prediction averages probabilities over five conditional draws. No target-batch adaptation is used.

The benchmark's final regimes (own-value MNAR, reversed MNAR, and two block-MCAR masks) are not represented in validation. The proposal's stated counterfactual is therefore not testable with its suggested four validation regimes.

## Prior art checked (primary sources)

- [Robust prediction under missingness shifts](https://arxiv.org/abs/2406.16484) establishes that ignorable shifts leave the Bayes predictor unchanged, while non-ignorable shifts can change it; its empirical results include settings where disregarding missingness helps. It does not establish that a featurewise stability gate generalizes to unseen mechanisms.
- [MIRRAMS: Learning Robust Tabular Models under Unseen Missingness Shifts](https://arxiv.org/abs/2507.08280) uses mutual-information robustness conditions to encourage label-relevant representations under unseen test-time missingness. This is direct prior art for the task of robustness to unseen missingness shifts and is not implemented as a constituent control here because the proposal does not specify a faithful MIRRAMS objective or its required neural model/training details.
- [What's a good imputation to predict with missing values?](https://arxiv.org/abs/2106.00311) studies impute-then-regress and NeuMiss. It shows why imputation quality alone is not predictive optimality and provides a stronger conceptual baseline than the proposal's informal plug-in claim.
- [Nearly Optimal Bayesian Inference for Structural Missingness](https://arxiv.org/abs/2601.18500) motivates integrating missing-value uncertainty rather than relying on one plug-in fill, but studies a Bayesian structural-missingness framework. A pairwise-Gaussian conditional sampler is not a reproduction of that method.
- [Resilient Vision-Tabular Multimodal Learning under Modality Missingness](https://arxiv.org/abs/2605.12031) uses stochastic modality dropout in a vision-tabular architecture. It is related only at the broad augmentation level, not a direct tabular missingness-shift comparator.

These sources make the broad problem and several components prior art. Whether this specific gate is useful or new is not established here.

## Assumptions and identifiable limits

The synthetic fixture keeps clean feature/label structure fixed while changing masks. The two anchors stay observed. The training fixture is already MCAR-masked, and its hidden clean values are never accessed. Pairwise means/covariances are estimated only from observed training values. Gaussianity and a positive-semidefinite projection are modeling assumptions, not fixture truths. Under MNAR, the conditional distribution estimated from observed pairs need not identify the missing-value distribution; a Gaussian draw cannot recover that without assumptions. The gating premise that validation-like mask effects proxy effects under unseen MNAR/reversal/block regimes is not identifiable from validation data.

## Implementation and controls

The validation implementation uses the shared fixture unchanged. Added corruption is applied only to entries observed in the already-masked training inputs. Training augmentation consists of the original view plus additional MCAR, anchor-MAR, and own-value-dependent masks. Train-only inner holdout rows estimate the gate across four regimes (the original MCAR view and three additional-mask views), avoiding using external validation labels to choose per-feature channels. This is a necessary deviation from the proposal's literal validation-based gate because scoring the same validation outcomes used to choose channels would leak the selection criterion into reported validation performance. External validation is used only to record/tune aggregate candidate/control comparisons.

Strong controls/ablations use matched train rows, augmentation views, random seeds, and HistGradientBoosting capacity wherever applicable: native-NaN HGB, augmented native-NaN HGB, the same imputed HGB with all masks, no mask channels, a random gate of matched size, conditional-mean K=1 prediction, unconditional-mean imputation, and no augmentation. The all-mask imputed model is the strongest constituent-method control for isolating the gate. MIRRAMS is described as prior art, not claimed as a reproduced baseline. Training/prediction parameters are fixed in code; no final data are available to the validation runner.

The gate's coefficient of variation is undefined/non-informative when mean permutation importance is nonpositive or near zero. Such channels are dropped. In a featurewise permutation test, importance can be distorted by correlated indicators; this is recorded as a limitation. All predictions are functions of a masked feature matrix only. There is no condition name or label input to prediction. The CLI includes a final mode with a distinct output path and refuses to overwrite an existing final output, but final mode must not be run before explicit parent review.

## Frozen evaluation outcome

After parent review of the source and validation artifact, the frozen final fixture was run once for seeds 9401, 9403, and 9407. The runner scored `final_cases` only; no learner accessed `test_x`, target labels, or condition identifiers. The preregistered strongest control remained `single_conditional_mean`, selected by pooled validation AP before phase two. Validation performance remains a failed result and was not retuned.

Across the four preregistered shifted cases (`mnar_own_feature`, `mnar_reversed_sign`, `block_mcar_2_8`, `block_mcar_8_14`), stable-gate mean AP was 0.88011 and strongest-control mean AP was 0.89937: paired mean delta -0.01926. Per-seed deltas were -0.02798, -0.01906, and -0.01075. The gate beat native-NaN HGB by +0.04082 mean AP, but beat the no-gate/all-mask ablation by only +0.00446, below the preregistered +0.005 threshold. Mean clean AP was 0.91270 for the gate and 0.92119 for the strongest control (delta -0.00849, inside the 0.01 clean-loss margin). The preregistered combined decision is `no_gain`, since the candidate did not improve over the strongest control.

Files: `validation.json` contains all three validation seeds and ablations; `results.json` contains the one final run; `verdict.json` records the fixed decision and limitations; `parent-approved-final.json` records the parent approval and frozen hashes. No atoms are promoted and no novelty is established.
