# molecules-1 — reconstruction protocol

## Status and scope

This is a preregistered reconstruction of the proposal in `proposal.json`, not a novelty claim or a new empirical finding. The fixed shared fixture supplies 3,000 training rows with MCAR 20% missingness on columns 2–19, and 1,000 validation rows in each of `clean`, `mcar_0.2`, `mcar_0.4`, and `mar_anchor_x0`. Validation does **not** include MNAR or reversed MNAR. The proposal's defining claim about shortcut sign reversal under MNAR therefore cannot be established in phase one; validation can only check utility and collateral effects on the available cases.

## Mechanism implemented

The proposed intervention retains each observed value and can only add missingness. On training rows, estimate an additional-mask rate `g(a)` from the training mask alone, in five quantile bins of always-observed anchors `a=(x0,x1)`. For each of three copies, draw a rowwise multiplier `u ~ Uniform(0.5, 2)` and independently hide each currently observed non-anchor value with probability `clip(g(a)*u, 0, 1)`. Keep the label unchanged and stack the original rows with the three remasked copies. On prediction, apply eight independently drawn extra masks and average the eight HGB probabilities. This is the literal value-preserving, anchor-rate-conditioned remasking idea; it is not a learned invariance penalty and does not estimate a missing-value posterior.

The training rate estimator is constrained by the fixture: training masks are MCAR, so true anchor dependence is absent. Any fitted bin differences are sampling noise. This means the procedure is a stochastic augmentation with a noisy anchor-rate schedule, not evidence that an anchor-conditional mechanism was learned. Also, the final fixture's MNAR mechanism depends on the missing feature itself, information unavailable at inference when that feature is already missing. The proposal's stability assumption is not identifiable from these training inputs alone.

## Controls and ablations

All HGB conditions use the same 4x training-row count and fixed estimator settings. Controls include the strongest named constituent, mean imputation plus explicit missing indicators followed by HGB; native-NaN HGB with duplicate rows; uniform random-remask augmentation; and the proposal with uniform rather than anchor-binned rates. Ablations remove inference averaging (`K=1`), training remasking, or anchor conditioning (permuted anchors). Candidate and controls see no validation labels during training or model selection. No target-batch adaptation is used. Prediction methods receive only masked feature matrices; condition names and labels are used only outside prediction to calculate metrics.

## Prior work checked (primary sources)

- [Robust prediction under missingness shifts](https://arxiv.org/abs/2406.16484v1) studies missingness-shift robustness and distinguishes ignorable shifts, where the Bayes predictor is invariant, from non-ignorable shifts, where it may change. It also reports that ignoring missingness can help when missingness is highly informative. This is directly relevant prior art for the proposal's motivation.
- [MIRRAMS: Learning Robust Tabular Models under Unseen Missingness Shifts](https://arxiv.org/abs/2507.08280) proposes MI-robustness conditions for label-relevant information under unseen missingness shifts, covering MCAR/MAR/MNAR without a fixed mechanism assumption. Its neural objective differs from this proposed tree/remasking recipe, but the robustness objective substantially overlaps.
- [No Data? No Problem: Robust Vision-Tabular Learning with Missing Values (RoVTL)](https://arxiv.org/abs/2512.19602) uses tabular attribute missingness as training augmentation. It is not a same-task tabular tree method, but establishes masking augmentation as prior art.
- [What's a good imputation to predict with missing values?](https://arxiv.org/abs/2106.00311) analyzes impute-then-regress and motivates the matched mean-imputation-plus-indicator control. Its theory does not establish that this finite-sample HGB control is optimal.
- [scikit-learn HistGradientBoosting missing-value documentation](https://scikit-learn.org/stable/modules/ensemble.html#missing-values-support) documents native NaN routing at each split, including routing missing values according to learned split gains. [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754v3) is related primary work on sparsity-aware tree boosting. These are baseline behaviors, not evidence for shift robustness.
- [Domain Adaptation under Missingness Shift](https://proceedings.mlr.press/v206/zhou23b/zhou23b.pdf) formalizes shifts in missingness patterns and studies adaptation under its stated assumptions.

This scan finds substantial overlap in both motivation and ingredients. Whether this specific combination has prior art is unresolved; no novelty is asserted.

## Assumptions, safeguards, and deviations

- Anchors 0 and 1 are always observed by the fixture. No claim is made that their relation to labels is invariant in a real deployment.
- Training remasking preserves observed values and labels, but can discard feature information; this is augmentation, not a causal intervention on the data-generating process.
- The proposal specified validation early stopping and 10 seeds. The shared task instead mandates three fixed seeds and prohibits tuning on validation labels; estimator capacity is fixed in advance (`max_iter=80`, `max_leaf_nodes=15`, `min_samples_leaf=20`, `learning_rate=0.08`, no early stopping).
- The proposal expected MNAR validation. The available validation fixture has only MCAR/MAR cases; MNAR reversal and block missingness are reserved for the final fixture. Thus the validation result cannot test the core claimed shortcut reversal. This limitation will be preserved in interpretation.
- All RNGs are locally seeded. One CPU thread is requested. No new dependency, paid API, private fixture helper, clean hidden training value, test array, or target label is used.

## Run protocol

Validation only:

```sh
../.venv-benchmark/bin/python scripts/with_compute_slot.py -- ../.venv-benchmark/bin/python reconstruction-studies/molecules-1/experiment.py --mode validation --output reconstruction-studies/molecules-1/validation.json
```

The script has distinct `validation` and `final` modes. Final mode checks the parent-approved record, current code hash, fixture hash, and validation artifact hash before loading the final fixture; it refuses to replace an existing final output. Phase two was run once for each of the three fixed seeds after parent approval. The final evaluator scored only the six entries in `fixture.final_cases`; it did not access `test_x` or pass condition names or labels to prediction.

Primary metric is mean AP across the four available validation cases, with mean AP also reported over the shifted cases (`mcar_0.4`, `mar_anchor_x0`); Brier and clean AP are secondary. The predeclared minimum useful effect is +0.005 mean AP over the strongest matched control with no clean AP decrease exceeding 0.01. The task's core MNAR claim remains unvalidated here.


## Final evaluation

The preselected candidate `anchor_full` had mean AP 0.8406 across the six final cases, versus 0.8556 for the phase-one strongest matched control `native_nan_duplicate` (paired difference −0.0150). The other required strong control, `anchor_train_only`, scored 0.8595 (candidate difference −0.0189). On clean cases, candidate AP was 0.8868 versus 0.9089 and 0.9018 respectively. Candidate mean AP across MNAR and block cases was 0.8262, below both controls (0.8383 and 0.8459). The block-missingness result is mixed: candidate beats native HGB on `block_mcar_2_8`, but loses to both controls on `block_mcar_8_14`. These results do not support the proposed utility threshold; phase-one failure remains a failure. See `results.json` for per-seed/per-condition AP, Brier, runtimes, hashes and errors, and `verdict.json` for the frozen decision.
