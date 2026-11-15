# tabular_missing-direct-2 — Missingness-Mask Confidence Shrinkage

Study date: 2026-11-16. This is a small synthetic pilot, not evidence of external novelty or statistical significance.

## Proposal and mechanism

The proposal trains median imputation plus a HistGradientBoostingClassifier on original training rows and a 20%-MCAR-masked duplicate. On labeled validation rows it observes the binary missingness mask and absolute probability error, fits a logistic conditional-mean model with 20 mask indicators plus 190 pairwise products, then adjusts a test score as

`p_adjusted = prevalence_train + (p_raw - prevalence_train) * (1 - predicted_absolute_error)`.

The stated predictor inputs to the reliability model are only the mask. It receives no raw observed values or held-out test labels. Its target and the base rate are determined from training/validation data only. Validation labels are permitted for calibration-model fitting under the assigned procedure.

### Specification issue and faithful operationalization

The proposal says “logistic regression” predicts per-row absolute calibration error, which is continuous in [0,1]. A conventional hard-label `sklearn.LogisticRegression` fit cannot accept that target. I implemented the same logistic conditional-mean mapping with fractional Bernoulli cross-entropy (soft labels), L2=1, and coefficient bounds [-20,20]. This retains the proposed mask-only features, outcome, and shrinkage map; it does not discretize the error or use another predictor family. The actual implemented model and bounds are pinned in `experiment.py` and `preregistration.json`.

## Prior art checked

The broad mechanism family is established. Prior work studies calibration under covariate shift, domain adaptation where missingness mechanisms change, and classifiers that use missingness masks. This pilot does not establish that the particular post-hoc relationship—predict absolute error from mask indicators/interactions, then contract probabilities toward a prior—is novel.

- Pampari and Ermon, [Unsupervised Calibration under Covariate Shift](https://arxiv.org/abs/2006.16405) (2020): formalizes calibration under domain shift and an importance-sampling correction. It establishes the broader calibration-under-shift problem; it does not describe this validation-label-trained mask-to-absolute-error shrinkage rule.
- Zhou, Balakrishnan, and Lipton, [Domain Adaptation under Missingness Shift](https://proceedings.mlr.press/v206/zhou23b.html) (AISTATS 2023): studies changes in missingness mechanisms and uses available missingness indicators in adaptation. This makes “missingness pattern as a signal under shift” non-novel as a general premise.
- Rockenschaub et al., [Robust prediction under missingness shifts](https://arxiv.org/abs/2406.16484) (2024): compares predictors with and without mask information and studies robustness to shifts in missingness rates/mechanisms. It concerns prediction robustness and training estimators, rather than this post-hoc error predictor.
- Dong, Jiang, and Yang, [Expectation Consistency Loss: Rethink Confidence Calibration under Covariate Shift](https://proceedings.mlr.press/v306/dong26e.html) (ICML 2026): current primary work on confidence calibration under covariate shift, using target-domain expectation consistency. It is a distinct mechanism and is relevant evidence that calibration under shift is an active, populated area.

These are primary papers/proceedings. Search was targeted, not exhaustive. Related components are prior art; novelty remains unverified and no claim of novelty is made.

## Implementation and reproducibility

Data: `make_classification` with 5,000 rows, 20 features, 5 informative, 5 redundant, class weights 80/20, class separation 1.0. Seeds are 137, 271, and 389. Each gets stratified 60/20/20 indices using the recorded two-stage split. Imputation is fit only on training rows. The baseline/control is the same augmented HGB without shrinkage. Pairwise interaction removal is the component ablation.

The reliability model is fit on validation MCAR corruption at 30% and 60%, exactly the candidate description’s two proposed stress rates. Primary validation and test metrics follow the assigned common protocol: clean plus 10%, 30%, and 50% MCAR, with fixed seed-derived corruption draws. Candidate and control see identical draws. Test indices are used only by the single frozen `final` invocation; there was no post-test tuning.

From `experiments/alien-science/upstream`:

```sh
../.venv-benchmark/bin/python scripts/with_compute_slot.py -- ../.venv-benchmark/bin/python candidate-studies/tabular_missing-direct-2/experiment.py validation --output candidate-studies/tabular_missing-direct-2/validation.json
../.venv-benchmark/bin/python scripts/with_compute_slot.py -- ../.venv-benchmark/bin/python candidate-studies/tabular_missing-direct-2/experiment.py final --output candidate-studies/tabular_missing-direct-2/results.json
```

The final output refuses to overwrite an existing `results.json`. `preregistration.json` records the validation-only decision, frozen rule, source SHA256, and split hashes. The validation results must be regenerated before preregistration if the source changes.

The complete 3-seed validation stage took about 0.93 seconds; final took about 0.60 seconds in this environment, with one CPU thread. HGB fit time was about 0.20 seconds per seed. Inference time was not separately instrumented; the total stage runtime includes training, calibration fitting, prediction, and scoring.

## Results

Primary metric is mean average precision across clean, 10%, 30%, and 50% dropout. Per-seed values and all condition-level scores are in `results.json`.

| Seed | Control mean AP | Shrinkage mean AP | Paired change |
|---:|---:|---:|---:|
| 137 | 0.744351 | 0.743810 | -0.000540 |
| 271 | 0.893747 | 0.894369 | +0.000621 |
| 389 | 0.901837 | 0.902422 | +0.000586 |
| Mean | 0.846645 | 0.846867 | +0.000222 |

The frozen pilot rule required positive average paired AP change and no seed clean AP loss greater than 0.02. Both conditions pass: the clean condition is unchanged by the strictly mask-only correction when no features are missing, and the mean AP change is positive but very small. Thus the mechanical pilot verdict is `supported_pilot`, not a claim of reproducible benefit.

At 50% missingness, the standard deviation of probabilities among misclassified samples fell for all seeds (137: 0.2724 to 0.2242; 271: 0.2999 to 0.2676; 389: 0.2474 to 0.2129). But this secondary result does not imply improved ranking: AP varied by seed and the average Brier score was marginally worse overall (0.064518 vs 0.064486 control). Removing pairwise interactions yielded similar validation AP, suggesting the interaction component was not clearly necessary in this pilot.

## Verdict and smallest falsification

`verdict.json` contains the machine-readable result. The mechanism is implementable and received a tiny positive mean AP change under its frozen pilot rule, but effects are mixed by seed and smaller than a thousandth AP. This is inconclusive evidence about usefulness beyond the synthetic setup. The smallest attempted falsification—the predeclared 50% dropout error-subset spread check—did not falsify the shrinkage behavior; the primary AP outcome itself is close to no change. Brier score does not support a broad calibration improvement claim.

Limitations: only three synthetic seeds; MCAR masks, class prior stability, and mask-pattern transfer are assumed; no external dataset or MNAR/structured missingness was tested; and the targeted prior-art scan cannot support a novelty claim. Human adjudication is still needed before promotion.
