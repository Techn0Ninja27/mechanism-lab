# Mirrored missingness augmentation with explicit masks

## Question and mechanism

This candidate tests whether training one histogram gradient boosting classifier on the original training rows plus four masked copies improves average precision when deployment has higher MCAR missingness. Every corrupted copy draws one rate per feature from `Beta(0.5, 2)` and then masks cells independently at that feature rate. Training inputs append 20 binary missingness columns. Evaluation uses the same clean and 10%, 30%, and 50% MCAR input draws for every method. The classifier is `HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, random_state=42)`.

The proposal's rationale is that masked copies expose the learner to feature absence patterns while the appended columns preserve the exact mask. The explicit-mask claim is falsifiable separately from the augmentation claim: this study permutes the mask columns across evaluation rows while leaving NaN positions fixed.

## Prior art checked

- [Le Morvan and Varoquaux, “Imputation for Prediction: Beware of Diminishing Returns” (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/file/65e837e76a5308df3d5544aab6196e21-Paper-Conference.pdf) directly studies appending missingness indicators as complementary features, including under MCAR, and reports that indicators can improve prediction. Its experiments also find the benefit of the mask smaller for XGBoost than for neural predictors. It is close prior art for the explicit mask component; it does not, from the inspected paper, establish this particular four-copy, per-feature-Beta training scheme.
- [Zhu et al., “Prediction with Incomplete Data under Agnostic Mask Distribution Shift” (arXiv:2305.11197)](https://arxiv.org/abs/2305.11197) directly formulates changing missingness-mask distributions at deployment. StableMiss jointly approximates mask-conditional predictors and uses decorrelation. This is close prior art for the deployment problem, though its mechanism differs from augmentation plus HGB.
- [Richardson-SGD with Missing Data (arXiv:2605.19641)](https://arxiv.org/abs/2605.19641) deliberately adds missingness to reduce gradient bias using Richardson extrapolation. It is a related use of synthetic additional missingness, but it studies gradient correction for parametric losses, not the proposed tree training stack.
- The [scikit-learn HGB missing-value documentation](https://scikit-learn.org/stable/modules/ensemble.html#histogram-based-gradient-boosting) says the tree learner learns which child receives NaNs at each split and can split based on missingness. That makes a native-NaN HGB comparison essential and makes explicit mask columns plausibly redundant for this model.

The search did not establish priority or novelty. Missingness indicators, missing-pattern shift, synthetic masking, and native missing-value routing are all established components. Any novelty claim about their exact combination remains unverified.

## Implementation and reproducibility

The implementation is in [experiment.py](experiment.py). It uses the prescribed synthetic dataset and three seeds (137, 271, 389), stratified 60/20/20 splits, training-only augmentation, shared per-seed MCAR evaluation masks, and one CPU thread. The controls and ablations are:

- `median_hgb`: the proposal's median-imputation plus HGB baseline.
- `native_hgb`: closest simple alternative, HGB trained on complete rows and using native NaN routing at evaluation.
- `mirror_mask`: candidate intervention.
- `mirror_no_mask`: same augmented copies and same NaN placement, without explicit mask columns.
- `mask_only`: ablation that replaces observed numeric values by train medians and tests whether mask patterns alone carry signal.

Run from `experiments/alien-science/upstream`:

```bash
../.venv-benchmark/bin/python scripts/with_compute_slot.py -- ../.venv-benchmark/bin/python candidate-studies/tabular_missing-random_seed-1/experiment.py validation --output candidate-studies/tabular_missing-random_seed-1/validation.json
../.venv-benchmark/bin/python scripts/with_compute_slot.py -- ../.venv-benchmark/bin/python candidate-studies/tabular_missing-random_seed-1/experiment.py final --output candidate-studies/tabular_missing-random_seed-1/results.json
```

Final mode requires the existing preregistration and refuses to overwrite `results.json` unless `--rerun-final` is explicitly passed. The frozen settings, validation summaries, split hashes, and source hash are recorded in [preregistration.json](preregistration.json). Full validation and held-out condition-level results, timing, and split hashes are in [validation.json](validation.json) and [results.json](results.json).

## Results

Primary metric is the equal-weight mean average precision over the four conditions, averaged across the three seeds. Held-out test results:

| Method | Seed 137 mean AP | Seed 271 mean AP | Seed 389 mean AP | All seed-condition mean AP | Mean AP at 50% MCAR |
|---|---:|---:|---:|---:|---:|
| Median-impute HGB | 0.7235 | 0.8424 | 0.7947 | 0.7869 | 0.6019 |
| Native-NaN HGB | 0.7349 | 0.8545 | 0.8432 | 0.8109 | 0.6530 |
| Mirrored augmentation + masks | 0.7564 | 0.9198 | 0.9022 | 0.8595 | 0.7633 |
| Mirrored augmentation, no masks | 0.7564 | 0.9198 | 0.9022 | 0.8595 | 0.7633 |
| Mask-only ablation | 0.2060 | 0.2055 | 0.2064 | 0.2060 | 0.2076 |

Paired primary deltas (`mirror_mask` minus `median_hgb`) were +0.0329, +0.0775, and +0.1074 by seed; mean +0.0726. At 50% MCAR the paired deltas were +0.0906, +0.1682, and +0.2255; mean +0.1614. Against native-NaN HGB, the mean paired delta was +0.0486. Thus the preregistered pilot performance rule passed.

The explicit-mask pathway did not pass its diagnostic: permuting the appended masks caused exactly 0.0000 AP change for each held-out seed at 50% MCAR. The no-mask ablation matched the candidate's scores at every recorded condition. In this HGB setup, the observed gain is attributable to exposure to masked training copies, not evidence that the additional indicator columns contribute. Mask-only performance near the class prevalence further argues against mask-pattern signal alone explaining the result.

Mean fit time was 1.376 s for `mirror_mask`, 0.787 s for `mirror_no_mask`, 0.344 s for native HGB, and 0.336 s for median-impute HGB. Mean prediction time over all four conditions was 0.136 s, 0.066 s, 0.076 s, and 0.063 s respectively. All experiments completed well within the stated compute budget. These are three-seed exploratory results, not a significance test or external validation.

## Verdict and smallest falsification test

**Supported pilot for augmentation; explicit-mask mechanism unsupported.** The preregistered held-out improvement criterion passed versus both baseline and native-HGB alternative. But the bundled explicit-mask claim is redundant in this implementation: removing or permuting the mask columns changed no scores. The proposal's specific mask rationale is falsified here; its augmentation result merits replication with more seeds or real shifted datasets before any broader claim.

The cheapest follow-up that could reverse the performance conclusion is to rerun the same paired three-seed benchmark with an independently chosen data seed set and compare `mirror_no_mask` against native-NaN HGB. A zero or negative paired gain would remove the remaining evidence for augmentation. To test explicit masks, use a learner that does not already encode missingness through native NaN routing and hold all other inputs and masks fixed.
