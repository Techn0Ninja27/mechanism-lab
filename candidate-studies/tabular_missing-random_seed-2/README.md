# Anchor-Conditioned Ridge Reconstruction: candidate study

Date: 2026-11-16. Candidate: `tabular_missing-random_seed-2`.

## Proposal and causal mechanism

The candidate proposes to estimate per-feature missingness rates from stress-masked training rows; features below 20% missing are stable anchors, while the rest are volatile. For each volatile feature, it fits Ridge regression on the anchor columns using complete training examples. At prediction time, missing volatile entries are replaced by predictions when all anchors are observed. HistGradientBoostingClassifier (HGB) handles remaining NaNs. The claimed causal path is local, sample-specific reconstruction from stable correlated features, reducing input degradation versus median imputation.

## Prior art and novelty check

Regression-based reconstruction from other variables is established. Multiple imputation by chained equations (MICE/FCS) fits conditional models for each incomplete variable using other variables; see van Buuren & Groothuis-Oudshoorn, *Journal of Statistical Software* 45(3), 2011: https://doi.org/10.18637/jss.v045.i03 (primary paper). scikit-learn's `IterativeImputer` likewise estimates each feature from the others and documents its MICE inspiration: https://scikit-learn.org/stable/modules/impute.html and https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html. Ridge is a standard regularized conditional regression estimator. The specific idea of screening low-missingness columns as a fixed predictor set for one-step Ridge filling may be a combination choice, but this scan did not establish that it is new; novelty remains unestablished.

Closest simpler control: median imputation + identical HGB. Closest method ablation: HGB with native NaN handling and no reconstruction. Under the benchmark's degenerate anchor patterns the candidate exactly reduces to the latter, so the final contrast against median imputation cannot identify the proposed reconstruction effect.

## Faithfulness and fixture mismatch

Implementation preserves the <20% threshold, Ridge alpha 1.0, complete-row Ridge fitting, shared HGB settings, and specified native-NaN fallback. For stress rate 10%, all 20 columns are anchors and none is volatile. For 30% and 50%, all columns are volatile and there are no anchors. This follows from the fixture's uniform MCAR mask, not from a code failure. Consequently no Ridge models are fit and zero entries are reconstructed; the proposed mechanism is not testable on this fixture as written. Creating a feature-specific missingness shift would invent information absent from the proposal/benchmark and is not treated as a faithful test.

## Protocol and reproducibility

Three predeclared seeds: 137, 271, 389. For each seed, `make_classification(n_samples=5000, n_features=20, n_informative=5, n_redundant=5, weights=[0.8,0.2], class_sep=1.0, random_state=seed)`, stratified 60/20/20 split (train_test_split test_size .4, then .5, random states seed and seed+1). Evaluation conditions are clean and independent fixed MCAR masks at 10%, 30%, and 50%. Main metric is average precision averaged across conditions. All transforms and models use training rows only; test features are used only for final prediction and reconstruction diagnostic, labels only for scoring. One CPU thread.

Run from `experiments/alien-science/upstream`:

```sh
../.venv-benchmark/bin/python scripts/with_compute_slot.py -- ../.venv-benchmark/bin/python candidate-studies/tabular_missing-random_seed-2/experiment.py validation --output candidate-studies/tabular_missing-random_seed-2/validation.json
# Freeze preregistration.json after validation, before reading final scores.
../.venv-benchmark/bin/python scripts/with_compute_slot.py -- ../.venv-benchmark/bin/python candidate-studies/tabular_missing-random_seed-2/experiment.py final --output candidate-studies/tabular_missing-random_seed-2/results.json
```

The script refuses to overwrite outputs unless `--allow-overwrite` is supplied. `preregistration.json` records the source SHA-256 and split hashes and was written after all validation results, before the sole held-out run. Validation scores are retained in `validation.json`; held-out details in `results.json` and the compact comparison in `results-summary.json`.

## Results and verdict

Validation completed on all seeds before held-out evaluation. The preregistered proposed-mechanism criterion failed: no anchors for the 30%/50% stress cases, no fitted Ridge model, no reconstructions, and thus no reconstruction R². The held-out study's native-NaN HGB fallback exceeded median-imputed HGB in average precision by a mean 0.0207 across all 12 seed-condition pairs (mean 0.0276 over the nine stressed pairs). This is a result for the classifier's native missing-value handling versus median imputation; it is not evidence for Ridge reconstruction.

The formal verdict is `invalid_mechanism` for this benchmark because the specified data-generating corruption cannot instantiate the assumed stable/volatile feature groups. `faithful_implementation` is true: the candidate faithfully degenerates to its explicitly stated fallback. No claim of significance, generalization, or novelty follows from these three synthetic seeds.

## Smallest falsification test

On the prescribed uniform-MCAR fixture, inspect counts before model evaluation: if no stress condition has both at least one anchor and one volatile column, the defining reconstruction operation cannot execute. It did not execute here, and the mechanism's predeclared minimum test condition (at least one reconstructed stress entry and mean clean held-out reconstruction R² ≥0.10, plus improved stressed AP) failed immediately.

## Limitations

A meaningful next study needs a predeclared feature-specific deployment shift (stable columns remain below 20%, selected volatile columns exceed it) and a matched native-NaN HGB ablation, before any evaluation. This candidate proposal did not specify which columns or rate pattern, so this study does not choose them. Test results are exploratory only.
