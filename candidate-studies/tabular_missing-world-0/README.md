# tabular_missing-world-0: literal formula study

## Question and proposal

The proposal claims that a scalar-weighted replacement plus appended binary missingness features improves average precision when deployment missingness rises. Its stated equation is

`c_i = v_i * (1 - m_i) + alpha * m_i * mu_i`,

where `mu_i` is the training-column mean. The equation—not the inconsistent prose—was implemented literally. For missing cells this is `alpha * mu_i`: alpha 0 means zero imputation; alpha 1 means mean imputation. Intermediate values shrink the training mean toward zero. The mask is appended as a binary feature.

There is a material internal contradiction: the prose says alpha 0 is median/mean imputation and suggests intermediate alpha encodes graded missingness confidence. Neither follows from the equation. The equation's appended mask is still binary. Furthermore, the proposal says append masks but does not specify training-time corruption/augmentation. None was added. Since training inputs are complete, mask columns are constant zeros during training; the classifier cannot learn mask-dependent behavior. This limitation is central to interpreting the result.

## Prior art checked

- The missing-indicator method (MIM) combines imputation with indicator columns and has been studied theoretically and empirically. The proposed indicator component is therefore established: [MIM: From Low to High Dimensions (arXiv)](https://arxiv.org/abs/2211.09259).
- Pattern-submodel work relates missingness indicators and imputation in prediction models and discusses bias from placeholder choice: [Lee et al., Biostatistics](https://pmc.ncbi.nlm.nih.gov/articles/PMC7868046/).
- The selected classifier itself natively routes NaNs and can learn missingness patterns: [scikit-learn HistGradientBoosting missing-value documentation](https://scikit-learn.org/stable/modules/ensemble.html).

The combination of scalar mean replacement and appended mask columns is a parameterized imputation choice, but this small search did not establish a novel mechanism. Established components and the candidate relationship should not be conflated.

## Method

Three dataset seeds (137, 271, 389); each uses `make_classification(n_samples=5000, n_features=20, n_informative=5, n_redundant=5, weights=[.8,.2], class_sep=1)` and stratified 60/20/20 splits. The validation-selected alpha is chosen separately per seed by mean validation AP over clean and MCAR 10%, 30%, and 50%. Each corruption mask is shared across arms. Test conditions and alpha selections were frozen before the one final pass. All arms use `HistGradientBoostingClassifier(max_iter=120)` and 3000 clean training examples.

Control: training-fitted median imputation plus HGB, no mask columns. Ablation: training-mean imputation plus appended mask. Candidate: literal formula at alpha in `[0, .3, .7, 1]`, plus appended mask. Each arm uses the same splits, data, classifier capacity, and prediction-time masks. The model has fixed capacity; the candidate's vector input is 40 columns while the control is 20 columns, with the extra 20 mask columns constant during training.

The common candidate-study protocol required MCAR 10/30/50% and per-seed dataset generation. This supersedes the proposal's random_state 42 and its alternative validation metric of MCAR 20/40% plus informative-feature block missingness. Those proposal-specific conditions were not tested. Training-time missingness augmentation was explicitly not introduced.

## Reproduction

From `experiments/alien-science/upstream`:

```sh
../.venv-benchmark/bin/python scripts/with_compute_slot.py -- ../.venv-benchmark/bin/python candidate-studies/tabular_missing-world-0/experiment.py validation --output candidate-studies/tabular_missing-world-0/validation.json
../.venv-benchmark/bin/python scripts/with_compute_slot.py -- ../.venv-benchmark/bin/python candidate-studies/tabular_missing-world-0/experiment.py final --output candidate-studies/tabular_missing-world-0/results.json
```

Final mode reads the frozen alpha choices from `validation.json`, never overwrites existing output unless `--allow-rerun` is explicitly supplied, and does not use test labels for selection. See `preregistration.json` for source SHA256, split hashes, and frozen choices; `validation.json` and `results.json` contain per-seed/per-condition scores and runtime.

## Results

Held-out mean AP over the four conditions:

| Seed | Selected alpha | Candidate | Median control | Paired delta |
|---:|---:|---:|---:|---:|
| 137 | 1.0 | 0.71964 | 0.72467 | -0.00503 |
| 271 | 0.0 | 0.84833 | 0.84173 | +0.00660 |
| 389 | 1.0 | 0.79468 | 0.78736 | +0.00731 |
| Mean | — | 0.78755 | 0.78459 | +0.00296 |

The mean lift is small and inconsistent across seeds. Validation selected endpoints for all three seeds (1, 0, 1). At alpha 1, candidate and mean-imputation-plus-mask are algebraically identical; their held-out scores match for seeds 137 and 389. Validation mean-imputation-plus-mask also equals alpha 1 exactly. Thus this study does not identify an advantage from a distinct graded mask mechanism. Inference timings and total run elapsed time are in the JSON files; total validation and final runtimes were approximately four seconds each across all arms and seeds. Separate training time by arm was not recorded.

## Verdict and smallest falsification test

Verdict: `no_gain` for this pilot, with no novelty established. A minimal falsifier is the preregistered three-seed validation-and-held-out comparison against median HGB and mean-imputation-plus-mask; it found no consistent candidate advantage and the simpler alpha-1 ablation is algebraically the same method. The pilot is too small to estimate a reliable effect. It does not support an out-of-domain robustness claim.
