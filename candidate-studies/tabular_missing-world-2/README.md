# tabular_missing-world-2 — missingness mask augmentation

## Proposal and mechanism

The proposal adds one binary feature per original feature (1 when absent) after median imputation and randomly masks training values at a 20% rate. Its operative hypothesis is that missingness patterns carry predictive information beyond imputed values. The fictional physical story about sound and light has no bearing on the statistical mechanism; the proposal itself correctly replaces it with an assumption about informative missingness.

This benchmark induces MCAR missingness in both training augmentation and evaluation. Thus it tests robustness to an increase in missingness rate, but does **not** test the proposal's informative/MNAR premise.

## Prior art check

The core mask-indicator technique is established prior art. Lenz et al., [“No Imputation without Representation” (arXiv:2206.14254)](https://arxiv.org/abs/2206.14254), evaluates missingness indicators with imputation across classifiers and reports that indicators generally help, with important classifier/data dependencies. Scikit-learn's [SimpleImputer documentation](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html) exposes `add_indicator=True` as a standard option. Tree models can also encode missingness without explicit appended columns: [HistGradientBoostingClassifier documentation](https://scikit-learn.org/stable/modules/ensemble.html#histogram-based-gradient-boosting) describes learned routing for missing values and splits on missingness when predictive. XGBoost's paper describes another established missing-value mechanism, learned default directions: [Chen & Guestrin (2016)](https://arxiv.org/abs/1603.02754).

Conclusion: no novelty claim is supported. The specific combination of randomized training-time masking plus explicit indicators is a familiar robustness recipe; the study measures its utility on this fixture only.

## Implementation and reproducibility

`experiment.py` fits medians on clean training rows only. Candidate training randomly masks every training row at 20%, appends the 20 binary indicators, imputes from frozen medians, and fits the same 120-iteration HGB classifier used by controls. The mechanism ablation receives the identical masked training matrix and imputation, but omits mask columns. Other methods are clean median-impute baseline and mask columns without train-time corruption. Evaluation masks are identical across methods for each seed and missingness rate. All models run with one CPU thread.

Run from `experiments/alien-science/upstream`:

```sh
../.venv-benchmark/bin/python scripts/with_compute_slot.py -- ../.venv-benchmark/bin/python candidate-studies/tabular_missing-world-2/experiment.py validation --output candidate-studies/tabular_missing-world-2/validation.json
../.venv-benchmark/bin/python scripts/with_compute_slot.py -- ../.venv-benchmark/bin/python candidate-studies/tabular_missing-world-2/experiment.py final --output candidate-studies/tabular_missing-world-2/results.json
```

The preregistration records the source hash, split hashes, settings, metric, and decision rule. The corrected validation was completed before the corrected final run. A first implementation accidentally duplicated clean and masked training rows; after noticing this departed from the proposal, it was archived as `results-superseded-unfaithful.json`. Its held-out scores were already seen, so corrected final scores are exploratory rather than a pristine untouched test. The correction was driven by implementation fidelity, not score tuning; no hyperparameters were changed.

## Results

Primary metric: mean of AP over 0%, 10%, 30%, and 50% MCAR per seed, then mean over seeds. Values are in `validation.json` and `results.json`; all 48 method-condition-seed rows are retained.

| Method | Validation mean AP | Corrected final mean AP |
|---|---:|---:|
| Clean median imputation (control) | 0.80238 | 0.78199 |
| Masked training, values only (mechanism ablation) | 0.86352 | 0.83397 |
| Masked training + explicit mask (candidate) | 0.86465 | 0.83789 |
| Clean training + mask | 0.80238 | 0.78199 |

Candidate minus value-only augmentation by seed was `[-0.00370, +0.00156, +0.00554]` on validation and `[+0.00254, +0.00469, +0.00453]` on corrected final. The final mean delta is +0.00392 AP, short of the preregistered +0.01 threshold. Mask-only clean training exactly matched the baseline in this fixture: with no missing training examples, its indicators are constant zeros and add no signal. Training-time masking improved robustness substantially relative to the clean-trained control; the extra mask columns contributed only a small incremental gain over value-only training corruption.

Fit/evaluation runtime was about 2.4 seconds per complete validation or final run on the benchmark environment, with one thread. Per-method fit and per-condition inference timings are in the JSON artifacts.

## Verdict

`no_gain` under the predeclared rule: the indicator contribution did not clear the minimum practical effect threshold versus its matched value-only masking ablation. This is not evidence that indicators never help. The fixture is synthetic and MCAR, the corrected final result has prior test exposure as documented, and three seeds are too few for a significance claim. The smallest useful falsification already run was the paired three-seed comparison of mask-plus-corruption against the same corruption without indicators.
