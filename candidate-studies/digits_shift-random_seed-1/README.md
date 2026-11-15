# digits_shift-random_seed-1: flip discrepancy scalar

## Proposal and mechanism

The candidate adds one feature to each normalized 8×8 image:

`sym_score(x) = exp(-gamma_sym * ||x - horizontal_flip(x)||^2)`

where `gamma_sym = 1 / (64 * var(X_train_augmented))`. A StandardScaler and an RBF SVM (`C=10`, `gamma='scale'`) are fit on four training examples per original: the clean image and three independently generated copies, each with Gaussian noise σ=0.5 in [0,1] pixel units and 10% pixel dropout. The proposal's causal claim is that this symmetry-derived scalar helps the SVM distinguish noise-corrupted regions beyond what pixel distances provide.

The control uses identical augmentation and model fitting on the 64 pixels. The structure-destroying ablation swaps horizontal reflection for one fixed random pixel permutation while keeping one scalar. The same augmentation samples and corruption draws are used by all methods within each seed.

## Prior art checked (2026-11-16)

- [Mairal et al., “A Kernel Theory of Modern Data Augmentation” (2019)](https://arxiv.org/abs/1803.06084): derives how augmentation induces kernel behavior and relates augmentation to invariance and regularization. It establishes the general kernel/augmentation connection, not this particular appended scalar.
- [Mei, Misiakiewicz & Montanari, “Learning with Invariances in Random Features and Kernel Models” (COLT 2021)](https://proceedings.mlr.press/v134/mei21a.html): studies invariant kernel methods and shows data augmentation with an unstructured kernel estimator can be equivalent to an invariant-kernel estimator under their assumptions.
- [Raj et al., “Local Group Invariant Representations via Orbit Embeddings” (AISTATS 2017)](https://proceedings.mlr.press/v54/raj17a.html): builds kernel representations over transformation orbits, including image classification experiments.
- [Handwritten Digit Classification, S. Maji](https://home.ttic.edu/~smaji/projects/digits/): documents raw-pixel SVM baselines and engineered image descriptors for handwritten digits.

These sources establish symmetry/invariance-based representations and kernel treatment as established ideas. In this targeted search I did not find the exact `exp(-gamma * squared horizontal-flip distance)` scalar appended to pixels for corruption-robust digit classification. That narrow absence is not evidence of novelty; novelty is unestablished. A key task-specific caution is that horizontal reflection is not generally label-preserving for handwritten numerals, so a reflection discrepancy is a descriptive feature rather than a justified invariance.

## Implementation and deviations

Implementation follows the proposal's 65-dimensional feature, gamma heuristic, augmentation count, augmentation levels, scaler, and SVM settings. It uses the study protocol's stratified 60/20/20 split and three seeds (137, 271, 389), in place of the proposal's inconsistent split description. Pixels are divided by 16 before augmentation, as implied by the proposal's phrase “normalized pixel values.” The proposal did not clarify whether σ=0.5 was on raw 0–16 values or normalized 0–1 values; this implementation uses normalized values literally. Consequently, many augmented pixels clip to 0 or 1. This ambiguity materially limits interpretation, but the setting was not softened after results.

Validation uses only the validation split. Final evaluation was run once after all validation results and preregistration were frozen. Splits, corruption draws, augmentation draws, preprocessing, and classifier capacity are matched. No fit step sees validation or test samples. All three variants completed without exceptions. Runtime was about 8.5 seconds for validation and 8.6 seconds for final across all seeds and variants.

## Reproduce

From `experiments/alien-science/upstream`:

```bash
../.venv-benchmark/bin/python scripts/with_compute_slot.py -- ../.venv-benchmark/bin/python candidate-studies/digits_shift-random_seed-1/experiment.py validation --output candidate-studies/digits_shift-random_seed-1/validation.json
../.venv-benchmark/bin/python scripts/with_compute_slot.py -- ../.venv-benchmark/bin/python candidate-studies/digits_shift-random_seed-1/experiment.py final --output candidate-studies/digits_shift-random_seed-1/results.json
```

Final mode refuses to overwrite an existing result unless explicitly passed `--rerun`. The study was run with Python 3.12 and scikit-learn 1.7.2. The code is offline and uses `sklearn.load_digits` only. `preregistration.json` records the source hash, split hashes, fixed settings, primary metric, and validation pass rule.

## Results

Mean accuracy on the nine held-out conditions (clean; noise 0.2/0.4/0.6; dropout 0.05/0.15/0.30; joint (0.4,0.15)/(0.6,0.30)):

| Seed | Pixel control | Flip scalar | Random permutation scalar | Flip − control | Flip − permutation |
|---:|---:|---:|---:|---:|---:|
| 137 | 0.809259 | 0.809568 | 0.806790 | +0.000309 | +0.002778 |
| 271 | 0.806790 | 0.808642 | 0.809568 | +0.001852 | −0.000926 |
| 389 | 0.799074 | 0.799383 | 0.799383 | +0.000309 | 0.000000 |
| Mean | 0.805041 | 0.805864 | 0.805247 | +0.000823 | +0.000617 |

The validation rule required the candidate to beat control in at least two of three seeds, have positive mean paired delta, and beat the permutation ablation. It failed: flip beat control in one validation seed only, and the permutation feature was slightly better in mean. The held-out candidate-control difference is positive but tiny; it does not change the preregistered verdict. Per-condition scores and runtimes are retained in `validation.json` and `results.json`.

## Verdict and smallest falsification test

**Verdict: no_gain.** The feature is implementable, but the validation evidence does not support its proposed causal mechanism. The held-out average difference is +0.000823 accuracy, with paired seed deltas +0.000309, +0.001852, +0.000309; these are descriptive pilot outcomes, not evidence of a reliable benefit. The random-permutation ablation is close, which weakens the claim that horizontal reflection structure caused the change.

The cheapest falsification test was the preregistered validation comparison of flip scalar against control and fixed-permutation scalar over the three seeds. It falsified the preregistered support criterion without needing held-out data. Limitations include only three splits, a single permutation draw per seed, ambiguous augmentation-noise units, and the questionable label-preserving status of horizontal reflection. No external novelty claim is made.
