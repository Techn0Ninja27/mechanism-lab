# digits_shift-world-2 — corruption-variance weighted PCA

## Proposal and mechanism

The candidate estimates each pixel's variance across clean and five corrupted versions of each training image, averages those variances over training examples, multiplies each pixel by the reciprocal variance, then fits `StandardScaler`, 32-component PCA, and an RBF SVM on the augmented training set. The claim is that this promotes corruption-stable image structure. The standardization order is part of the proposal and was preserved exactly.

## Prior art and novelty check

Noise-aware PCA is established. Bailey's *Principal Component Analysis with Noisy and/or Missing Data* estimates measurement errors and uses them in a weighted EM-PCA objective so components reflect underlying signal rather than measurement noise: https://arxiv.org/abs/1208.4122. That is a more principled noise-aware PCA method than diagonal rescaling followed by ordinary PCA. Hong et al. study optimal weighting for heteroscedastic PCA and show inverse-noise-variance weights are not generally optimal: https://arxiv.org/abs/1810.12862. Spatially weighted PCA for imaging classification uses feature weighting in PCA: https://pmc.ncbi.nlm.nih.gov/articles/PMC4467033/. Corruption/data augmentation for image classifiers, including SVM settings, is also established; Wong et al. compare augmentation approaches: https://arxiv.org/abs/1609.08764.

The specific per-pixel corruption variance estimate and its combination with augmentation may be a distinct recipe, but novelty is not established. More decisively, in this proposal the subsequent per-feature standardization cancels positive pixel multipliers on nonconstant features: `StandardScaler(a*x)` equals `StandardScaler(x)` for positive `a`. Thus the stated weighting cannot alter the PCA input (apart from floating-point effects and treatment of constant features). This is the smallest falsification test and it returned bit-identical held-out predictions versus uniform weights.

## Implementation and reproducibility

The script uses sklearn digits, three stratified 60/20/20 splits with seeds 137, 271, 389, five train-only corruption copies, shared deterministic test corruption draws, identical 32-component PCA/SVM capacity, and validation-only selection from a fixed five-setting `(C, gamma)` grid. The original proposal described two corruption conditions; the controlling study protocol expands evaluation to clean, three noise severities, three dropout severities, and two joint conditions. Both weighted and uniform methods use the same augmented examples. The no-standardization ablation removes the post-weight `StandardScaler` but is therefore not the original pipeline.

Run from `experiments/alien-science/upstream`:

```bash
../.venv-benchmark/bin/python scripts/with_compute_slot.py -- ../.venv-benchmark/bin/python candidate-studies/digits_shift-world-2/experiment.py validation --output candidate-studies/digits_shift-world-2/validation.json
# Write preregistration.json with the resulting validation choices and source/split hashes before final:
../.venv-benchmark/bin/python scripts/with_compute_slot.py -- ../.venv-benchmark/bin/python candidate-studies/digits_shift-world-2/experiment.py final --output candidate-studies/digits_shift-world-2/results.json
```

The held-out output refuses overwrite. `preregistration.json` records source and split hashes, validation-selected hyperparameters, metric, conditions, and pass rule. Whole-run wall time was 20.52 s validation and 22.11 s final with one CPU thread. Per-fit training and per-prediction inference timings were not separately instrumented.

## Results and verdict

Held-out mean accuracy over nine conditions:

| Seed | Weighted PCA (proposal) | Uniform PCA control | Weighted PCA without standardization (ablation) |
|---|---:|---:|---:|
| 137 | 0.801852 | 0.801852 | 0.816358 |
| 271 | 0.813272 | 0.813272 | 0.820988 |
| 389 | 0.805864 | 0.805864 | 0.812346 |
| Mean | 0.806996 | 0.806996 | 0.816564 |

The proposal's paired deltas are exactly `[0, 0, 0]`; it fails the preregistered `+0.01` absolute improvement rule. The ablation's higher mean is exploratory and cannot rescue the proposal because it changes the operation order, was not the candidate, and its result does not establish a novel mechanism. Verdict: `invalid_mechanism` for the originally stated weighting mechanism under its stated preprocessing sequence. The result is a small controlled replication, not proof of external novelty or statistical significance.
