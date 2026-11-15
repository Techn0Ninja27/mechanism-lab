# digits_shift-random_seed-0 — Fisher-Porosity Augmentation

## Exact proposal and mechanism

The proposal computes class-specific per-pixel Fisher discriminability on the training partition, inverts and normalizes it into a porosity map, and allocates additive Gaussian noise and pixel dropout in proportion to that map. The claim is that corrupting less discriminative pixels preserves class-separating structure while teaching robustness to image degradation. The intended classifier is a standardized RBF SVM (C=5, gamma=`scale`) with the scaler fitted on original training data only.

The causal contrast is Fisher-porosity weighting versus uniform corruption augmentation at matched sample count, corruption type/severity distribution, train/validation/test split, scaler, classifier, and random corruption draws. Clean-only and complemented-map variants are ablations. This is a small digits benchmark; it cannot demonstrate real-world image robustness.

## Prior art checked (2026-11-16)

- [Saliency guided data augmentation strategy for maximally utilizing an object’s visual information](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0274767) uses saliency to select visually important regions and constructs augmentation from them. This is close in broad principle (spatial importance guides augmentation), although its patch construction differs from noise/dropout placement.
- [On Interaction Between Augmentations and Corruptions in Natural Corruption Robustness](https://arxiv.org/abs/2102.11273) studies how augmentation-corruption similarity affects robustness and warns that gains may not transfer to dissimilar test corruptions. It supports testing matched corruption families, not the proposed Fisher map itself.
- [Data Augmentation as Feature Manipulation](https://arxiv.org/abs/2203.01572) analyzes augmentation as changing relative feature importance/learnability. This is conceptual prior art for the proposed mechanism’s feature-selective rationale.
- [Weighted Fisher Discriminant Analysis in the Input and Feature Spaces](https://arxiv.org/abs/2004.01857) is prior art for feature weighting with Fisher discrimination, but is a discriminant-learning method, not this augmentation policy.

Searches for Fisher-weighted pixel noise/dropout augmentation and saliency-guided augmentation found adjacent techniques, not a verified exact match for the class-conditional inverse-Fisher corruption allocation. That is not evidence of novelty: this was a targeted search, not a systematic review. **Prior-art status: adjacent components and rationale are established; exact relationship unresolved. External novelty is not established.**

## Implementation and deviations

`experiment.py` uses one-vs-rest feature Fisher scores `(mean_c - mean_rest)^2 / (var_c + var_rest + 1e-6)`, inverts with a `1e-3` stabilizer, and normalizes each class map to mean one. It adds one noise view and one dropout view per original training item. Noise magnitude is sampled uniformly from normalized levels 0.2/0.4/0.6 and map-scaled with each image’s RMS perturbation normalized to match uniform noise energy. Dropout probability is multiplied by the normalized map and clipped at one; therefore realized dropout fraction can differ from the nominal uniform rate. The complement ablation inverts the porosity map and re-normalizes it. The same seeded corruption draws are used across compared methods. Features are clipped to [0,16] after test corruption.

The user-provided fixture protocol specifies 3 seeds (137, 271, 389), stratified 60/20/20 splits, and nine conditions: clean, noise 0.2/0.4/0.6 (Gaussian standard deviation equals fraction × 16), dropout 0.05/0.15/0.30, and joint (0.4,0.15)/(0.6,0.30). This supersedes the older proposal text’s 1/3/5 noise and 0.1/0.3/0.5 dropout levels. This test distribution is unusually severe for 8×8 digits, so absolute scores should not be generalized.

## Reproduction

From `experiments/alien-science/upstream`:

```sh
../.venv-benchmark/bin/python scripts/with_compute_slot.py -- ../.venv-benchmark/bin/python candidate-studies/digits_shift-random_seed-0/experiment.py validation --output candidate-studies/digits_shift-random_seed-0/validation.json
../.venv-benchmark/bin/python scripts/with_compute_slot.py -- ../.venv-benchmark/bin/python candidate-studies/digits_shift-random_seed-0/experiment.py final --output candidate-studies/digits_shift-random_seed-0/results.json
```

Validation completed for all seeds before `preregistration.json` was written; the final test command was then run once. It refuses to overwrite an existing output unless `--allow-rerun` is explicitly passed. The preregistration records source SHA256, split hashes, fixed settings, metric, control, and decision rule. Total experiment CPU time was about 12 seconds across validation and final; the compute gate was used. Per-fit training time is recorded. Inference timing was not separately instrumented; total wall time is recorded.

## Results and verdict

Held-out mean accuracy across conditions, averaged by split first:

| Seed | Clean-only | Uniform | Fisher porosity | Complement | Porosity − uniform |
|---:|---:|---:|---:|---:|---:|
| 137 | 0.4722 | 0.5633 | 0.5707 | 0.4602 | +0.0074 |
| 271 | 0.4691 | 0.5466 | 0.5380 | 0.4608 | −0.0086 |
| 389 | 0.4691 | 0.5867 | 0.5855 | 0.4651 | −0.0012 |
| Mean | 0.4702 | 0.5655 | 0.5647 | 0.4620 | −0.0008 |

The preregistered pilot-support threshold was mean paired improvement above 0.01 and wins on at least two seeds. The candidate improves on uniform augmentation in one of three seeds; its mean is slightly lower, so the verdict is **no_gain**. Complement weighting is substantially worse than uniform in this pilot, consistent with—but not sufficient to validate—the proposed direction of selectivity. The candidate does not meet the pilot-support threshold.

Smallest falsification test: on the same three frozen splits and corruption draws, compare porosity against uniform augmentation. A non-positive paired mean (as observed) falsifies the practical claim of improvement under this fixture. This does not show that feature-selective augmentation is generally ineffective.

Limitations: only 3 seeds and one tiny dataset; severe and synthetic test corruptions; no uncertainty interval or external evaluation; map estimates may be noisy; per-pixel independent weighting ignores spatial structure; dropout clipping changes expected deletion rates; the SVM and fixed C/gamma may not be optimal. No claim of statistical significance or external novelty is made.
