# digits_shift-world-3

## Question and proposal

Can a three-pipeline RBF-SVM ensemble improve digit accuracy under pixel noise and dropout, using 2×2 block occlusion augmentation in one pipeline and validation mean decision margins to weight predictions?

The proposal joins two mechanisms: structured occlusion augmentation and post-hoc validation-margin weighting. Its explanatory claims are plausible but not independently demonstrated by the proposal. In particular, raw SVM decision margins are not calibrated confidence estimates, and margin scale can depend on the fitted model and preprocessing.

## Prior art checked

- [Cutout: Improved Regularization of Convolutional Neural Networks](https://arxiv.org/abs/1708.04552) masks square input regions during training to improve robustness and performance. This is already a close precedent for contiguous input occlusion augmentation.
- [Occlusions for Effective Data Augmentation in Image Classification](https://www.robots.ox.ac.uk/~vgg/publications/2019/Fong19b/fong19b.pdf) studies occlusion augmentation directly. This makes the general augmentation mechanism known; this study only checks a small 2×2/SVM combination.
- [Ensemble Selection from Libraries of Models](https://www.cs.cornell.edu/~alexn/papers/shotgun.icml04.revised.rev2.pdf) establishes validation-driven selection and combination of model libraries as established practice.
- [A probabilistic classifier ensemble weighting scheme based on cross-validated accuracy estimates](https://pmc.ncbi.nlm.nih.gov/articles/PMC6790343/) is a primary study of validation-estimated classifier weights. It supports the general idea of weighted combination, not the specific use of uncalibrated raw SVM margins.

**Novelty status:** no evidence of external novelty. The ingredients are established; whether this exact combination has been published was not exhaustively searched.

## Implementation

`experiment.py` trains four matched RBF-SVM arms on each training split: (1) StandardScaler with ordinary noise-plus-pixel-dropout augmentation (also the control), (2) StandardScaler with the same pixel augmentation, (3) PCA(32) with block occlusion augmentation, and (4) MinMaxScaler with mixed noise/dropout augmentation. The three non-control models form the proposal ensemble. Each source image contributes two augmented copies in each arm. Preprocessing is fit only on training data. Validation weights are each classifier's mean maximum absolute one-vs-rest decision score across the fixed validation corruption conditions, normalized to sum to one.

The weighting ablation replaces the validation-margin weights with uniform weights. A second ablation compares the ensemble with the validation-accuracy-selected single pipeline. Comparing the block pipeline with the pixel pipeline is suggestive but does not isolate block geometry from preprocessing: the proposal pairs block augmentation with PCA(32), while the pixel arm uses StandardScaler. This is a material limitation on attributing gains specifically to block dropout.

Corruption levels follow the candidate-study protocol: clean, noise σ=.2/.4/.6, dropout=.05/.15/.30, and joint (.4,.15)/(.6,.30), with noise scaled by 16 and pixel values clipped to [0,16]. Shared corruption seeds are used across methods. Seeds are 137, 271, and 389 with stratified 60/20/20 splits. The held-out test split is used only in final mode.

Run from `experiments/alien-science/upstream`:

```sh
../.venv-benchmark/bin/python scripts/with_compute_slot.py -- ../.venv-benchmark/bin/python candidate-studies/digits_shift-world-3/experiment.py validation --output candidate-studies/digits_shift-world-3/validation.json
../.venv-benchmark/bin/python scripts/with_compute_slot.py -- ../.venv-benchmark/bin/python candidate-studies/digits_shift-world-3/experiment.py final --output candidate-studies/digits_shift-world-3/results.json
```

The preregistered primary metric is equal-weight mean accuracy over nine conditions. The source SHA256 and validation split hashes are in `preregistration.json`. Runtime was roughly 7 seconds for validation and 13 seconds for final on the benchmark environment, at one CPU thread. Fit times are included per model in the JSON; aggregate evaluation time is approximate and is not per-model inference latency.

## Results

| Seed | Margin ensemble | Uniform ensemble | Val-selected single | Ordinary augmentation control |
|---:|---:|---:|---:|---:|
| 137 | 0.7272 | 0.7272 | 0.8188 | 0.6052 |
| 271 | 0.7380 | 0.7377 | 0.8117 | 0.5914 |
| 389 | 0.7349 | 0.7349 | 0.8049 | 0.6034 |
| Mean | **0.7333** | 0.7332 | **0.8118** | 0.6000 |

The margin ensemble gains 0.1333 absolute accuracy over the control, but margin weighting adds only 0.0001 over uniform averaging. It trails the validation-selected block/PCA pipeline by 0.0785. The block/PCA arm is the validation-selected single model in all three seeds. Thus the result supports the multi-pipeline ensemble over the simple augmentation control on this benchmark, but does not show that margin weighting is useful; simple validation model selection is stronger here. Detailed per-condition scores and runtimes are in `results.json`.

PCA emitted repeated NumPy runtime warnings (`divide by zero`, `overflow`, and `invalid value encountered in matmul`) during evaluation, although the run completed and emitted metrics. Those metrics are retained, but the warning behavior lowers confidence in the PCA arm and should be resolved before treating the apparent block/PCA advantage as stable.

## Verdict and limitations

The preregistered candidate criterion was a margin-ensemble gain of at least .01 over control without underperforming uniform averaging. It passed that criterion and is recorded as a **supported pilot**, not a confirmed effect. The seed count is small, this is one small dataset and synthetic corruption family, and the gain could arise from model diversity or the PCA pipeline rather than contiguous occlusion itself. Margin weights were nearly uniform and raw margins are scale-sensitive. This is an exploratory replication; it establishes neither external novelty nor generalization to real occlusions.

**Smallest useful falsification test:** resolve the PCA warnings, then repeat with matched preprocessing for pixel-dropout and block-dropout arms. If the matched block arm does not improve on random dropout under the same corruption draws, the proposed occlusion explanation is falsified. If margin weighting remains indistinguishable from uniform averaging, drop the margin-specific claim.
