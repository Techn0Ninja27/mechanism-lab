# digits_shift-direct-0 study

## Proposal and verdict

The proposal replaces normalized 64-pixel inputs by 128 features: 56 horizontal differences, 56 vertical differences, and 16 nonoverlapping 2×2 means, then fits a standardized RBF SVM to clean and four corrupted copies. Its hypothesis is that differences cancel additive noise and turn pixel dropout probability (p) into a (p^2) gradient corruption rate, while block means reduce noise variance.

The intervention was implemented as specified. The causal explanation does not hold for independent pixel corruption. For independent equal-variance pixel noise, the variance of a difference is (2\sigma^2), not a reduced variance. A difference changes if exactly one endpoint drops, with probability (2p(1-p)); both dropped endpoints yield zero difference, which is generally also a corruption. Thus the feature map is a testable representation, but its stated structural mechanism is not supported by the equations.

**Verdict: no gain against the matched raw-pixel augmentation control.** Candidate mean accuracy was 0.79095; control was 0.79198, paired deltas by seed were −0.00309, −0.00123, +0.00123. The gradient-only ablation averaged 0.77418 and block-means-only averaged 0.66553. This is a small three-seed exploratory replication, not a significance test or novelty claim.

## Prior art checked

Gradient-based feature extraction for handwriting recognition predates this proposal: [Liu et al., “Handwritten digit recognition: investigation of normalization and feature extraction techniques”](https://www.sciencedirect.com/science/article/abs/pii/S0031320303002243) compares gradient features and reports their strong recognition performance. [A multiple feature/resolution approach to handprinted digit recognition](https://citeseerx.ist.psu.edu/document?doi=e98d4622e07baa0feb299f2a0ed65ffa2671c7ab&repid=rep1&type=pdf) combines local gradients with resolution/region features. [Wavelet Integrated CNNs for Noise-Robust Image Classification](https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_Wavelet_Integrated_CNNs_for_Noise-Robust_Image_Classification_CVPR_2020_paper.pdf) is a more recent primary example of pooling/frequency decomposition explicitly used for noise robustness. The elementary components here—finite differences and local averaging—are established. This search did not find an exact prior paper testing this precise 128D concatenation with this exact corruption setup; that absence does not establish novelty.

## Method and deviations

`experiment.py` uses the repository fixture: `sklearn.load_digits`, stratified 60/20/20 splits from seeds 137, 271, 389, train-only scaling, and corruption magnitude on normalized intensities (noise scale is `severity*16` before clipping). Train augmentation uses four fixed variants: Gaussian noise .2 and .4, dropout .15 and .30. The same model capacity and corruption draws are used for all compared representations. The control is raw 64D inputs with identical augmentation and SVC (`C=10`, `gamma=scale`). Ablations are gradients only and block means only. Evaluation conditions are clean, noise .2/.4/.6, dropout .05/.15/.30, and joint (.4,.15)/(.6,.30).

The original proposal called for an 80/20 then nested validation split and five specific metric conditions (noise .3/.6 and dropout .2/.4), while the shared study instructions prescribe 60/20/20 and a common nine-condition fixture. The common fixture was used for reproducibility across candidates. The proposal omitted exact augmentation severities, so these were fixed before validation as above. No hyperparameter tuning was done. Validation completed for all seeds before preregistration and one final test run.

## Reproduction and artifacts

From `experiments/alien-science/upstream`:

```sh
../.venv-benchmark/bin/python scripts/with_compute_slot.py -- ../.venv-benchmark/bin/python candidate-studies/digits_shift-direct-0/experiment.py validation --output candidate-studies/digits_shift-direct-0/validation.json
../.venv-benchmark/bin/python scripts/with_compute_slot.py -- ../.venv-benchmark/bin/python candidate-studies/digits_shift-direct-0/experiment.py final --output candidate-studies/digits_shift-direct-0/results.json
```

The final command was run only after `preregistration.json` recorded the source hash, split hashes, metric, comparison, and decision threshold. It refuses to overwrite an existing result path. The run uses one CPU thread. `results.json` contains condition-level held-out scores and fit times; `validation.json` contains validation scores and split hashes. Fitting took approximately 0.15–0.46 seconds per method/seed in this environment. Inference time was not separately instrumented, a reporting limitation.

## Smallest falsification test

The cheapest kill test is the matched representation comparison on one fixed held-out split under Gaussian noise alone: independent noise should increase the difference-channel variance to twice the pixel-noise variance. The full three-seed study adds dropout, joint corruption, and representation ablations. The observed overall candidate-control difference is slightly negative, so the preregistered support rule fails.

## Limits

This dataset is tiny and low resolution; three seeds provide weak evidence beyond this fixture. Accuracy across the chosen conditions averages easy clean cases with severe corruption. The results do not establish whether other datasets, preprocessing, feature scalings, or augmentation policies behave differently. Novelty remains unestablished and human adjudication has not occurred.
