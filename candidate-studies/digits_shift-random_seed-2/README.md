# Candidate study: sinusoidal severity sweep

## Proposal and mechanism

The candidate trains an RBF SVM on clean digits plus eight augmented copies per training example. Each copy uses Gaussian pixel noise and random pixel dropout, scaled by a severity value. The proposal's schedule is `s_i = 0.5 * (1 + sin(2*pi*i/8))`; its matched control uses eight uniformly spaced severities from 0 to 1. The proposed mechanism is that redistributing a fixed augmentation budget across severity levels changes which examples shape the SVM margin and improves robustness at high test severity.

The schedule as written is not an equal-arc-length sweep. Its actual severities are approximately `[.5, .854, 1, .854, .5, .146, 0, .146]`, which repeats some values and places extra weight near endpoints compared with a uniform grid. Also, Gaussian standard deviation and dropout probability are linear functions of severity in this experiment, so the proposal's claim that the corruption functions are nonlinear is not supported by the implemented corruption definitions.

## Prior art checked

- [RandAugment (Cubuk et al., 2019)](https://arxiv.org/abs/1909.13719) uses a reduced augmentation-policy space with a global distortion magnitude. It establishes that randomized magnitude-controlled augmentation is a known component.
- [AugMix (Hendrycks et al., 2019/2020)](https://arxiv.org/abs/1912.02781) combines diverse augmentation chains to improve corruption robustness. It establishes broad augmentation mixtures as prior art.
- [On the Interaction Between Augmentations and Corruptions in Natural Corruption Robustness (Mintun et al., NeurIPS 2021)](https://arxiv.org/abs/2102.11273) finds that robustness depends on similarity between training augmentations and test corruptions. This cautions against assuming that a severity distribution helps independently of the test corruption distribution.

These sources cover randomized magnitudes, mixtures, and augmentation/corruption matching. I found no source in this focused search testing this exact eight-point sinusoidal severity schedule with a fixed-budget RBF SVM on sklearn digits. That narrow absence does not establish external novelty; the underlying technique is a simple choice of augmentation sampling distribution.

## Implementation and deviations

`experiment.py` implements the proposal and matched controls. StandardScaler is fit only on clean training pixels. Augmented copies are generated in raw pixel space, then transformed with that fixed scaler. Each method has the same split, sample count, classifier settings, and fixed per-copy random streams. The `constant_mid` ablation retains eight copies at severity .5; `single_mid` uses one copy to expose the effect of training-set size.

The repository protocol supersedes the proposal sketch's seed-42 80/20 split and four test conditions: this study uses stratified 60/20/20 splits from seeds 137, 271, and 389 and the prescribed nine corruption conditions. Pixel corruption noise uses the benchmark convention `noise_fraction * 16` in raw pixel units. These are documented protocol changes, not results under the original sketch's exact split/metric.

Reproduce validation, then final (final refuses to overwrite an existing output):

```sh
../.venv-benchmark/bin/python scripts/with_compute_slot.py -- ../.venv-benchmark/bin/python candidate-studies/digits_shift-random_seed-2/experiment.py validation --output candidate-studies/digits_shift-random_seed-2/validation.json
../.venv-benchmark/bin/python scripts/with_compute_slot.py -- ../.venv-benchmark/bin/python candidate-studies/digits_shift-random_seed-2/experiment.py final --output candidate-studies/digits_shift-random_seed-2/results.json
```

Validation was completed for all seeds and variants before the final evaluation. The frozen split hashes and source SHA256, parameters, metric, and decision rule are in `preregistration.json`. The held-out final outputs are in `results.json`. All model fits used one CPU thread. No variants failed.

## Results

Primary metric: unweighted mean accuracy over clean, noise levels .2/.4/.6, dropout .05/.15/.30, and joint (.4,.15)/(.6,.30). Values below are from the held-out test splits.

| Seed | Sinusoidal | Uniform control | Paired delta | Constant mid | Single mid |
|---:|---:|---:|---:|---:|---:|
| 137 | 0.6534 | 0.6537 | -0.0003 | 0.6731 | 0.5556 |
| 271 | 0.6475 | 0.6494 | -0.0019 | 0.6611 | 0.5503 |
| 389 | 0.6716 | 0.6821 | -0.0105 | 0.6849 | 0.5759 |
| **Mean** | **0.6575** | **0.6617** | **-0.0042** | **0.6730** | **0.5606** |

Sinusoidal training did not beat uniform severity on any seed; its mean paired difference was -0.42 percentage points. The constant-mid ablation was better than both on all three seeds. Single-mid performed substantially worse, consistent with the augmentation sample count mattering, but this does not rescue the proposed schedule. Candidate train time was 2.11–2.73 s per seed and inference over all conditions was 1.16–1.38 s. Uniform control train time was 2.18–2.58 s and inference 1.15–1.21 s. Validation took 29.85 s total; final evaluation took 34.08 s total.

## Verdict and limits

**No gain in this pilot.** The preregistered support rule required at least +0.005 mean accuracy over uniform and a win on at least two of three seeds. The candidate instead lost by 0.0042 on average and lost all three seeds. The smallest falsification test was therefore satisfied by the matched held-out comparison; no additional tuning was done.

This is a small benchmark with three fixed seeds and synthetic corruptions. It does not establish statistical significance, disprove all severity-distribution choices, or establish novelty. The sweep's mechanistic story is also weakened by the actual schedule's repeated severities and the linear corruption parameterization.
