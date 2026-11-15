# digits_shift-direct-3 — Gaussian pre-smoothing

## Proposal and mechanism

The candidate proposes a fixed 2D Gaussian blur (σ selected from 0, 0.4, 0.6, 0.8, 1.0) before a StandardScaler and RBF-SVM. Its causal claim is that additive pixel noise and pixel dropout introduce high-frequency variance, and smoothing should reduce within-image corruption distances while retaining 2–4 pixel digit strokes. The direct control uses the same train-only corruption augmentation and classifier, with σ=0. This is a small empirical mechanism test, not a novelty claim.

## Prior art checked (2026-11-16)

- Gaussian filtering as image denoising / preprocessing is established; it trades noise attenuation against edge/stroke blurring. See the primary empirical study on noise and image classification: [Costa et al., “An empirical study on the effects of different types of noise in image classification tasks” (2016)](https://gbpcosta.github.io/assets/publications/Costa_WVC2016.pdf).
- Corruption benchmarks explicitly include Gaussian blur and noise as distinct image corruptions: [Hendrycks & Dietterich, “Benchmarking Neural Network Robustness to Common Corruptions and Perturbations” (ICLR 2019)](https://openreview.net/forum?id=HJz6tiCqYm).
- Noise augmentation is a strong, widely used baseline for corruption robustness: [Rusak et al., “Increasing the Robustness of DNNs against Image Corruptions by Playing the Game of Noise” (ICLR 2020)](https://trustworthyiclr20.github.io/rusak.pdf).
- Randomized smoothing is a different method: it averages predictions over random Gaussian input noise and gives an L2 certificate; it is not the deterministic spatial low-pass filter tested here: [Cohen, Rosenfeld & Kolter (ICML 2019)](https://proceedings.mlr.press/v97/cohen19c.html).

Thus Gaussian smoothing, noise augmentation, and corruption evaluation are known components. This check did not establish that this exact preprocessing/augmentation/RBF-SVM combination on 8×8 digits is previously published, and absence of a located exact match is not evidence of novelty.

## Implementation

`experiment.py` uses sklearn `load_digits`, stratified 60/20/20 splits for seeds 137, 271, 389, train-only StandardScaler fitting, `SVC(C=10, gamma='scale')`, and fixed per-condition corruption draws shared across methods. Smoothing uses `scipy.ndimage.gaussian_filter(..., mode='reflect')`. Augmentation appends a Gaussian-noise copy (σnoise=0.4×16) and a dropout copy (p=0.15) to training data. These augmentation rates are concrete implementation choices because the proposal did not specify them.

The required benchmark grid here is nine conditions: clean, noise 0.2/0.4/0.6, dropout 0.05/0.15/0.30, and joint (0.4,0.15)/(0.6,0.30); scores are mean accuracy. This supersedes the proposal’s different six-condition metric as required by the study protocol. Validation is completed across all seeds before final evaluation; each seed independently selected σ=0.4. The held-out fitting used the train split only; the proposal described refitting on train plus validation. That discrepancy is a protocol limitation and means these scores are an exploratory pilot rather than a faithful replication of that final refit detail. The test outputs were evaluated once and not tuned against.

## Reproduction

Run from `experiments/alien-science/upstream`:

```sh
../.venv-benchmark/bin/python scripts/with_compute_slot.py -- ../.venv-benchmark/bin/python candidate-studies/digits_shift-direct-3/experiment.py validation --output candidate-studies/digits_shift-direct-3/validation.json
../.venv-benchmark/bin/python scripts/with_compute_slot.py -- ../.venv-benchmark/bin/python candidate-studies/digits_shift-direct-3/experiment.py final --prereg candidate-studies/digits_shift-direct-3/preregistration.json --output candidate-studies/digits_shift-direct-3/results.json
```

The script refuses to overwrite outputs unless `--allow-rerun` is supplied. `preregistration.json` records the source hash, split hashes, selected validation sigmas, control, ablation, and pass rule. Dependencies are the pinned shared benchmark environment (Python 3.12.14, NumPy 2.2.6, SciPy 1.16.3, scikit-learn 1.7.2, threadpoolctl 3.6.0). Total measured model train+inference time was under 12 seconds across validation and final runs.

## Results and verdict

| Seed | σ=0.4 + augmentation | σ=0 + augmentation | Paired gain |
|---:|---:|---:|---:|
| 137 | 0.8062 | 0.7951 | +0.0111 |
| 271 | 0.8049 | 0.7948 | +0.0102 |
| 389 | 0.7932 | 0.7836 | +0.0096 |
| Mean | 0.8014 | 0.7912 | +0.0103 |

Clean accuracy was unchanged versus the control for all seeds. The preregistered pilot rule (mean gain >1 percentage point, each seed positive, clean loss ≤2 points) is narrowly met. The no-augmentation ablation at σ=0.4 averaged 0.469, with near-chance scores on higher noise and joint conditions; it shows that the observed result depends strongly on the specified corruption augmentation, not smoothing alone.

**Verdict:** `supported_pilot` for this local setup only. The effect is small and close to the chosen threshold, with only three seeds and one dataset. There is no significance claim and novelty is not established. The smallest falsification experiment is a new preregistered three-seed replication with the proposal’s train+validation final refit and an independently fixed augmentation recipe; failure to reproduce a positive paired mean gain would falsify this pilot claim.
