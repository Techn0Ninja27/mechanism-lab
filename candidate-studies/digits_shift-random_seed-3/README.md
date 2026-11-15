# digits_shift-random_seed-3 — frozen candidate study

## Proposal and causal claim

The candidate proposes four augmented copies of each training image. For every copy, sample an image-specific angle uniformly on `[0, π/2]`, set Gaussian noise scale `σ = 0.25 cos(θ)` and pixel dropout probability `p = 0.25 sin(θ)`, then train a standardized RBF SVM (`C=10`, `gamma=scale`). Its claim is that covering this quarter-circle in `(σ,p)` space improves mixed-corruption accuracy at fixed augmentation count.

The tested mechanism is faithful to those parameter equations. However, the proposal's claim of equal realized L2 corruption energy is mathematically inaccurate: `σ²+p²` fixes a parameter-space radius, not pixelwise corruption energy. Gaussian energy varies stochastically with image size, and zeroing a pixel removes an amount dependent on its intensity. Also, its assumed test budget is violated by the preregistered fixture conditions: test noise reaches 0.6 and dropout reaches 0.30, beyond β=0.25.

## Prior art

Corruption robustness benchmarks already evaluate noise and other distribution shifts (Hendrycks & Dietterich, [ICLR 2019 / arXiv:1903.12261](https://arxiv.org/abs/1903.12261)). AugMix samples and mixes augmentation chains to improve robustness under unforeseen corruptions ([ICLR 2020 / arXiv:1912.02781](https://arxiv.org/abs/1912.02781)). Research also directly studies interactions between train-time augmentations and test-time corruptions ([arXiv:2102.11273](https://arxiv.org/abs/2102.11273)). These establish corruption augmentation and mixed augmentation distributions as known components. They do not establish this exact quarter-circle parameterization as a distinct relationship. No systematic novelty search was performed; external novelty is unestablished.

## Implementation and deviations

`experiment.py` uses sklearn digits, stratified 60/20/20 splits for seeds 137, 271, 389, train-only `StandardScaler` fit on clean rows, fixed RBF SVM settings, and the shared fixture corruption convention (Gaussian scale is severity times 16 in raw 0–16 pixels, then clipped). Validation and final modes are separate. Test evaluation was run only after all validation seeds completed and preregistration was frozen.

All methods have the same fivefold training count (clean plus four copies): `arc` (candidate), `fixed_joint` (independent noise at σ=.25 and dropout p=.25 together; matched control), `noise_axis`, `dropout_axis`, and `clean_resampled`. Corruption draws for evaluation are paired across methods. The primary score is mean accuracy across nine conditions, averaged across the three seeds. The proposal originally described six conditions and a smaller split; the study instructions' common fixture (nine conditions, 60/20/20) was used for comparability.

## Reproduce

From `experiments/alien-science/upstream`:

```sh
../.venv-benchmark/bin/python scripts/with_compute_slot.py -- ../.venv-benchmark/bin/python candidate-studies/digits_shift-random_seed-3/experiment.py validation --output candidate-studies/digits_shift-random_seed-3/validation.json
../.venv-benchmark/bin/python scripts/with_compute_slot.py -- ../.venv-benchmark/bin/python candidate-studies/digits_shift-random_seed-3/experiment.py final --output candidate-studies/digits_shift-random_seed-3/results.json
```

Final mode refuses to overwrite an existing output and requires preregistration and validation artifacts. Per-seed fit and prediction times are in the JSON outputs. The total final run took about 17 seconds.

## Results

Primary mean accuracy (validation → held-out test):

| Method | Validation | Test |
|---|---:|---:|
| Arc candidate | 0.5900 | 0.5908 |
| Fixed joint control | 0.6006 | 0.6107 |
| Noise axis | 0.6309 | 0.6416 |
| Dropout axis | 0.4773 | 0.4798 |
| Clean resampled | 0.4649 | 0.4716 |

Paired test deltas, arc minus fixed-joint control, were −0.0142, −0.0225, and −0.0228 for seeds 137, 271, and 389. The arc candidate also trailed noise-axis augmentation by 0.0491, 0.0568, and 0.0463. Mean accuracy across the two joint test conditions was 0.2662 for arc and 0.3204 for noise-axis; arc did not beat both axis ablations. Every seed underperformed the matched control on the primary metric.

Train fit seconds per seed (arc / fixed-joint / noise-axis / dropout-axis / clean-resampled):

- 137: 0.880 / 0.763 / 0.797 / 0.229 / 0.061
- 271: 0.916 / 0.858 / 0.844 / 0.247 / 0.081
- 389: 1.166 / 0.729 / 0.752 / 0.241 / 0.065

## Verdict and smallest falsification test

**No gain** on this fixture: the arc schedule loses to the matched joint control and the simpler noise-axis augmentation on held-out mean accuracy. The preregistered mixed-condition criterion also fails. The smallest decisive falsification already run was this three-seed, matched-count comparison against the two axis ablations on the two joint-corruption conditions; it found no arc advantage.

Limitations: this is a small sklearn digits pilot with synthetic pixel corruptions, and the severities include values outside the candidate's stated β budget. It does not establish general performance or external novelty. The parameter-space constant-radius interpretation should not be confused with constant realized corruption energy.
