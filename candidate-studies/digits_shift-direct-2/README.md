# digits_shift-direct-2 — Corruption-envelope standardization

## Question and exact proposal

Does replacing clean per-pixel standard deviations in a standardized RBF SVM with the proposal's corruption-envelope denominator improve mean accuracy under known additive Gaussian noise and independent pixel dropout, beyond the same two-copy corruption augmentation?

For each evaluation condition `(eta, p)`, training-only statistics `mu_i` and `var_i` define `d_i = sqrt(var_i + eta^2 + p(1-p) mu_i^2)`, floored at `1e-8`. We center by `mu_i` and divide by `d_i`. The primary candidate combines this transform with two fixed augmented training copies (one noise std .4 in normalized [0,1] units, one dropout p=.15). The control uses clean StandardScaler with the identical augmentation and an RBF SVC (`C=10`, `gamma='scale'`).

## Prior art and novelty status

Feature scaling before an RBF SVM is established: scikit-learn's [StandardScaler documentation](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) notes RBF kernel objectives can behave poorly when feature variances differ substantially. Corruption augmentation is established; e.g., [AugMix](https://arxiv.org/abs/1912.02781) and [NoisyMix (AISTATS 2024)](https://proceedings.mlr.press/v238/erichson24a.html) train with corrupted/noisy inputs. A nearby normalization idea is [per-corruption adaptation of normalization statistics (PAN)](https://arxiv.org/abs/2407.06450), which adapts normalization statistics to identified corruptions, though it uses online statistics in neural networks rather than this analytic fixed diagonal metric for an SVM.

Search did not establish whether this exact formula has appeared before. These sources establish ingredients and nearby approaches, not external novelty. Novelty is unestablished and requires broader literature search and human adjudication.

## Mechanism audit

For a random clean pixel `X` with mean `mu` and variance `v`, zeroing it with probability `p` yields variance `(1-p)v + p(1-p)mu^2`. Thus the proposal's term `v + p(1-p)mu^2` omits attenuation of clean variance. If Gaussian noise is added and then dropout is applied, its variance contribution is also attenuated by `(1-p)` (before clipping). The implementation tests the proposal's written expression, rather than silently correcting it. Input clipping to `[0,1]` also makes additive-noise variance only approximately `eta^2` near boundaries.

## Implementation and deviations

- `load_digits`, normalized to `[0,1]`; stratified 60/20/20 splits using seeds 137, 271, 389 (second split seed is first + 1).
- Nine fixed conditions: clean; noise std .2/.4/.6; dropout .05/.15/.30; joint (.4,.15) and (.6,.30). Each method shares the same corruption draw for each seed and condition.
- Envelope uses corresponding known evaluation noise/dropout parameters. This is condition-aware and requires selecting the scaler/model for the known condition; it is not blind robustness.
- Both augmented copies are generated only from training rows, fixed as noise (.4, 0) and dropout (0, .15). Primary control has identical augmentation.
- Ablations: envelope without augmentation; envelope with noise term removed; envelope with dropout term removed.
- Proposal wording mentioned a fixed split; protocol requires three seeds, so three fixed stratified splits were used. No external data or pretrained weights.
- Validation and final outputs are separate; final test output was created only after preregistration was frozen. No held-out tuning.

## Reproduction

From `experiments/alien-science/upstream`:

```sh
../.venv-benchmark/bin/python scripts/with_compute_slot.py -- ../.venv-benchmark/bin/python candidate-studies/digits_shift-direct-2/experiment.py validation --output candidate-studies/digits_shift-direct-2/validation.json
../.venv-benchmark/bin/python scripts/with_compute_slot.py -- ../.venv-benchmark/bin/python candidate-studies/digits_shift-direct-2/experiment.py final --output candidate-studies/digits_shift-direct-2/results.json
```

The script refuses to overwrite an output unless `--force` is provided. Each SVC fit uses one CPU thread. `preregistration.json` records split hashes, source hash, seeds, hyperparameters, primary metric, and decision rule.

## Validation and frozen test results

The primary endpoint is unweighted mean accuracy across nine conditions, averaged over three seeds. The preregistered pilot pass rule requires at least +0.01 over the augmented StandardScaler control and positive paired differences on all seeds. Validation means (candidate vs control) were .4302 vs .5518, .4444 vs .5531, and .4441 vs .5853. It failed on every seed; final evaluation proceeded as prespecified and was not tuned against.

See `validation.json` and `results.json` for per-condition scores and runtimes. Ablations separate the envelope from augmentation: envelope-only validation means were .7196/.7369/.7561, while removal of the noise term collapsed to near chance. These exploratory results show the combination performs worse than either the augmented StandardScaler or envelope-only validation models; the small study does not establish the cause.

## Smallest falsification test and verdict

The smallest useful test is the paired three-seed comparison of `envelope_aug` with `standard_aug` on nine fixed conditions. Validation already falsifies the preregistered practical claim that the envelope improves the augmented control by at least one percentage point. Envelope-only ablation can outperform the augmented control on validation, but combining envelope with the proposed augmentation hurts. This does not validate the candidate's stated combined mechanism.

Verdict: no supported pilot for the proposed combined intervention. The dataset is small, corruptions synthetic, `gamma='scale'`, and condition-specific model fitting assumes the corruption is known. No claim of external novelty or statistical significance is made.

Frozen test means for `envelope_aug` were .4511/.4523/.4287 versus .5782/.5761/.5839 for `standard_aug` (paired deltas -.1272/-.1238/-.1552). Across seeds the candidate mean was .4440 and control mean .5794. The final results are reported once for the frozen split; validation remains the only model selection evidence.
