# digits_shift-world-1 study

## Verdict

**Supported pilot, with the gain isolated to the PCA branch.** Across three held-out stratified splits, the full arm averaged **0.7917** accuracy over nine corruption conditions versus **0.4806** for the matched augmented RBF-SVM control. Paired per-seed gains were **+0.3059, +0.3136, +0.3139**. The PCA-only ablation scored **0.7954**, slightly above the full method. Centroid anchoring was neutral to slightly negative, and the composite selection rule chose the same hyperparameters and scores as validation-only selection.

This is a small exploratory replication, not evidence of external novelty or statistical significance. The large gain is specifically from PCA reconstructing low-margin raw images before classification; it should not be credited to the entire bundled method.

## Candidate mechanism and implementation

The proposal joins three operations:

1. Replace the impossible shared train/validation gradient with a hyperparameter score averaging clean-train 3-fold CV accuracy and held-out validation accuracy.
2. Generate noise and dropout augmentations, then pull each standardized augmented vector toward its clean training class centroid by a severity-scaled interpolation coefficient.
3. At inference, use a low SVC decision value as a confidence proxy; reconstruct only those samples from a rank-32 PCA basis fit on clean training images.

The baseline is a train-standardized RBF SVC, fit on clean training samples plus the same noise and dropout augmentation views without centroid anchoring. The baseline selects C and gamma by validation accuracy and performs no PCA reconstruction. All compared arms share the split, augmentation draws, scaler, SVC class, hyperparameter grid, and test corruptions.

The proposal did not specify the two augmentation severities. This study uses the shared fixture convention: one Gaussian noise view at sigma `0.15 * 16` and one pixel-dropout view at `0.15`, in addition to clean samples. Centroid pull weights are severity-normalized (noise severity / `0.6`; dropout rate / `0.30`). This is a documented instantiation of the underspecified augmentation detail.

The exact test conditions are clean; noise `0.2`, `0.4`, `0.6`; dropout `0.05`, `0.15`, `0.30`; and joint `(0.4, 0.15)`, `(0.6, 0.30)`. Noise is scaled by 16 and images are clipped to `[0,16]`. Each condition receives an independent, deterministic corruption draw, shared by every method.

## Prior art checked

- [Test-Time Adaptation with Principal Component Analysis (Cordier et al., 2022)](https://arxiv.org/abs/2209.05779) already projects representations into a PCA basis, suppresses components, reconstructs them, and evaluates corruption robustness. Its approach learns a spectral filter at test time on CNN features; this candidate instead applies fixed rank truncation to raw 8x8 input and gates it using a validation-calibrated SVC score. The mechanism family is established; this study does not establish novelty for the simpler variant.
- [Neural networks become centroid classifiers when trained on heavily corrupted inputs (Tahmassebpur et al., 2026)](https://arxiv.org/abs/2606.11319) derives and tests nearest-class-mean behavior under severe training-input corruption. That is adjacent evidence that class means can remain predictive under corruption; it does not test the candidate's operation of moving each augmented example toward its clean-label class mean.
- [PCA API and mathematical behavior (scikit-learn documentation)](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) confirms PCA projection and inverse transformation as the low-rank reconstruction implemented here.
- [On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation (Cawley & Talbot, 2010)](https://www.jmlr.org/papers/v11/cawley10a.html) documents selection-criterion overfitting and selection bias. The candidate's equally weighted CV-plus-validation score is not itself validated by this source; it is a hypothesis tested here, and it did not change selections on these splits.
- Prototype/class-centroid methods and feature augmentation are established families. The source search did not identify a primary paper matching this exact linear centroid pull applied to corruption-augmented pixels for an RBF-SVM on digits. That limited search is not evidence of novelty.

## Results

Primary metric: equal-weight mean accuracy across nine conditions, then equal-weight mean across seeds.

| Seed | Control | Composite only | Centroid only | PCA only | Full | Full − control |
|---:|---:|---:|---:|---:|---:|---:|
| 137 | 0.4852 | 0.4852 | 0.4833 | 0.7941 | 0.7910 | +0.3059 |
| 271 | 0.4682 | 0.4682 | 0.4731 | 0.7833 | 0.7818 | +0.3136 |
| 389 | 0.4883 | 0.4883 | 0.4846 | 0.8086 | 0.8022 | +0.3139 |
| **Mean** | **0.4806** | **0.4806** | **0.4803** | **0.7954** | **0.7917** | **+0.3111** |

The PCA arm primarily helps additive noise. Mean control accuracy for noise `0.2/0.4/0.6` was `0.157/0.112/0.103`; PCA-only was `0.958/0.805/0.557`. For dropout `0.05/0.15/0.30`, control was `0.969/0.929/0.846` and PCA-only was `0.971/0.928/0.842`. At joint `(0.4,0.15)` and `(0.6,0.30)`, control was `0.121/0.104` and PCA-only `0.699/0.412`. Thus the gain is not broad improvement across all corruptions; it is a strong recovery under Gaussian noise and joint severe corruption.

Validation mean scores were control `0.4750`, composite-only `0.4750`, centroid-only `0.4728`, PCA-only `0.7832`, and full `0.7809`. The selected `(C, gamma)` values are in `validation.json` and `results.json`; the six preregistered candidates were `C ∈ {1,10}`, `gamma ∈ {0.01,0.1,scale}`.

Final-mode elapsed time was 28.84 seconds total. Training plus hyperparameter selection took about 1.3–2.0 seconds per arm and seed; inference over all nine conditions took 0.21–0.56 seconds. Validation took 28.86 seconds total. One earlier, larger-grid run exceeded the initial compute budget and was interrupted before producing results or accessing test metrics; the reduced six-point grid was fixed before final evaluation.

## Ablation interpretation and limitations

- Composite selection versus validation-only: identical selections and metrics in all three seeds.
- Centroid anchoring: no aggregate gain; `anchor_only` mean was `0.4803` against `0.4806` control.
- PCA reconstruction: the main contributor; PCA-only mean was `0.7954`.
- Full method: positive paired gain on all three seeds, but 0.0037 below PCA-only on average. Added components do not improve the PCA result in this pilot.
- The PCA trigger is the maximum of scikit-learn's transformed one-vs-rest decision scores, not a calibrated probability or canonical geometric margin.
- scikit-learn emitted divide-by-zero, overflow, and invalid-matmul `RuntimeWarning`s during PCA transforms/reconstruction. Reconstructed arrays and stored metrics remained finite, but this should be diagnosed before relying on the method.
- The result uses one small dataset, one classifier family, three splits, and an intentionally limited hyperparameter grid. It does not establish robustness to natural corruptions or new datasets.
- An initial larger-grid run with six separate augmentation severities was interrupted after it exceeded the initial compute cap. It produced no reportable variant metrics and did not access the held-out test split; the final grid and matched augmentation convention were fixed before final evaluation.

## Reproduction

From `experiments/alien-science/upstream`, with the provided benchmark environment:

```sh
../.venv-benchmark/bin/python scripts/with_compute_slot.py -- ../.venv-benchmark/bin/python candidate-studies/digits_shift-world-1/experiment.py validation --output candidate-studies/digits_shift-world-1/validation.json
../.venv-benchmark/bin/python scripts/with_compute_slot.py -- ../.venv-benchmark/bin/python candidate-studies/digits_shift-world-1/experiment.py final --output candidate-studies/digits_shift-world-1/results.json
```

The frozen parameters, pass/fail rule, source hash, and split hashes are in `preregistration.json`. The final run was made once per seed. `results.json` contains per-condition values, per-seed summaries, and runtimes; `verdict.json` contains the machine-readable verdict. Do not rerun final mode over the frozen file without an explicit `--force` request.

## Smallest falsification test

The preregistered test was the paired full-versus-control mean accuracy on the three held-out seed splits. It would falsify a positive pilot effect if any one seed failed to improve and the mean paired delta were nonpositive. The observed deltas were positive on all three seeds. For the proposed *combined* mechanism, the ablations already falsify the claim that centroid anchoring or composite selection contributed to the observed gain in this setup: PCA-only matched or exceeded the full arm, and composite-only matched control exactly.
