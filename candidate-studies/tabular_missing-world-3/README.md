# tabular_missing-world-3 — candidate study

## Proposal and mechanism

The proposal is **Asymmetric Missingness Curriculum with Weighted Mask Indicators for Deployment-Shift Robustness**. It posits that training copies with clean, observed-rate (10%), then elevated-rate (chosen as 30%) missingness expose a classifier to deployment corruption; appended binary masks make absence explicit; training-only mutual information (MI) scales mask columns; and training medians are frozen for imputation.

This is a test of the proposed composite intervention. The actual dataset is synthetic and MCAR, so it cannot establish performance under realistic MAR/MNAR missingness or changing feature-specific collection policies.

## Prior art checked (16 Nov 2026)

- **Median imputation plus mask is established.** A primary benchmark across real health databases explicitly compared median and median+mask approaches, found mask indicators useful after imputation, and also found that boosted trees with native missing-value handling can perform well: [Benchmarking missing-values approaches for predictive models on health databases (2022)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9012100/). Therefore, neither median imputation nor explicit mask features are new components.
- **Robustness to shifted missingness is already directly studied.** [MIRRAMS: Learning Robust Tabular Models under Unseen Missingness Shifts (2025)](https://arxiv.org/abs/2507.08280) targets changing test-time missingness, using mutual-information robustness conditions and loss terms. It is a different deep-learning mechanism but establishes close prior art for the problem and MI connection.
- **Learner limitation checked.** scikit-learn's [HistGradientBoostingClassifier documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html) documents native NaN routing and defines warm-start as reusing a previous fit to add estimators. Its warm-start glossary requires refitting on the same data for valid results. Thus sequentially changing the training corpus between boosting phases is not valid for this learner.

The unverified novelty-shaped relation here is the combination of a three-rate cumulative augmentation schedule with MI-scaled mask columns for expected deployment MCAR. This small synthetic study is not a novelty search and does not establish that this relation is new.

## Implementation and deviation

- `experiment.py` implements train-only medians, train-only `mutual_info_classif`, mask augmentation and four methods: median+HGB control, weighted masks plus curriculum, uniform masks plus curriculum, and weighted masks with elevated-rate augmentation but without the 10% intermediate phase.
- All methods share the same generated splits and deterministic test corruption draws. HGB capacity is fixed at 100 iterations, 15 leaves, 20 minimum leaf samples, L2=1, no early stopping. A single CPU thread is used.
- **Material limitation/deviation:** HGB is not an online learner. In place of repeated optimizer phases, the candidate fits once on a cumulative training corpus containing clean, 10%, and elevated-rate masked replicas. This implements staged *exposure strata*, not a sequential curriculum in optimizer time. We did not replace HGB with another algorithm.
- Validation compared elevated rates 30% and 50%. Weighted-mask mean validation AP was 0.864026 at 30% and 0.864078 at 50%, a negligible gap. The preregistration froze 30% as the deployment-rate setting before the final output was inspected.

## Reproduction

From `experiments/alien-science/upstream`:

```sh
../.venv-benchmark/bin/python scripts/with_compute_slot.py -- ../.venv-benchmark/bin/python candidate-studies/tabular_missing-world-3/experiment.py validation --output candidate-studies/tabular_missing-world-3/validation.json
../.venv-benchmark/bin/python scripts/with_compute_slot.py -- ../.venv-benchmark/bin/python candidate-studies/tabular_missing-world-3/experiment.py final --output candidate-studies/tabular_missing-world-3/results.json
```

Final mode refuses to overwrite an existing result path. Results, split hashes, seeds and frozen configuration are stored in the adjacent JSON files. The source hash in `preregistration.json` identifies the code used for evaluation. The chosen configuration and decision rule were fixed after validation and before final execution, but a path error delayed writing the preregistration file until after the final process completed. Held-out metrics were not inspected until after that file was written; this procedural deviation means the result is exploratory rather than strictly preregistered.

## Results

Primary metric is mean AP over clean/10%/30%/50% test conditions, equally weighted within seed, then equally averaged over seeds.

| Seed | Control HGB | Weighted curriculum | Uniform mask ablation | No-intermediate-phase ablation | Weighted − control |
|---:|---:|---:|---:|---:|---:|
| 137 | 0.698951 | 0.734619 | 0.736645 | 0.728985 | +0.035668 |
| 271 | 0.827038 | 0.888371 | 0.888329 | 0.882859 | +0.061334 |
| 389 | 0.790068 | 0.893483 | 0.893367 | 0.876623 | +0.103415 |
| **Mean** | **0.772019** | **0.838824** | **0.839447** | **0.829489** | **+0.066806** |

Candidate training averaged 0.313 seconds and inference 0.0033 seconds per condition; control averaged 0.188 seconds training and 0.0037 seconds inference. Full final run took 3.36 seconds. No variant failed.

The composite candidate beat the no-augmentation control on all three seeds. However, MI-weighted masks did **not** beat uniform mask weights on average (0.838824 vs 0.839447), and the weighted-minus-uniform paired differences were -0.002026, +0.000042, +0.000116. The extra intermediate phase versus elevated-only augmentation improved mean AP by 0.009335, with positive differences on all seeds (+0.005634, +0.005512, +0.016860). This supports a small pilot signal for augmentation/curriculum exposure, not for MI weighting as a special mechanism.

## Verdict and smallest falsification test

The composite candidate clears the preregistered pilot rule against the matched baseline. The MI-weighting subclaim fails its intended ablation: uniform weights are at least as good. The result is an **exploratory supported pilot** for augmentation under this synthetic MCAR setup, not evidence of external novelty or real-world shift robustness. See `verdict.json`.

The smallest direct falsification was the three-seed, four-condition held-out comparison against the same HGB control. It did not falsify composite augmentation; the uniform-weight ablation does falsify any claim that MI scaling was responsible for the improvement here. A decisive next check would use realistic naturally shifted missingness and a larger, independently held-out collection of datasets, but this study does not establish those results.
