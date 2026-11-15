# molecules-0 reconstruction study

Status: validation phase only. This directory studies a task-specific hypothesis; it does not establish a new result or novelty.

## Question and mechanism

The proposal asks whether augmenting native-NaN histogram gradient boosting with value-dependent self-masking groups, then upweighting groups with high training-only loss, improves prediction when the missingness mechanism shifts. The base training matrix is already MCAR-masked by the shared fixture. Additional masks are applied only to finite, observed training entries in columns 2 onward. For each such feature, an empirical CDF is estimated from finite training values. A positive-direction copy drops an entry with probability `sigmoid(4 * (q - .5))`; the reverse copy uses `sigmoid(-4 * (q - .5))`. The two anchor columns are never masked.

The proposal's written outer loop calculates group losses using validation labels and then refits the model. That would turn validation labels into fitting data. This implementation instead uses a fixed, stratified, three-fold calibration split within the training rows: for each round, a model fitted on two folds scores all four training-derived mask groups on the held-out fold, and group weights are updated as `w_g <- w_g * exp(eta * loss_g)`. After three rounds, one final HGB is fit on all training rows with the resulting group weights. No validation row or label enters reweighting. The HGB prediction interface receives only a masked matrix.

## Controls and ablations

All grouped arms use the same rows, labels, group count, and HGB capacity. The random-mask control replaces the two self-masking groups with independent extra masks at probability 0.5 on currently observed non-anchor entries, approximately matching the self-mask groups' marginal rate. The primary matched control is that random-mask model with the same training-only DRO procedure and selected `eta`. Additional comparisons are native-NaN HGB without augmentation, self-mask groups with uniform weights, one self-mask direction, and shuffled quantile ranks (preserving the rank distribution while removing each row's own-value association). A mask-blind median-filled HGB without missingness indicators is included as a descriptive ablation, not as the strongest control.

## Validation design

Only the shared fixture's validation cases are used: clean, MCAR 0.2, MCAR 0.4, and anchor-driven MAR. No additional validation masks are generated. The fixture's MNAR and block-missingness cases are reserved for the later final mode. Training corruption, sample counts, data splits and model capacity are fixed across arms. Validation selects `eta` from `{0.5, 1, 2}` by mean AP across the four shared validation conditions, with ties resolved toward smaller `eta`; this same `eta` is used for the primary random-mask DRO control. AP is the fixed primary metric, Brier is secondary. No condition name is passed into prediction.

## Prior art checked (primary sources)

- [Robust prediction under missingness shifts (Rockenschaub et al., 2024)](https://arxiv.org/abs/2406.16484): studies ignorable and non-ignorable missingness shifts; reports that missingness-agnostic methods can be preferable for highly informative non-ignorable shifts. Directly relevant to the problem setting and to the mask-blind control.
- [MIRRAMS: Learning Robust Tabular Models under Unseen Missingness Shifts (Lee et al., 2025)](https://arxiv.org/abs/2507.08280): addresses unseen test-time missingness shifts through mutual-information robustness conditions without an MCAR/MAR/MNAR assumption. Strong prior method; this study does not claim a full reproduction of its objective.
- [RoVTL (Hasny et al., 2025/2026)](https://arxiv.org/abs/2512.19602): uses tabular missingness augmentation for robustness in vision-tabular contrastive pretraining and downstream tuning. Its setting and random augmentation differ from value-dependent self-masks here.
- [REMIND (Wu et al., 2026)](https://arxiv.org/abs/2603.00046): combines modality-combination groups with group DRO for high-modality medical data. It establishes close prior art for group reweighting under missingness, though with modality groups and specialized fusion models.
- [Distributionally Robust Neural Networks for Group Shifts (Sagawa et al., 2019/2020)](https://arxiv.org/abs/1911.08731): foundational group-DRO treatment; emphasizes that training group loss alone can fail to capture worst-group generalization without regularization. Motivates held-out, training-only calibration and the need to avoid overclaiming from this small synthetic study.
- [NeuMiss networks (Le Morvan et al., 2020)](https://arxiv.org/abs/2007.01627): analyzes predictors under MAR and self-masking MNAR and explicitly models missingness indicators. Relevant mechanism-level context.
- [scikit-learn HistGradientBoosting missing-value documentation](https://scikit-learn.org/stable/modules/ensemble.html#histogram-based-gradient-boosting): documents that the grower learns which child receives missing values at each split, the specific behavior this intervention seeks to regularize.

These sources show substantial prior work on missingness-shift robustness, missingness augmentation, and group DRO. The candidate's exact combination has not been established as novel; this is not an exhaustive novelty search.

The proposal also named a “MIRRAMS-style MLP” comparator, but specified neither a network/objective nor the paper's MI robustness terms. A generic MLP would not be a faithful MIRRAMS control, so it was not included or labeled as one. The comparison instead prioritizes matched HGB controls that isolate the proposed components. A full MIRRAMS reproduction remains outside this phase.

## Assumptions, limits, and reproducibility

The narrow target assumption is monotone per-feature self-masking with stable `P(Y|X)`. The simulator uses only training-observed values and train-derived quantiles; this is an explicit refinement of the proposal's use of train-observed empirical CDFs. The fixture is synthetic and fixed, validation has no MNAR case, and this phase cannot substantiate robustness to unseen MNAR. The chosen HGB capacity and small number of seeds also limit inference. Primary outputs are saved by the experiment script.

## Validation result (2026-11-16)

All three preregistered seeds completed without errors. Mean AP across seed-condition pairs was 0.84125 for self-mask DRO and 0.83671 for the matched random-mask DRO control (paired mean difference `+0.00454`, short of the preregistered `+0.005` threshold). Clean AP did not decrease against that control. However, the candidate also trailed native-NaN HGB (0.84486) and the self-mask uniform-weight ablation (0.84256) on overall mean AP. Selected eta was 0.5 for seed 9401 and 2.0 for seeds 9403 and 9407. Thus the validation did not meet the preregistered utility threshold.

## Final result (2026-11-16)

After parent review, the all-seed final evaluation ran once. Across the six final conditions, candidate mean AP was `0.85998`, compared with `0.86360` for the matched random-mask DRO control (`-0.00363` paired mean), `0.86065` for self-mask with uniform weights, and `0.85800` for native-NaN HGB. The candidate's clean AP was `0.89746`, versus `0.91162` for native HGB; its AP was `0.84794` on own-feature MNAR and `0.85598` on reversed-sign MNAR. Candidate mean Brier was `0.06777`; random-mask DRO was `0.06726`.

The final verdict is `no_gain`. The final results do not change the validation threshold failure. This is a utility result for the corrected train-only-OOF implementation on this synthetic fixture; it establishes neither universal robustness nor novelty. See [results.json](results.json), [verdict.json](verdict.json), and the pre-final hash record [parent-approved-final.json](parent-approved-final.json). Do not rerun final mode.

Run validation from this directory's parent (`upstream`):

```bash
../.venv-benchmark/bin/python scripts/with_compute_slot.py -- ../.venv-benchmark/bin/python reconstruction-studies/molecules-0/experiment.py --mode validation
```

Final mode refuses to overwrite `results.json`. The authorized all-seed final run has completed; do not rerun it.
