# Learned missing-value scalars with augmentation

## Proposal and mechanism

The candidate proposes a PyTorch MLP with 20 trainable feature-specific replacement values initialized to training medians. During training, each row receives an independently sampled MCAR rate from Uniform(0, 0.4); masked features are replaced by the learned values, and the binary mask is concatenated to the inputs. The claim is that classification loss can move replacement values away from medians when a different substitution better preserves the decision function, while the mask lets the classifier condition on missingness.

The underlying ingredients are established. NeuMiss derives missingness-aware predictors and explicitly uses the mask in its architecture; Le Morvan et al. show joint prediction and imputation can help finite-sample prediction; GRAPE learns downstream labels directly from missing-value patterns. These papers do not establish that this proposal's particular 20-scalar parameterization helps. The implementation is therefore tested as a narrow incremental mechanism, not described as novel.

## Prior art checked

- Le Morvan et al., [NeuMiss networks: differentiable programming for supervised learning with missing values](https://arxiv.org/abs/2007.01627), NeurIPS 2020. Mask-dependent architecture; the mask itself is central to the predictor.
- Le Morvan et al., [What's a good imputation to predict with missing values?](https://proceedings.neurips.cc/paper_files/paper/2021/hash/5fe8fdc79ce292c39c5f209d734b7206-Abstract.html), NeurIPS 2021. Studies the predictive objective of imputation and joint imputation/regression.
- You et al., [Handling Missing Data with Graph Representation Learning (GRAPE)](https://proceedings.neurips.cc/paper/2020/hash/dc36f18a9a0a776671d4879cae69b551-Abstract.html), NeurIPS 2020. Learns feature imputation and downstream label prediction from incomplete data.
- Choudhury & Pal, [Imputation of missing data with neural networks for classification](https://doi.org/10.1016/j.knosys.2019.07.009), Knowledge-Based Systems 2019. Uses supervised classification and neural imputations.

This is enough to establish close prior art for supervised imputation, mask-aware prediction, and augmentation-style robustness. It does not prove the exact trainable-constant-plus-mask combination was previously published; that narrower search remains incomplete, and no novelty claim is warranted.

## Implementation and deviations

Dataset: sklearn `make_classification(5000, 20, 5 informative, 5 redundant, weights=[.8,.2], class_sep=1.0)`, using each prescribed seed (137, 271, 389) for data and stratified 60/20/20 splits. Imputation values are train medians. The MLP is [40,64,32,1], BatchNorm, ReLU, dropout .2, Adam 1e-3, batch 256; maximum 80 epochs with patience 10. Metrics are equal-weight mean AP across clean, 10%, 30%, and 50% MCAR. The primary comparison is against the same MLP with median fill values frozen, with missingness mask and augmentation retained. Other controls remove mask inputs or augmentation. HGB median-impute + HistGradientBoosting (120 iterations) is also reported.

**Protocol deviations invalidate a confirmatory interpretation.** The script sampled fresh Torch augmentation masks separately for each arm rather than replaying identical training masks. Early stopping used clean validation AP rather than the preregistered mean AP over all four validation conditions. Also, the augmentation-level draws are per-example rates as specified, but final/test condition masks are generated separately with matched seeds. These defects were noticed after the sole held-out run; the held-out scores are retained without rerun, and this study is marked inconclusive. Validation scores did not tune parameters beyond the stated fixed configuration. Per-condition scores remain in the JSON artifacts.

## Reproduction

From `experiments/alien-science/upstream` in the repository, with the provided environments:

```sh
../.venv-benchmark/bin/python candidate-studies/tabular_missing-direct-0/prepare.py
../.venv-benchmark/bin/python scripts/with_compute_slot.py -- ../.venv-benchmark/bin/python -c "import subprocess; subprocess.run(['../.venv-benchmark/bin/python','candidate-studies/tabular_missing-direct-0/prepare.py'],check=True); subprocess.run(['../.venv-benchmark/bin/python','candidate-studies/tabular_missing-direct-0/score_hgb_final.py'],check=True); subprocess.run(['../.venv/bin/python','candidate-studies/tabular_missing-direct-0/experiment.py','final','--output','candidate-studies/tabular_missing-direct-0/results.json'],check=True)"
```

The first command only creates fixtures and validation baseline scores. The second command is the historical one-shot held-out workflow and should not be rerun for scientific interpretation. The source and split hashes frozen before held-out scoring are in `preregistration.json`. Experiment CPU usage was under the five-minute budget; each model used one Torch CPU thread.

## Results

Exploratory held-out mean AP over four conditions:

| Method | Seed 137 | Seed 271 | Seed 389 | Mean |
|---|---:|---:|---:|---:|
| Learned scalars + mask + augmentation | 0.79312 | 0.90698 | 0.91925 | 0.87312 |
| Fixed medians + mask + augmentation | 0.79920 | 0.89637 | 0.92358 | 0.87305 |
| No mask input | 0.77510 | 0.89135 | 0.87783 | 0.84809 |
| No augmentation | 0.71493 | 0.83722 | 0.78120 | 0.77778 |
| Median imputation + HGB | 0.71927 | 0.84571 | 0.78448 | 0.78316 |

Paired learned-minus-fixed differences: seed 137 **-0.00608**, seed 271 **+0.01062**, seed 389 **-0.00434**; mean **+0.00007 AP**. This is far below the preregistered +0.01 pass threshold and is positive on only one of three seeds. Validation paired deltas were -0.00843, +0.00779, -0.00277 (mean -0.00114). The fixed-median network being essentially tied with learned scalars is evidence against a useful scalar contribution in this pilot; however, mask replay and validation-selection defects prevent a clean causal conclusion.

Torch training times for the learned arm were 0.70s, 0.32s, and 0.33s; inference time was not separately recorded. JSON files contain per-condition metrics, split hashes, and timings. The AP aggregation has equal weight per condition, then equal weight per seed.

## Smallest falsification test and verdict

The direct smallest falsification is a properly matched paired run: replay identical augmentation masks and model initialization across learned-scalar and frozen-median arms, early-stop on mean validation AP across the four conditions, then evaluate once on the still-unused held-out split. A negligible learned-vs-fixed delta would falsify the claimed scalar benefit for this setup. The current exploratory held-out delta is +0.00007 and does not meet the preregistered criterion.

Verdict: **inconclusive** due to the protocol deviations. This does not establish external novelty or generalize beyond this synthetic MCAR task.
