# digits_shift-direct-1 — blur-first SVM study

## Proposal and mechanism

The candidate proposes Gaussian smoothing (σ=0.8) on every image before standardization, with the scaler fitted only on clean training images. Add two matched training copies, one with Gaussian pixel noise (σ=0.15×16) and one with 15% pixels dropped to zero. At inference, use the same blur→scaler pipeline. The stated mechanism is that blur suppresses high-frequency noise and spatially fills dropped pixels, while augmentation teaches the SVM boundary to tolerate corruptions.

## Prior art and novelty status

The components are known. Rusak et al. study Gaussian noise augmentation for corruption robustness ([paper](https://arxiv.org/abs/2001.06057)). Saikia et al. show frequency-selective effects from Gaussian blur/noise augmentation and associated clean-data tradeoffs ([ICCV paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Saikia_Improving_Robustness_Against_Common_Corruptions_With_Frequency_Biased_Models_ICCV_2021_paper.pdf)). Lopes et al. analyze how similarity between training augmentations and test corruptions predicts robustness ([paper](https://arxiv.org/abs/2102.11273)). These establish the general techniques, not this exact small-data blur-before-scaling pipeline. No external novelty is established.

## Implementation and deviations

`experiment.py` compares baseline, blur only, augmentation only, and both. Each arm uses the same stratified 60/20/20 split for seeds 137, 271, and 389, RBF SVC (C=10, gamma=scale), and fixed corruption draws. StandardScaler fits only clean training samples (raw for baseline/augmentation-only, blurred for blur arms); augmentation is applied after fitting the scaler and before model fitting. All transforms are fit on training data only. The proposal originally listed five conditions; the experiment uses the required nine: clean, three noise levels, three dropout levels, and two joint levels. Gaussian smoothing uses SciPy `gaussian_filter` with σ=0.8 and reflect boundary handling.

A first implementation accidentally fit the scaler after augmentation. Its validation and test outputs are preserved as `validation_invalid_v1.json` and `results_invalid_v1.json`, explicitly excluded from conclusions. The corrected implementation was preregistered in `preregistration.json`, validated on all seeds, and then evaluated once on held-out test splits. The corrected source SHA256 is recorded in the JSON artifacts.

## Reproduction

From `experiments/alien-science/upstream`:

```sh
../.venv-benchmark/bin/python scripts/with_compute_slot.py -- ../.venv-benchmark/bin/python candidate-studies/digits_shift-direct-1/experiment.py validation --output candidate-studies/digits_shift-direct-1/validation.json
../.venv-benchmark/bin/python scripts/with_compute_slot.py -- ../.venv-benchmark/bin/python candidate-studies/digits_shift-direct-1/experiment.py final --output candidate-studies/digits_shift-direct-1/results.json
```

Final mode refuses to overwrite an existing output. The script limits numerical-library threads to one through the shared compute-slot gate. All experiment runs took under four seconds of measured Python wall time per full mode in this environment.

## Results

Primary metric: mean accuracy across nine conditions, averaged across the three seeds.

| Arm | Test mean accuracy | Per-seed means (137, 271, 389) |
|---|---:|---|
| Baseline | 0.4702 | 0.4694, 0.4716, 0.4694 |
| Blur only | 0.5363 | 0.5278, 0.5429, 0.5383 |
| Augmentation only | 0.5425 | 0.5259, 0.5426, 0.5590 |
| Blur + augmentation | 0.6492 | 0.6509, 0.6509, 0.6457 |

Combined-arm paired deltas over baseline were +0.1815, +0.1793, and +0.1762. Clean accuracy was 0.9861 combined versus 0.9880 baseline. The combined method therefore passes the preregistered pilot rule. Per-condition results and train/inference times are in `results.json`; validation results and preregistered thresholds are in `validation.json` and `preregistration.json`.

The mechanism is more mixed than the aggregate result suggests. On noise σ=0.6×16, augmentation only (0.1324) exceeds blur only (0.1120), falsifying the proposal’s own high-noise causal discriminator. Blur alone falls from 0.8194 baseline accuracy to 0.7333 on 30% dropout, consistent with smoothing damaging spatially localized evidence. The combination is much better on noise 0.2/0.4 and both joint conditions, yet is not best on the hardest dropout-only condition.

## Smallest falsification test and verdict

The smallest preregistered mechanism test was to compare blur-only and augmentation-only at noise_0.6 over the three validation splits. Augmentation-only scored 0.1411 versus 0.1077 for blur-only, so the proposed blur-based distance repair is not the dominant explanation at that severity. The full held-out experiment nevertheless shows an aggregate pilot gain for the combined arm. Verdict: `supported_pilot` for this fixed synthetic benchmark and pipeline, with the blur causal story specifically weakened/falsified by the ablation. See `verdict.json`.

## Limitations

This is a tiny dataset with synthetic pixel corruptions and only three split seeds. Noise σ=0.2×16 already makes the unaugmented baseline near chance, so the high aggregate gain partly reflects a deliberately harsh corruption distribution. These are paired exploratory replications, not evidence of statistical significance or external novelty. Runtime numbers are local and hardware-dependent. No failures occurred in the corrected runs.
