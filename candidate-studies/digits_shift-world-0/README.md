# digits_shift-world-0: Corruption-Variance Pixel Weighting

## Proposal and mechanism

The proposal estimates each pixel's variance over ten train-only corruptions at each target noise/dropout severity, sets `W[i] = 1/(1+variance[i])`, sets `W[i]=0` when the corrupted-pool mean is outside [2,14], scales raw pixel values by W, then fits a StandardScaler and an RBF SVM. Training augmentation is suppressed on collapsed pixels. The stated mechanism is that uncertain dimensions distort the kernel metric and should contribute less to pairwise distance.

## Prior art and novelty status

Feature scaling as a learned diagonal metric for kernel SVMs, including zero weights for feature selection, is established work: Canu & Grandvalet, *Adaptive Scaling for Feature Selection in SVMs* (NeurIPS 2002), https://proceedings.neurips.cc/paper_files/paper/2002/hash/ee26fc66b1369c7625333bedafbfcaf6-Abstract.html. Input uncertainty in SVM classification is also directly studied by Zhang & Bi, *Support Vector Classification with Input Data Uncertainty* (NeurIPS 2002), https://proceedings.neurips.cc/paper/2743-support-vector-classification-with-input-data-uncertainty.pdf. Feature selection and scaling factors in multiclass SVMs are covered by Xu, Nie & Han, IJCAI 2017, https://www.ijcai.org/Proceedings/2017/0442.pdf.

The exact heuristic pairing training-corruption variance weights with the [2,14] mean cutoff was not located in this bounded search. That does not establish novelty. Its main weighting claim is mathematically null under the specified preprocessing order: for a nonzero scalar w_i, StandardScaler(w_i*x_i) returns the same standardized feature as StandardScaler(x_i) (up to numerical precision). The zero weights do survive as constant dimensions. Thus the intervention actually tests a corruption-derived pixel mask, not continuous reliability weighting. The variance-only ablation matches the control exactly; candidate matches the collapsed-mask-only ablation exactly. This is a central mechanism limitation, not an implementation substitution.

## Implementation and procedure

`experiment.py` uses sklearn `load_digits`, stratified 60/20/20 splits for seeds 137, 271, and 389, corruption draws shared by all methods, train-only estimation of weights and collapsed pixels, one CPU thread, and the fixed conditions in `preregistration.json`. The control is the same clean-plus-noise/dropout augmentation pool with all pixels retained. Ablations remove the collapsed mask or the continuous weights. The preregistered C=10, gamma=0.01 were selected by the control's mean validation accuracy over the fixed grid before final evaluation. `validation.json` contains only validation metrics; `results.json` contains the one held-out pass. No test score was used to tune parameters.

The candidate pools use ten copies at each target severity to estimate variance and two augmentation copies (noise .2, dropout .15), matching the proposal's cheap augmentation description. StandardScaler is fit on the resulting training pool, then applied to held-out data. This ordering is required by the proposal and is what causes the continuous weights to cancel. The diagnostic correlation in the final result compares train-derived per-pixel variance with held-out covariate variance; it uses no held-out labels and did not select the method.

Run from `experiments/alien-science/upstream`:

```sh
../.venv-benchmark/bin/python scripts/with_compute_slot.py -- ../.venv-benchmark/bin/python candidate-studies/digits_shift-world-0/experiment.py validation --output candidate-studies/digits_shift-world-0/validation.json
# preregistration.json freezes selection, source hash, and split hashes before the held-out run
../.venv-benchmark/bin/python scripts/with_compute_slot.py -- ../.venv-benchmark/bin/python candidate-studies/digits_shift-world-0/experiment.py final --output candidate-studies/digits_shift-world-0/results.json
```

The two measured runs took 12.0s validation and 13.3s held-out on this environment. `experiment.py` refuses to overwrite an existing final output.

## Held-out result

Primary metric is the average accuracy across clean, three noise, three dropout, and two joint conditions. For the frozen hyperparameter choice:

| Seed | Control | Candidate | Paired delta |
|---:|---:|---:|---:|
| 137 | 0.7203 | 0.8233 | +0.1031 |
| 271 | 0.7047 | 0.8206 | +0.1158 |
| 389 | 0.7172 | 0.8119 | +0.0947 |
| Mean | 0.7141 | 0.8186 | +0.1045 |

All three held-out seeds favored the mask-containing methods. But candidate equals collapsed-mask-only (per-seed and per-condition), while variance-weight-only equals control. The mean clean accuracy is about 0.993 for both control and candidate, so the mask did not reduce clean accuracy here. Per-condition results and runtimes are in `results.json`. Each fit took roughly 0.05–0.10s and the ten-condition inference pass roughly 0.20–0.29s. The train variance / held-out noise-instability correlations were about 0.983; that association does not show that weighting helps.

## Verdict and cheapest falsification

The composite proposal clears its preregistered positive-delta pilot rule, but its continuous variance-weight mechanism is falsified by the exact invariance of subsequent per-feature standardization and by the matching ablation. The measured gain is attributable to the zeroed-pixel mask, an established feature-selection-like operation; it does not establish external novelty. The smallest falsification is the variance-weight-only ablation with StandardScaler after weighting: if it differs from the unweighted control, either the scaler/order differs or numerical/pathological details are at play. Here it matched exactly. A separate evaluation would be needed to assess the mask outside these three splits and this corruption distribution.
