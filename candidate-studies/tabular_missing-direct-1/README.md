# tabular_missing-direct-1 — candidate study

## Exact proposal and mechanism

The proposal is “Missingness-Pattern Prototype Routing with Per-Cluster Isotonic Recalibration”: train a HistGradientBoostingClassifier using median-imputed features plus missingness indicators; fit eight k-means prototypes over training missingness masks; fit an isotonic probability map per prototype on an independent labeled training fold; route each deployment row to its nearest Hamming-distance prototype and apply that map. Its stated causal story is that score calibration varies by missingness regime while within-regime ranking stays useful.

## Prior art checked

The proposal combines established pieces. Pattern-conditioned prediction is studied directly in [Muller, Scornet & Josse, “When Pattern-by-Pattern Works” (arXiv:2507.13024)](https://arxiv.org/abs/2507.13024), which analyzes learning separate models for missingness patterns. Held-out classifier calibration with isotonic regression is standard; see [scikit-learn calibration user guide](https://scikit-learn.org/stable/modules/calibration.html). Calibration under covariate shift has prior work, including [Park et al., “Unsupervised Calibration under Covariate Shift” (arXiv:2006.16405)](https://arxiv.org/abs/2006.16405). I found no evidence establishing novelty for the specific combination, and make no novelty claim.

## Mechanism audit and implementation

The proposal specifies clean-only training and contains no step that generates missingness-corrupted masks or supplies naturally missing training rows. Consequently every training mask is the all-zero vector. K-means finds one distinct cluster, not eight. All deployment masks therefore route to the same available calibrator (ties resolve to the lowest cluster ID). The proposed pattern-dependent effect cannot occur. The exact intervention is therefore not faithfully implementable as written.

To quantify the closest faithful execution without inventing a missingness-generation method, `experiment.py` fits the base model on one half of clean training data, fits isotonic regression on its predictions for the other clean half, and compares that collapsed single-map pipeline with the same base model without calibration. The architecture, split, data, and corruption draws are matched. This diagnostic is not evidence about the intended prototype-routing method. A routing-disabled ablation is mathematically identical to the candidate because only one prototype exists.

## Reproduction

From `experiments/alien-science/upstream`:

```sh
../.venv-benchmark/bin/python scripts/with_compute_slot.py -- ../.venv-benchmark/bin/python candidate-studies/tabular_missing-direct-1/experiment.py validation --output candidate-studies/tabular_missing-direct-1/validation.json
../.venv-benchmark/bin/python scripts/with_compute_slot.py -- ../.venv-benchmark/bin/python candidate-studies/tabular_missing-direct-1/experiment.py final --output candidate-studies/tabular_missing-direct-1/results.json
```

The final mode refuses to overwrite an existing result file. Seeds are 137, 271, 389; test split indices are constructed and hashed but not scored in validation. Splits are stratified 60/20/20, seed-specific. MCAR masks are shared between control and candidate at each seed and condition. Training/inference uses one CPU thread. See `preregistration.json` for the locked source hash, hyperparameters and validation split hashes.

## Results

The validation run completed for all three seeds before preregistration and the held-out run. K-means emitted the expected warning: only one distinct cluster among eight requested. Across all 12 held-out seed-condition pairs, the collapsed isotonic candidate mean AP was **0.7465**, versus **0.7858** for the uncalibrated matched control (mean paired difference **-0.0394**). Per-seed metrics and runtimes are in `results.json`; validation metrics are in `validation.json`.

Because AP measures ranking and a monotone calibrator can introduce ties, these AP results assess ranking impact, not calibration quality. They do not test the proposed conditional recalibration mechanism.

## Falsification and verdict

The proposal's stated support rule required at least +0.01 AP at 50%% missing and within ±0.005 at 0%%; its falsification rule required nonpositive improvement at each rate. The collapsed diagnostic showed no gains and was negative across the evaluated settings, but the primary verdict is **invalid mechanism**, because the eight-regime technique does not instantiate under the documented clean-only training procedure.

Smallest test that would resolve the design: specify exactly how the training/calibration fold obtains labeled examples under distinct missingness regimes (including mask generation probabilities and whether labels may be used), then rerun the preregistered three-seed matched comparison. Such a specification changes the proposal and requires a new preregistration.
