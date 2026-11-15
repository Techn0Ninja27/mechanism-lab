# Reconstruction studies pre-final audit — 2026-11-16

This review is static: it reads each experiment, preregistration, validation
artifact and shared fixture. It does not execute heavy code or inspect final
test labels. `approve_final` means the final gate is technically safe to run;
it does not mean the mechanism is novel or likely to win.

## Per-case decisions

| Case | Decision | Reason |
|---|---|---|
| abstracts-0 | approve_final | Prediction functions receive feature matrices only. Iterative-imputer statistics and augmentation masks are fit from masked training rows; validation labels are used only for metrics. Candidate and stronger constituent ablations are present. Normal validation selection is allowed. |
| abstracts-1 | approve_final | Teacher/student targets are generated from training-only masked views; prediction receives X only. Native, median, augmentation, K-view, raw-y and MCAR-only ablations are present. Validation selection is explicit. Confirm final mode preserves the same frozen method choice. |
| atoms-0 | approve_final | Pairwise Gaussian parameters, feature gate and augmentation are estimated from train data; the gate uses an internal train split, not held-out final data. Native/no-mask/mean and random-gate controls exist. Final code is parent-gated. |
| atoms-1 | approve_final | Candidate, uniform, indicator, anchor-only, no-anchor and native controls are present. Pool weights/temperature are selected on validation. Final function is intentionally blocked and does not expose test arrays yet. |
| molecules-0 | approve_final | Train-only empirical ranks and group masks; OOF group losses use training labels only. Uniform, random, one-direction, shuffled and mask-blind controls are present. Validation eta selection is normal. |
| molecules-1 | approve_final | Anchor rate model uses only masked train X. Train augmentation and prediction receive masked X only; uniform, k=1, permuted-anchor and indicator controls are present. Final is explicitly parent-gated. |

## Conditions before running final

1. Keep `scripts/missingness_shift_fixture.py` frozen at the recorded hash.
   The renamed final conditions (`block_mcar_2_8`, `block_mcar_8_14`) and
   `test_x=None` in validation are metadata/interface changes only; they must
   not be changed while fitting final models.
2. Run every final once into the predeclared `results.json` path. Preserve the
   validation artifacts and source hashes. Do not tune on final metrics.
3. Record the exact selected hyperparameter/variant from validation in the
   final receipt. Normal validation selection is not leakage; changing a choice
   after seeing final scores is.
4. For `abstracts-0`, check that the intended ADMIB arm is compared with the
   strongest `completed_aug_unweighted` and `mean_completed_aug` controls.
   For `abstracts-1`, compare candidate mixed-K8 against both augmentation-only
   and the K4/K1 ablations.
5. For `atoms-0` and `atoms-1`, verify final scoring uses only `final_cases`
   predictions and applies labels after prediction. The current code structure
   supports this; do not add condition names or labels to prediction helpers.
6. For `molecules-0`, retain the OOF group-loss history and selected eta. For
   `molecules-1`, retain anchor-rate bins, permutation rates and all ablation
   rows. These are needed to tell a relation gain from extra masking.

No code currently shows a hidden final-label training oracle in the reviewed
phase-one paths. The main residual risks are ordinary validation-selection
overfitting, the tiny synthetic fixture, and whether the chosen control is the
strongest constituent baseline. Final results must remain descriptive until
independent data and prior-art review are added.
