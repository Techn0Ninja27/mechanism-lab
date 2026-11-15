# abstracts-1: mask-marginalized teacher/student reconstruction

Status: phase one and parent-approved phase two completed. Final results remain descriptive because the candidate failed the preregistered validation gate.

## Mechanism and interpretation

The proposal trains a HistGradientBoosting teacher on the observed training rows plus eight further-masked views per row. Each synthetic mask only hides values already present in the fixture's masked training matrix; it cannot reveal or use hidden clean values. The student target is the mean teacher probability over those eight synthetic views and the student regresses its clipped logit on the original masked row. Anchors 0 and 1 stay observed. The declared intervention is target distillation from synthetic-mask-marginalized teacher probabilities.

There is a material mechanism problem in the proposal. Its stated “indicator shuffle” cannot break the mask-to-label path: the indicators are not passed to the model, while NaN locations themselves are visible to HistGradientBoosting. Shuffling a separately computed indicator matrix that is never consumed is an algebraic no-op. The student can still recover missingness patterns from NaN locations, and targets depend on the original row's observed features. The validation script therefore logs the indicator-shuffle ablation as an explicitly identical no-op and includes median-imputed HGB as an explicit mask-blind control. The experiment tests utility of the implementable distillation procedure; it does not validate the claimed mask-invariance mechanism.

The proposal's isotonic calibration on validation MCAR20 would reuse validation labels for both fitting and scoring. It is omitted; all methods use uncalibrated model probabilities. The clean/MCAR/MAR validation metrics are descriptive, and no parameters are selected from these metrics. This follows the frozen validation-only phase-one instruction and avoids validation-label leakage.

## Prior work checked (primary sources)

- Rockenschaub et al., [Robust prediction under missingness shifts](https://arxiv.org/abs/2406.16484): distinguishes ignorable shifts (where the Bayes predictor is unchanged) from non-ignorable shifts and reports that disregarding missingness can help when it is highly informative. This is motivation, not support for the proposed teacher/student algorithm.
- Catoire et al., [Prediction with Missing Data: Target Probabilities and Missingness Mechanisms](https://arxiv.org/abs/2603.17599): distinguishes targets defined with and without using observation indicators and conditions under which they coincide. This makes target definition central; it does not establish that the proposed distillation removes indicator dependence.
- Lee et al., [MIRRAMS: Learning Robust Tabular Models under Unseen Missingness Shifts](https://arxiv.org/abs/2507.08280): directly studies robustness under unseen missingness shifts with mutual-information conditions and losses. Close prior art on the goal; its method is not reproduced here.
- Le Morvan et al., [What's a good imputation to predict with missing values?](https://arxiv.org/abs/2106.00311): proposes joint imputation/regression (NeuMiss) and analyzes prediction under missingness. It does not test the specific shift-robustness claim here.

These abstracts establish substantial prior art for robust prediction with missingness shifts and mask-independent targets. No novelty claim is made. Full-text/source inspection confirmed the abstracts and framing; this small study is not a systematic literature review.

## Assumptions and fixture

The synthetic fixture provides 5,000 samples, 20 features, labels on training and validation splits, and always-observed anchors 0 and 1. Training inputs are MCAR20-masked; validation has clean, MCAR20, MCAR40, and anchor-MAR cases. The method assumes the relevant shift changes only the observation mechanism while the clean feature and label relation remains fixed. This assumption does not justify robustness to MNAR or shifts in feature/label distributions. The learner never accesses fixture-clean training values, test values, target labels, or condition names.

## Frozen methods and controls

- Candidate: HGB teacher on original plus K=8 supermasked training views; HGB regressor student on original NaN-bearing inputs with mean clipped teacher logits.
- Strong constituent controls: native-NaN HGB; median-impute (no indicator) then HGB; direct HGB trained on the same original-plus-eight-view augmentation.
- Ablations: teacher probabilities replaced with raw labels; indicator-shuffle operation (expected identical/no-op); MCAR-only versus mixed MCAR/group masks; K=1 and K=4 target views. Training views are deterministically generated from each seed; all base HGB configurations use the same tree capacity (`max_iter=100`, `max_leaf_nodes=15`, `learning_rate=0.1`, `l2_regularization=1`).

All model calls receive only a feature matrix; the prediction interface has no condition-name or label parameter. Validation runs use `load_fixture(..., mode="validation")`. The separate final mode refuses to overwrite `results.json`; parent approval was recorded in `parent-approved-final.json` before it was run. The frozen final fixture returns `test_x=None`; final scoring uses `final_cases` only.

## Reproduce phase one

From `experiments/alien-science/upstream`:

```bash
../.venv-benchmark/bin/python scripts/with_compute_slot.py -- ../.venv-benchmark/bin/python reconstruction-studies/abstracts-1/experiment.py --mode validation --all-seeds
```

One CPU thread is enforced by the compute-slot wrapper. Validation budget was initially capped at 300 seconds total. Outputs include per-seed/condition AP and Brier, runtimes, fixture hashes, split hashes, error logs, and ablation results in `validation.json`.

## Phase two: parent-approved held-out evaluation

The final fixture was run once for all three seeds after the parent review gate. Held-out shift mean AP (equal average over the five non-clean final mechanisms) was **0.8186** for mixed-K8 distillation and **0.8631** for compute-matched 200-tree random-mask augmentation; the paired difference was **-0.0445**. Clean mean AP was 0.8857 versus 0.9111 (difference -0.0254). Final metrics, per-seed paired comparisons, runtimes, and hashes are in `results.json` and `verdict.json`.

The preregistered validation threshold had already failed: mixed-K8 mean shift AP was 0.7984 versus 0.8457 for the strongest control. The held-out results confirm the direction but are not used to revise that validation-stage decision. Verdict: `no_gain`; no novelty is established. `parent-approved-final.json` pins the phase-two code, fixture, validation, and preregistration hashes. The frozen fixture API changed after phase one so that final mode returns `test_x=None` and exposes the masks only through `final_cases`; current validation training-array hashes were checked against all three saved validation records before final mode was loaded.
