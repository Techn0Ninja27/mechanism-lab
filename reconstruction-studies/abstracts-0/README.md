# abstracts-0: ADMIB reconstruction

## Status and scope

This is a preregistered validation study of a proposed missingness-shift method, not a novelty claim. It uses the shared synthetic fixture (5,000 rows, 20 columns, columns 0 and 1 always observed; three fixed seeds). Training inputs are already MCAR-masked. The learner never receives clean training values, validation labels during prediction, condition names, nor final-mode arrays in phase one.

## Mechanism and feasibility

The proposal combines (1) iterative conditional completion learned only from observed entries, (2) augmentation with additional MCAR, anchor-MAR, and random block masks, and (3) inverse propensity weighting intended to reduce dependence on mask regime. The implementation learns an IterativeImputer from masked training inputs only. It makes three additional masked copies per input (so all compared augmentation arms receive four copies total), completes them, and fits a histogram gradient booster. A four-class multinomial logistic model estimates augmentation-regime propensity from completed features; clipped inverse class propensity is normalized per class and supplies sample weights.

Important identifiability caveat: in the fixture, labels depend on the full latent feature vector, while the artificial regime is assigned independently of labels conditional on observed data. Since augmented copies deliberately discard information and the imputations are uncertain, balancing regime propensity cannot generally enforce the asserted conditional independence of mask and label given completed features. The method may lose information and is not guaranteed to improve the reversed self-masking MNAR mechanism. This study therefore tests the concrete recipe, not its stated causal guarantee.

## Prior art checked (primary sources)

- [XGBoost, Tree boosting with missing value branch learning](https://arxiv.org/abs/1603.02754): tree learners can route missing values through learned default branches, making missingness informative.
- [Le Morvan et al., What's a good imputation to predict with missing values?](https://arxiv.org/abs/2106.00311): impute-then-regress can be Bayes optimal for broad imputation functions, while joint NeuMiss is proposed as a finite-sample alternative. This makes impute-then-regress an established constituent, not a novelty.
- [Rockenschaub et al., Robust prediction under missingness shifts](https://arxiv.org/abs/2406.16484): distinguishes ignorable shifts, under which the Bayes predictor is unchanged, from non-ignorable shifts, which can change it; empirical robustness depends on shift type.
- [Lee et al., MIRRAMS](https://arxiv.org/abs/2507.08280): explicitly targets unseen missingness shifts using mutual-information robustness conditions. This is close prior work on the objective.
- [Hasny et al., No Data? No Problem](https://arxiv.org/abs/2512.19602): uses missingness augmentation for robustness in a vision-tabular task. It is not a direct tabular-only comparison but supports augmentation as existing practice.

Novelty remains unestablished; further literature review would be required for a publication claim.

## Arms and ablations

Matched controls use the same train split and four-copy row budget: native-NaN HGB; native-NaN HGB with MCAR-only augmentation; completed HGB without augmentation; completed HGB with augmentation but no weights (strongest constituent control); completed HGB with MCAR-only augmentation and weights. Ablations also include mean completion, and completion plus explicit missingness indicators. HGB capacity is fixed across booster arms. Iterative imputation uses Bayesian ridge and a fixed iteration cap. CPU threads are limited to one.

The proposal's MLP NeuMiss control is not reproduced: a faithful NeuMiss architecture would require model-specific implementation and substantially exceed the phase-one compute gate. Its primary paper is cited above; results here must not be described as beating NeuMiss.

## Frozen validation rule

Primary metric: mean average precision (AP) averaged over all fixture validation conditions (clean, mcar_0.2, mcar_0.4, mar_anchor_x0) and all three seeds. Secondary metrics: Brier score, clean AP, worst-condition AP, and per-condition runtime. Candidate selection is fixed to the preregistered recipe; no hyperparameter search uses final data. The predeclared practical threshold is candidate mean AP at least 0.005 above the strongest matched control, with no clean AP decrease greater than 0.01. Report validation across clean, MCAR, and MAR; the fixture's final-only self-MNAR and structural masks remain held out.

## Run

`../.venv-benchmark/bin/python scripts/with_compute_slot.py -- ../.venv-benchmark/bin/python reconstruction-studies/abstracts-0/experiment.py --mode validation`

## Phase two result (single frozen run)

The parent-approved final run covered all three seeds and six held-out conditions (144 arm-condition records, 23.5 seconds, no errors). ADMIB mean AP was 0.90572. The strongest named constituent control, completed input with the same three augmentation copies and no propensity weights, reached 0.90674; paired seed-level ADMIB deltas were -0.00264, -0.00179, and +0.00138, for a mean delta of -0.00102. ADMIB therefore did not reach the preregistered +0.005 utility threshold. Its mean Brier was 0.05346 versus 0.05353 for that control, and clean AP was effectively tied (0.91756 vs 0.91754).

Against mean-completed augmentation, ADMIB gained 0.04499 AP, but that weak control is not evidence for the proposed weighting intervention. Against completed input without augmentation it gained 0.00568 AP on final data; that does not erase the failed validation screen or the lack of gain against the matched augmented control. Verdict: no gain for ADMIB's added propensity weighting in this small synthetic study. Novelty is not established; close prior art is documented above. See `results.json` and `verdict.json` for full per-condition metrics and hashes.
