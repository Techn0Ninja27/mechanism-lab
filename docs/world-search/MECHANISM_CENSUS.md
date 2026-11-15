# Exploratory mechanism census — 2026-11-16

This is an unblinded, exploratory coding pass over the completed records in
`results/world-search-pilot/candidates/`. It was written before looking at the
candidate benchmark results. It is not an independent human novelty study and
it does not establish that any proposal is new, useful, or better than a
baseline. The purpose is to identify repeated mechanism families and to choose
the smallest molecule-level ablations.

At the time of the first coding pass there were 15 complete records: ten for
the digit shift task and five for missing tabular features. The pilot later
reached 21 complete records and three failed calls out of 24. The additional
records add another corruption-variance weighting variant and more missingness
mask, soft-imputation, auxiliary-reconstruction, and curriculum variants; they
do not change the mechanism families below. Exact wording and arm labels were
visible during coding.

## Digit shift task

The direct arm mostly proposes robust feature transformations. Its records
include gradient and block-mean features, Gaussian pre-smoothing, and a
corruption-aware scaling denominator. These are three different implementations
of a common causal idea: estimate or suppress the part of the input variation
that is caused by corruption before the RBF kernel sees it. The smoothing and
corruption-envelope proposals are near neighbors even though one filters in
space and the other changes the normalization geometry. The gradient/block
proposal is a related representation change, with a different prediction about
which spatial statistics survive corruption.

The random-seed arm contains two augmentation-family proposals and two
feature-family proposals. Severity-swept augmentation and arc-sampled joint
corruption augmentation both change the training distribution rather than the
classifier. The former changes the density over severity; the latter changes
the path through the joint noise/dropout space while holding a corruption
energy budget approximately fixed. Fisher-porosity augmentation and the
horizontal-flip symmetry feature instead add a data-dependent feature or
sampling rule. They are not automatically new primitives: Fisher weighting is
feature selection under a new corruption objective, and a flip score is a
single engineered invariance statistic.

The world arm contains PCA or preprocessing changes, structured block dropout,
and validation-based selection/ensembling. The strongest world-specific
translation is not the fictional law itself. It is the translation from an
in-world causal rule to a measurable design distinction: contiguous occlusion
versus independent dropout, or corruption variance versus ordinary signal
variance. Several world proposals then add familiar optimization machinery
(PCA, centroid anchoring, or model selection). That makes the world arm useful
as a source of hypotheses, but it also makes it easy to mistake extra machinery
for a new mechanism.

The most promising candidate molecule for a controlled test is:

`corruption model -> representation/augmentation operator -> fixed classifier`

The relevant edge types are different in the records: a corruption model may
set a feature metric (envelope scaling), alter the representation (smoothing,
gradients, PCA), or generate training examples (severity/shape augmentation).
The smallest informative ablation holds the classifier, split, corruption
draws, and training count fixed while swapping only the operator. A second
ablation removes the corruption model and uses ordinary augmentation. If the
effect disappears when the operator is detached from its stated corruption
model, that is evidence against the proposed causal molecule.

## Missing-tabular-feature task

The direct records split into two families. Missingness-pattern prototype
routing and confidence shrinkage use a post-hoc reliability signal: the model
first predicts, then a pattern-specific calibration or shrinkage stage changes
the output. The reconstruction proposal uses observed anchor variables to
replace missing values. These are different interfaces and should not be
collapsed into one atom.

The random-seed proposal appends explicit mask columns and augments training
with random missingness. The world records repeat that explicit-mask idea,
then add either a soft imputation rule, an auxiliary reconstruction objective,
or a missingness curriculum. Those are plausible molecules, but explicit mask
features plus train-time dropout is already a standard missing-data pattern in
many model families. The candidate distinction would need to be the interaction
with a specific deployment shift and a pre-registered calibration or robustness
prediction, not the name “mirrored” or “asymmetric.”

The smallest informative molecule here is:

`missingness observation -> value treatment -> prediction/calibration`

Possible edge types are `mask informs imputation`, `mask informs predictor`,
and `mask informs confidence correction`. Compare these edges against the same
base predictor with (a) imputation only, (b) mask only, and (c) joint mask plus
imputation. A proposal should not be promoted merely because the joint system
has more parts.

## Candidate status coding

The records currently support these provisional categories:

| Category | Examples | What would be needed next |
|---|---|---|
| Known technique or close variant | Gaussian smoothing, explicit missingness indicators, augmentation schedules, PCA preprocessing | Exact prior-art search and a matched baseline |
| Structured combination | corruption model + representation operator; mask + imputation + calibration | Component-removal ablations and a new task |
| Candidate new relation | corruption severity used to choose a representation geometry; missingness mask used to route confidence correction | A counterfactual prediction that the unordered components do not make |
| Candidate primitive | None can be admitted from these records alone | Recurrence, compression/reuse, independent held-out transfer, prior-art review, and executable evidence |

The world-generated records do not yet show a new atom. They do show a useful
search operator: they produce alternative causal decompositions that make
specific ablations easy to state. A future molecule ledger should retain both
the successful proposal and its rejected translations, because the rejected
translation identifies which fictional assumption failed during bridging.

## What this census changes

The next evaluation should not compare whole prose ideas by semantic distance.
It should compare typed molecules with the same components under different edge
structures. For each molecule, record the predicted direction of each edge,
the smallest component-removal test, and whether the relation recurs on the
other task. Only after that should a candidate be considered for promotion to a
macro-atom. Any prior-art review remains bounded and automated until a human
adjudicates it.
