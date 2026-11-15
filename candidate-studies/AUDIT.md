# Candidate-study audit — 2026-11-16

All 21 assigned studies now have verdict artifacts and pass the repository's
fresh artifact-integrity check. This audit does not rerun experiments or
establish novelty. It separates the saved status label from the strength of
the causal evidence.

## Cohort summary

- **Clean negative / no-gain (7):** digits/direct-0, direct-2, random-0,
  random-1, random-2, random-3, and tabular/world-0.
- **Provisional or methodologically compromised (9):** digits/direct-1,
  direct-3, world-0, world-1, world-3; tabular/direct-0, direct-2, random-1,
  world-1, world-2, world-3. (The list contains 10; `tabular/world-2` is
  included here because its corrected run was exploratory after an earlier
  held-out result had been observed.)
- **Invalid/non-identifiable mechanism (3):** digits/world-2,
  tabular/direct-1, tabular/random-2.
- **Inconclusive (2):** tabular/direct-0 and tabular/world-1.

The overlapping wording above is intentional: `supported_pilot` is not a
clean-evidence category. There are no claims of a new atom or scientific
novelty anywhere in the cohort.

## Why the positive labels are not admissions

`digits_shift-world-0` reports +0.1045 accuracy, but positive variance weights
are canceled by the following `StandardScaler`; the candidate equals the
mask-only ablation. The observed gain is masking, not the claimed weighting.

`digits_shift-direct-1` reports a large gain, but augmentation-only explains
most of it. Blur alone loses on severe dropout, so the claimed blur relation is
not isolated.

`digits_shift-direct-3` has a small positive result, but its held-out fit used
train only although the proposal specified train-plus-validation refitting.

`digits_shift-world-1` has a `supported_pilot` label, but its composite score
made the same choices as validation-only scoring; it is still a tiny digits
pilot with no external transfer.

`digits_shift-world-3` compares against a weak control. Its validation-selected
single model beats the ensemble; block augmentation is coupled to PCA, and
numerical warnings were emitted. The margin weights are nearly uniform.

`tabular_missing-random_seed-1` reports a large AP gain, but explicit mask
columns are redundant with native HGB routing: the no-mask ablation matches
and mask permutation changes AP by zero. This is augmentation performance,
not evidence for a mask molecule.

`tabular_missing-world-3` reports a gain, but its preregistration was recorded
after final evaluation, its control is weak, and MI weighting did not beat
uniform weights. It is exploratory.

`tabular_missing-direct-2` is effectively null (+0.00022 AP mean); its saved
`supported_pilot` label should be read as a small descriptive result, not a
meaningful improvement.

`tabular_missing-world-2` is a no-gain result, but the corrected final was
exploratory after a superseded held-out result had been seen. Keep it out of
the clean cohort.

## Invalid or non-identifiable mechanisms

- `digits_shift-world-2`: weighting followed by standardization produces exact
  control predictions.
- `tabular_missing-direct-1`: clean all-zero training masks produce one
  k-means prototype, so the claimed eight-way routing does not exist.
- `tabular_missing-random_seed-2`: the MCAR fixture has no stable-anchor/
  volatile-feature structure, so no Ridge reconstruction was performed.

## Artifact integrity

The repository-level [artifact-integrity.json](artifact-integrity.json) checks
all 21 source hashes using each study's declared `source_sha256`,
`source_code_sha256`, or multi-file hash map. All checks pass. Earlier warnings
about missing hashes or mismatches were based on reading only one possible
field and are superseded by this complete check.

The seven clean negatives are the strongest current conclusion: generated
proposals often reduce to known techniques, malformed mechanisms, or gains
explained by simpler components. A molecule should be promoted only when the
claimed relation beats the strongest component-matched control, survives
independent data and seeds, and receives human prior-art adjudication.
