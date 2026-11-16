# Source-consistent coherence pilot

This report describes a small, source-lineage coherence pilot trained from
the original `papers/clusters.json` assignments. It is a language-model
co-occurrence check only: it is not a scientific-novelty result, temporal
forecast, or repaired full Alien ranker. Availability was not retrained or
combined with this run.

## Frozen run

- Seed 42; grouped canonical-set split: 3,152 train / 675 validation / 676 test.
- 30,580 original non-noise units from 13,419 mapped papers; 4,503 unique
  canonical sets; 273 source clusters; 274-token compact vocabulary.
- Two-layer, 128-wide causal transformer, dropout 0.1, max 400 steps;
  validation-selected step 175.
- Validation NLL: **4.0354012913**. Test NLL, evaluated once after selection:
  **3.9867861903**.

## Corrected baselines

`baseline-audit-conditional.json` evaluates the same saved validation/test
rows (validation encoded first, then test with the same RNG) and fits baselines
on train sets only:

| split | model | NLL |
|---|---|---:|
| validation | transformer | 4.0354012913 |
| validation | unigram | 4.3650371695 |
| validation | permutation-symmetric conditional pair | 4.7097392943 |
| test | transformer | 3.9867861903 |
| test | unigram | 4.3186238223 |
| test | permutation-symmetric conditional pair | 4.6624511099 |

The older `baseline-audit.json` used an invalid joint-pair comparison and is
not used here. The corrected audit's recorded manifest digest is
`79b5ff177710e340cce17fc3cb45c40dd9e47283c2fa9caf142a56ac2bcbc4b4`; retain
it as a historical receipt rather than rewriting it after provenance repair.

## Provenance limits

Source hashes: `clusters.json`
`99b157f1baedba7908fca0236fb7dca23b3a49da44054ada305035036aa27013`;
source-unit JSONL
`571590d60efe1ea57a0b0d9b9a12a16a2d794a99867f5712493a32309592ae79`.

The exact training source snapshot recorded by the original run was not
recoverable. `results/source-coherence-v1/training_script.py` is explicitly an
unverified reconstruction (hash `51be7b297585522400cd94a153096d75a627099eeb61586103b46dba21b5c7e6`),
while the immutable run manifest retains the originally recorded script hash
`4ada13257ecb42983f6b59ee711e6b2632aa1280ec6287d4a753a24cf9f0f1a2`.
During recovery the manifest field was temporarily edited while checking a
candidate reconstruction, then restored to this original recorded value; no
old receipt or model summary was rewritten.
The current trainer has changed since the frozen run. Therefore these numbers
are an auditable pilot record, not a claim of byte-for-byte reproducibility or
full-ranker repair.
