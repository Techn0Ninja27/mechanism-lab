# First full-method extraction attempt

2026-11-17. Haiku produced two proposed cards from the DMRS/A_min method
excerpts. The batch failed exact quote validation: the DMRS admission-times
quote was present in chunk0003 but cited chunk0002. No semantic review call was
made, and no draft was accepted, curated or composed.

The raw response also used the string `unknown` as a phase value. A subsequent
validator change rejects such placeholder strings in nullable typed fields;
JSON null is required. The original failed result remains unchanged.

Cost: **$0.0026791**. [Result](result.json) and [receipt](call-receipts.json)
retain the failure and accounting. The original raw model response and source
code snapshot remain local under `results/full-contract-extraction-v1/`.
The [separate Opus pilot](../full-contract-extraction-v2/README.md) changed the
extractor, instructions and validation, so it is not a controlled model comparison.
