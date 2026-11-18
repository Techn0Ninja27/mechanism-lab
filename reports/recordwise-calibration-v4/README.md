# Direct, isolated calibration reviews

2026-11-19. Root launched 16 fresh native reviewers directly: two per frozen
packet, eight packets total. Each received only one packet and common source-only
instructions. The earlier coordinated attempt failed its isolation procedure
and is excluded; its files are preserved separately.

Both reviewers matched every field verdict in the independently checked answer
key. Four packets were narrowly supported and four contained plausible unsupported
or contradicted conditions. This is a small development calibration, not a
population accuracy estimate or proof of reviewer independence.

The complete calibration **did not pass**: 14 of 16 reviews passed mechanical
evidence validation. Both reviews of one packet malformed the same LaTeX-heavy
Sancus quotation, so its submitted spans are not exact source substrings. No
reviewer output or verdict was repaired. Semantic-label agreement and evidence
validity are reported separately in `direct-attempt-summary.json`.

Input, instruction, key and all 16 output hashes are recorded. The phase ledger
anchors the completed review set; `anchor-receipt.json` identifies its head.
This metadata operation changed no reservations or charges. Raw source excerpts
and review files remain in the local calibration directory.

The next instrument prototype uses stable field/evidence IDs, with source text
materialized by code, to avoid asking reviewers to recopy mathematical markup.
That prototype does not retroactively make this attempt pass. Production paid
execution remains blocked by the existing failed calibration.
