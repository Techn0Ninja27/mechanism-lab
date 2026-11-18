# Campaign 2 chunk-support review

Review one prepared representation draw against the complete text of the
source chunks it cites. The quote in `evidence` is a pointer, not the exclusive
evidence boundary. Every substantive field must be adjudicated: `statement`
(or `summary`), every input/output/condition item, and every unknown item. The
validator derives the record verdict from field verdicts; do not add a separate
record-level verdict or invent extra fields.

Use these fixed definitions. `supported` means the cited full chunk states the
claim or it follows logically without an outside assumption. `contradicted`
means the cited chunk states the opposite. `unsupported` means the claim is not
stated and does not follow from the cited text. `indeterminate` means a
malformed source span or unresolved referent prevents adjudication. Reviewer
hesitation is not indeterminate. An interface claim passes only when the source
explicitly identifies the role or gives a clear equivalent paraphrase; never
invent tensor formats, timing, shapes, communication semantics, or APIs.

For a supported positive claim, return an exact `support_quote` and its
`chunk_id`; the quote must be a substring of that cited chunk and the chunk
must belong to the record's source. For an unsupported, contradicted, or
indeterminate field, provide a specific `disputed_field` and reason. Unknowns
may be supported as explicit missing-information statements with an
`absence_reason` and no forced quote, but an unknown fails if the cited chunks
already answer it. Do not score novelty, usefulness, or executability.

Use criterion `cited_chunk_support.v1`, preserve one record per ID, and return
all field items. Two source-only reviewers are required for each draw; any
non-supported field in either review stops that draw. Reviewer identity is a
declared provenance field, not proof of independence. No reviewer sees the task
block, generated candidates, or experimental results.

The exact JSON shape is:

```json
{
  "schema": "chunk-grounded-draw-review.v1",
  "reviewer_id": "declared-reviewer-name",
  "input_sha256": {"source_pack": "source-pack-digest", "atoms": "canonical-sha256-of-validated-atoms"},
  "records": [{
    "record_id": "record-id-from-input",
    "criterion": "cited_chunk_support.v1",
    "field_items": [{
      "field": "statement", "value": "exact field value from input",
      "verdict": "supported", "reason": "short operational reason",
      "chunk_id": "source-chunk-id", "support_quote": "exact substring of that chunk"
    }]
  }]
}
```

For a distillation, use `input_sha256.distillations` and fields `summary` and
`unknowns[i]`. For atoms, cover `statement`, `inputs[i]`, `outputs[i]`,
`conditions[i]`, and `unknowns[i]`. Every field item requires `field`, `value`,
`verdict`, and `reason`; supported non-unknown items additionally require
`chunk_id` and `support_quote`; supported unknowns require `absence_reason` and
must omit quote fields; non-supported items require `disputed_field`. Do not
add record-level verdicts, unlisted metadata, or extra fields.
