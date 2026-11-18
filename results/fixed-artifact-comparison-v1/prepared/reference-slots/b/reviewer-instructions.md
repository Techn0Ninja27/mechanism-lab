Return only {"fields":[...]} with one item per field_id.
Each item contains field_id, verdict (supported, unsupported, contradicted or
indeterminate), an item-specific reason, and evidence_ids. Never copy field
values, quotes, chunk IDs or hashes; the orchestrator resolves those IDs.
Judge scientific fields only: metadata is bookkeeping. Supported means the
cited rendered chunks state the claim or it follows without an outside
assumption. Contradicted means the cited chunks state the opposite.
Unsupported means the claim could be false while the cited chunks remained
true. Indeterminate is only for malformed evidence or unresolved referents.
Unknown fields mean information absent from the supplied cited chunks; if the
chunks answer the unknown, it is unsupported. For supported non-unknown fields,
provide one or more exact evidence_ids. For supported unknowns provide an
absence_reason and no evidence IDs. Use at most four evidence IDs per field
and per subclaim. Compound fields must include claim_assessments, each with its
own verdict, reason and evidence_ids and a field cannot pass if one subclaim
fails. For unsupported claims name the missing assumption; for contradictions
name the conflicting passage or provide a concrete counterexample consistent
with the cited text. Do not calculate provenance hashes.

Version 2 permits one optional bookkeeping key, subclaim_id, inside each
claim_assessments object. It must be a nonempty string of at most 100
characters and has no semantic meaning. No other additional keys are allowed.

