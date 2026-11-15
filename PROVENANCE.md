# Atom provenance

Cluster IDs are local integers, not global identifiers. A candidate generated
by a checkpoint is valid only if its IDs occur in the checkpoint's own token
mapping and in the exact cluster assignment file used to build that mapping.
`generation.provenance.audit_provenance` checks this, and can compare a second
semantic catalogue by `(paper_id, text)` to detect relabelled releases.

The released `clusters_80.json` and `clusters.json` should not be joined by
numeric ID without this audit. Canonical names may have been copied onto a
different label ordering. A mismatch warning means descriptions and supporting
papers must not be treated as evidence for scorer IDs; use the exact scorer
assignment file or rebuild the mapping from the original clustering run.
