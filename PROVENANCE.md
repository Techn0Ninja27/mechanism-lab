# Atom provenance

Cluster IDs are local integers, not global identifiers. A candidate generated
by a checkpoint is valid only if its IDs occur in the checkpoint's own token
mapping and in the exact cluster assignment file used to build that mapping.
`generation.provenance.audit_provenance` checks this, and can compare a second
semantic catalogue by `(paper_id, text)` (excluding noise assignments) to detect
relabelled releases. Set `require_safe=True` to fail closed before decoding.
`compare_paper_ids` separately checks whether a scorer JSONL contains the same
paper population; matching paper sets do not prove matching atom IDs.

The released `clusters_80.json` and `clusters.json` should not be joined by
numeric ID without this audit. Canonical names may have been copied onto a
different label ordering. A mismatch warning means descriptions and supporting
papers must not be treated as evidence for scorer IDs; use the exact scorer
assignment file or rebuild the mapping from the original clustering run.

## Local release audit (2026-11-16)

All 15,203 released coherence-dataset paper atom sets exactly match the expanded
`clusters_80.json` assignments; none exactly matches `clusters.json`. Thus the
scorer corpus is identifiable, but that does not validate its canonical names.
Across 24,541 shared non-noise units, exact numeric-label agreement between the
two cluster artifacts is only 0.0088016. A majority remap is also unsafe: nearly
all expanded clusters mix many original labels. The causal source of this
artifact inconsistency is not established by this audit.

The corrected proposal baseline uses original `clusters.json` descriptions and
source units as unranked concepts, strips scorer scores from decoder inputs,
and makes no claim that this semantic interpretation inherits scorer ranking.
Repairing the scoring experiment itself requires reconstructing a coherent
assignment/description corpus and training or validating checkpoints against it;
no repaired scorer is claimed here.
