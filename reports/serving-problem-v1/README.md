# Disaggregated-serving source preparation

2026-11-17. **Draft evidence preparation, not a generated proposal.** No paid
API call has run. The latest offline plan is local
`results/serving-problem-v1/plan-offline-v3/` (protocol 7, $2/run, shared $20).

The primary source is [DMRS](https://arxiv.org/html/2610.15579v1), with
[Arrow](https://arxiv.org/html/2505.11916v2) as a practical scheduling baseline.
The current private pack contains 48,689 characters in 14 chunks. It includes
DMRS model/formulation, Algorithm 1 description, evaluation/related work,
limitations and the admission-control implementation; Arrow contributes its
design and evaluation sections. Full snapshots and excerpts stay local.
[Receipts](receipts.json) record the source hashes; the pack digest is
`ccc0244949507259e19b60c938f17d425d4b1e2de39906b387d4100859f84ea5`.

DMRS explicitly assumes known output lengths and instantaneous KV transfer in
its theory. Its GPU measurements include transfer overhead. The authors list
online and non-clairvoyant scheduling as future work. These are source-stated
scope limits, **not proof of an externally novel gap**. Its reported algorithms
also lose to colocated serving on some conversation workloads, and its appendix
reports higher P99 completion time despite lower P50. Both strengths and
counterevidence are retained in the pack. See DMRS §5.2, §7 and §H.2.

Before generation, add/read the directly adjacent Feng non-clairvoyant
scheduling and Chen uncertain-output-length work identified in DMRS related
work. A source's future-work paragraph alone is insufficient prior-art checking.
The MoE allocation paper is retained as a separate literature lead; its
expert/GPU assignment model is not forced into this admission-control problem.

Two preparation errors were caught before any paid call: the initial search
configuration had unsupported fields, and excerpt truncation discarded the
DMRS assumptions/limitations despite a claimed coverage check. The corrected
builder uses inspected line ranges and fails if source-pack limits truncate
them. Earlier `plan-offline` and `plan-offline-v2` are obsolete drafts and must
not be executed. Schema validity does not establish source coverage.

The public [builder](prepare.py) expects a `sources/` directory alongside it
containing the two HTML snapshots named in the source list. With those local
snapshots it rebuilds normalized excerpts, manifest and pack without network
access. The complete working copy is retained under
`results/serving-problem-v1/`. The [question](question.txt) and
[retrieval configuration](search.json) are planning inputs only.
