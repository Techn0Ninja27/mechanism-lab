# Independent audit of the accepted DMRS draft

Reviewed 2026-11-17 against `results/full-contract-extraction-v1/pack.json` and
the exact draft in `results/full-contract-extraction-v2/run/result.json`.
This is an audit of a model-proposed contract, not a novelty or executable-
compatibility determination.

## What the draft gets right

The draft identifies the correct high-level DMRS operation: the paper maps a
batch of request tuples and two pool capacities to prefill and decode admission
times. Its evidence IDs are valid against the cached source chunks. The
`request_characteristics` input is appropriately scoped to an offline batch,
with characteristics known in advance, release at time zero, and individual
prefill/peak-decode feasibility. The draft also keeps the LP solution and
dyadic groups as intermediate outputs rather than pretending they are returned
serving artifacts.

The Opus review's acceptance is therefore reasonable as a *limited draft*:
the cited text supports the existence and broad inputs/outputs of Algorithm 1.
It does not mean that the full algorithm is executable from this excerpt.

## Missing or over-broad fields

1. The request tuple should be decomposed into named inputs (`m_i`, `p_i`,
   `d_i`) or a typed tuple contract. The draft does not say that `m_i` is a
   measured prefill duration, `p_i` is prompt-token memory, and `d_i` is the
   exact output-token count. Those distinctions matter when comparing DMRS to
   the non-clairvoyant or interval-bound methods.
2. Non-preemption within each phase is a model condition, not just an unknown
   implementation detail. It should be recorded explicitly, together with the
   slot convention (one decode iteration per slot) and the relation
   `h_i = p_i + d_i`.
3. The `alpha_lt_1` and `beta_lt_1` conditions need their definitions:
   `alpha=max(p_i/M_P)` and `beta=max(h_i/M_D)`. Without these, the port is a
   label rather than a checkable condition. The draft does not expose `h_i` as
   a derived artifact.
4. `admission_times` is potentially too broad. The selected excerpt contains
   the algorithm heading and output label, but the remaining algorithm lines
   are truncated. It supports a proposed schedule output at a descriptive
   level; it does not support an implementation-ready schedule contract.
5. The draft's unknowns omit two material source-stated boundaries: theory
   assumes instantaneous KV transfer, while the GPU experiment includes
   transfer overhead; and GPU enforcement uses an external proxy/software
   accounting. These should be unknown or limitation fields before any
   composition with a runtime operator.

## Composition disposition

Keep this card as `model_proposed_not_curated`, with the Opus decision
`accept`, but mark it `draft_supported_but_not_composition_ready`. It can be
used as a source-linked description of an offline scheduling interface. It
must not be connected directly to an online/non-clairvoyant contract or to a
hardware transfer operator: the information regime, phase assumptions and
transfer model differ. The existing contract bridge correctly creates no
molecule for Chen's certified lower bound feeding DMRS's exact output-length
input.

The next integration should add the missing derived fields and source-backed
limitations through a revision review. It should not silently promote this
accepted draft, infer `h_i`, or treat the paper's approximation result as a
runtime guarantee. A later molecule may represent a proposed adapter only
after the adapter's inputs, state transitions, transfer assumptions and
falsification test are explicitly contracted and independently reviewed.
