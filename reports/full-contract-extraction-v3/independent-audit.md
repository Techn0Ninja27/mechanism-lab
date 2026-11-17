# Independent audit: full-method contract extraction v3

Audited 2026-11-17 after the extraction and review calls completed. This is a
source-fidelity and completeness audit, not an accuracy score or a claim that
any draft is executable. I compared the generated cards and review decisions
with the frozen excerpts in `results/contract-eval-packet-v1/` and with the
operation scopes fixed before the calls.

## Overall finding

The run is aligned with the preselected operations. The different sources and
scopes prevent an accuracy comparison with the earlier pilot. All three cards are source-linked and remain
`model_proposed_not_curated`; the review accepts only the narrowly scoped PERA
card and requests revision for GradLite and PacTrain. No card should enter the
curated registry or be composed into an executable molecule. The packet is
still partial: selected method excerpts are not complete papers, and the
contracts do not establish novelty, usefulness, or implementation fidelity.

## GradLite (`2510.22467v1`)

The draft follows the requested Algorithm 1 operation. Its learning rate,
parameters, accumulator, minibatch, projected approximate gradient, corrected
gradient, and returned state are supported by the selected method excerpt.
The unknowns correctly preserve missing construction details for the low-rank
basis, the compressed activation representation, accumulator initialization,
and the approximation-error estimate.

The reviewer correctly identifies hidden prerequisites: `U_t`, `V_t`, and the
top-level error signal `delta_t` are used by the projection but are not exposed
as input ports. The draft also does not expose the residual-estimation input or
state needed to form `Delta_t`. The source is internally ambiguous here: the
displayed update equation uses the exact gradient `g_t`, while Algorithm 1
describes estimating the approximation error. The review should therefore be
read as identifying an unresolved boundary, not as proving that an exact
gradient must be available as an external runtime input. A revised card should
expose either an exact-gradient prerequisite or a named residual-estimator
interface, with the choice left unknown when the excerpt cannot decide.

The review mentions `delta_t` in its reason but does not make it a separate
issue with dedicated evidence. That omission matters for composition: the
card remains incomplete even though the reviewer decision is correctly
`revise`. The source excerpt also does not establish how `U_t,V_t` are formed
or adapted, so the card cannot support an executable projection operator.

## PacTrain (`2505.18563v1`)

The draft captures the requested stable-versus-unstable communication branch,
including GSE, mask tracking, compression in the stable case, and full
AllReduce in the unstable case. The cited algorithm lines are appropriate.
The unknowns appropriately retain the missing stability criterion, compression
format, aggregation semantics, and model-update application.

The reviewer is right that `dense_tensor` is not justified for the unstable
AllReduce output by the cited algorithm line. The selected prose associates
dense reformatting with the sparse communication path; the unstable branch is
only specified as AllReduce/full synchronization. That representation should
remain null or unknown.

The review's claim that `sparse_mask_state` must be intermediate is too strong.
The algorithm updates a mask using a tracker before testing stability, but the
tracker may consume prior mask/tracker state supplied by the surrounding
training loop. The excerpt does not settle the interface boundary. The safer
revision is to retain a prior mask/tracker state as a required or optional
state input, mark its update as intermediate state, and keep the initial-state
and tracker API unknown.

The card also does not expose pruning configuration or the pre-trained/pruned
model as explicit prerequisites; its `model_pruned` condition records only a
condition token. That is acceptable for a draft, but insufficient for a
composition-ready operator. The stable path's relationship between sparse
nonzero communication and any dense AllReduce representation remains an
explicit source ambiguity and must not be filled in from the paper's headline
claims.

## PERA (`2604.11841v1`)

The accepted draft is faithful to the selected operation. It preserves the
important distinction between standard `Poly^2(B)` and coefficient-weighted
Hadamard `Poly_H^2(A)`, includes the stated B/A/h initializations, carries the
expanded dimensions, and produces an `m`-by-`n` `Delta W` for a linear layer.
The review's `accept` is appropriately limited to this source-supported card;
it does not imply a curated or executable contract.

The remaining unknown exposure for `Delta W` is reasonable because the excerpt
defines the mathematical update and the forward integration separately but
does not specify an implementation API or whether the update is materialized.
The card does not include the base weight/input forward path (`W_0`, `x`) or a
LoRA scaling factor; those are outside the narrowly fixed Delta-W operation or
remain unspecified in the excerpt. A future composition should not infer them.
The selected text also contains a source-level rank/feature discussion and
limitations, but the card correctly does not turn those claims into ports or
evidence of usefulness.

## Disposition

The run demonstrates fixed operation targeting and catches several hidden
interfaces, but it does not establish full-method recovery. Preserve all three
drafts as audit artifacts. Keep GradLite and PacTrain unreconciled until their
state and residual/communication boundaries are explicitly curated. PERA may
be a candidate for human curation of its limited mathematical update card,
but it is not yet an executable operator or a validated research molecule.
