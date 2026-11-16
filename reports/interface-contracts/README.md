# Source-linked operational contract pilot

2026-11-16. Two manually curated method substeps now expose their required
inputs, output artifacts, representation, granularity and availability. This
addresses a concrete omission in the abstract atoms; it is not an automatic
extractor, executable type system, formal proof or improved novelty result.

The cards in `docs/method-grounding/interface-contracts.json` distinguish:

| Substep | Required inputs | Produced values |
|---|---|---|
| FIER importance estimation | Query, binary key codes, groupwise scale/offset metadata | Per-token scores and selected indices, both intermediate values |
| WindowQuant partitioning | Visual KV windows, window similarity scores, per-layer thresholds | KV blocks grouped by target bit width, an intermediate value |

FIER's source algorithm groups keys by a parameter `g` along each channel.
The card requires the group size and channel layout to be known; it does not
hardcode `g=32`, which is a reported setting rather than a universal interface.
WindowQuant's partitioning phase is labelled
`after_prefill_attention_before_decode`: a curated interpretation of its
instruction to compute the prefill output before reordering, still within the
prefill stage. A separate source span supports per-layer threshold dependence.
FIER's query availability phase remains unknown in the card.

## What the checks find

All three selected connections need explicit bridging:

- Token importance → window similarity: different artifacts, representations
  and granularities. Aggregation and a scoring-rule change must be stated.
- Token importance → KV cache: scores do not contain the key/value tensors.
- Selected indices → KV cache: indices need a gather/materialization step and
  the original cache, with window membership and position handling preserved.

The consumer still needs its other required inputs. Exposing an intermediate
value may require an adapter even when its semantic type matches; this is not
itself a claim that the algorithms are incompatible. Unknown fields are recorded
as unresolved, never accepted as compatible wildcards. Required conditions not
declared by the producer remain unestablished. Metrics can feed explicitly
metric-valued consumers; the checker does not ban all measured values.

The cards cite exact source-pack chunks and hashes. Validation checks quote
identity, source attribution, schema, linked evidence and bounds. It does **not**
prove that a quote entails every curated field. The port vocabulary and condition
labels are interpretations, not automatically inferred mathematical properties.
Even an exact declared match returns `declared_match_unverified`, with
`executable_compatibility_proven: false`.

## Integration and reproduction

Grounded proposal protocol 4 includes matching source cards inside its untrusted
evidence context for both proposal and review. It asks the model to account for
all inputs and identify necessary adapters. Card and reader hashes are frozen
in the plan, so changed annotations cannot silently alter a planned paid run.
No paid call was used to validate this integration, and its effect on proposal
quality remains unmeasured. Original generated proposals and experiments remain
unchanged.

Run the checker from the engine root with a fresh output path:

```sh
../.venv/bin/python scripts/check_interface_contracts.py \
  --pack results/interface-proposal-v2/pack.json \
  --connection fier_importance_substep:importance:windowquant_partition_substep:similarities \
  --connection fier_importance_substep:importance:windowquant_partition_substep:cache \
  --connection fier_importance_substep:selected_indices:windowquant_partition_substep:cache \
  --output /tmp/source-contract-assessment-fresh.json
```

The input pack contains the previously downloaded FIER paper, WindowQuant
method pages and pinned FIER code. No new source or model downloads are needed.
`assessment.json` records input and code hashes. Tests cover unknown fields,
unestablished conditions, intermediate exposure, additional inputs, metric roles,
bad source quotes, malformed records, duplicate IDs and partial source coverage.
