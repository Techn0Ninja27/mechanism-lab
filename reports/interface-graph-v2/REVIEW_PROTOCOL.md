# Blinded abstract-level interface review

2026-11-16. Each native agent received the same instructions and packet path,
with only its output path differing. Both started without the parent transcript.
They were told not to inspect the key, graph shortlists, other reviews, history
or the web. Paper identities and source abstracts were visible; shortlist origin,
scores and producer-kind metadata were not.

Task: assess whether the producer's stated output can feed the consumer's
stated input. Do not assess novelty or scientific usefulness. Classify each
case exactly once:

- `direct_interface_supported`: same operational artifact, with enough contract
  supplied by the evidence.
- `requires_specific_bridge`: a concrete bounded transformation could connect
  the methods, but it is not verified by the supplied evidence.
- `unsupported_connection`: generic word overlap or incompatible objects,
  without an identifiable bounded bridge.
- `non_operational_producer`: a metric/observation rather than a reusable
  operation is being offered as the producer.
- `insufficient_information`: the abstracts do not support choosing a category.

Reviewers were told not to reject a bridge merely because it was unproven, not
to invent an unbounded new method to rescue an arbitrary pair, and not to assume
the extracted atom was faithful if its abstract contradicted it. Each decision
required a paraphrased reason of at most 400 characters, a missing-contract or
bridge explanation of at most 500 characters, and low/medium/high confidence.
No quotations or paid calls were requested.

The review files bind the exact packet SHA-256. The summarizer verifies the
packet/key hashes, ID coverage, labels and explanation bounds before joining
labels to cohorts. Individual model build/sampling parameters are not recorded
by the native-agent interface; identical judgments on rerun are not promised.
Same-family model agreement is not ground truth or independent human review.
