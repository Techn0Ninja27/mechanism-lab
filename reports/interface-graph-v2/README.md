# Pair diversity does not establish mechanism compatibility

2026-11-16. A method-only, paper-diverse shortlist reduces repeated paper pairs
but did **not** improve abstract-level interface judgments in this exposed pilot.
It remains an explicit experimental option; the original lexical selector is
still the default. Neither selector establishes novelty or usefulness.

## Selection result

Both modes use the same 49 retained source-linked atoms. The original mode
has 424 eligible directed pairs, including empirical findings as producers.
The alternative has 182 method-to-method pairs. It greedily selects by the same
lexical scores, allowing each unordered paper pair once and each paper at most
three times. Its random comparison uses the same eligibility and constraints.

| Ranked shortlist | Candidates | Distinct paper pairs | Distinct papers | Maximum uses of one paper | Finding producers |
|---|---:|---:|---:|---:|---:|
| Original lexical | 20 | 15 | 16 | 7 | 9 |
| Mechanism-diverse | 18 | 18 | 15 | 3 | 0 |

The alternative does not increase every notion of diversity: it covers one fewer
paper. Greedy selection stops at 18 under these constraints; this is not a proof
that no larger feasible selection exists. The original ranked and random orders
and numerical scores were checked against the saved prior report and are unchanged.
New output fields record selection policy, kinds, inputs and code hashes.

## Blinded machine review

We took the first eight entries from each ranked shortlist and the first eight
from the constrained random shortlist. Deduplicating directed atom pairs gives
20 unique cases. They were shuffled with seed 84631; source abstracts and atom
descriptions were shown, while scores, source kind labels and shortlist origins
were hidden. Two separate native agents were instructed to review the same packet without
reading the key or each other's labels. Their exact model build/sampling settings were
not exposed; these are not independent human evaluations.

Both reviewers assigned **zero** cases to `direct_interface_supported`.
Their counts for `requires_specific_bridge` were:

| Exposed cohort, eight cases each | Reviewer A | Reviewer B |
|---|---:|---:|
| Original lexical top eight | 3 | 5 |
| Mechanism-diverse top eight | 1 | 3 |
| Constrained random eight | 1 | 1 |

They agreed on the categorical label for 15 of 20 unique cases. Agreement is not
ground truth. Cohorts overlap, papers repeat, judgments use only abstracts, and
the selection change was motivated by inspecting the original data. This is a
development diagnostic, not held-out precision, a statistical comparison or a
novelty benchmark. Requiring a bridge is not inherently bad: it identifies the
work a new hypothesis must specify. The problem is that source-pair diversity
alone did not supply those missing contracts.

Method labels also did not guarantee operational content. The reviewers called
some method-labelled random entries non-operational. Abstract atomization can
describe a package, result or intended benefit without specifying an artifact
that another algorithm actually consumes.

`candidate-index.json` makes case identities reviewable without republishing full
abstracts. `reviewer-a.json`, `reviewer-b.json`, `key.json`, `manifest.json` and
`blind-summary.json` preserve the labels, provenance and aggregation. The original
full abstract packet remains local at `results/interface-review-v1/packet.json`.
Labels are not eligible for automatic training or novelty promotion.

## Decision and next investigation

Do not replace the default ranker or declare an improvement in proposal quality.
Keep the constrained mode for controlled exploration. Before expanding this
atomization scheme, extract operational contracts from methods: produced object,
representation, consumer requirements, phase of availability and statistical
assumptions, with unknowns left explicit. Separate measured outcomes from these
interfaces. A full-method check is needed before treating a reviewer-suggested
bridge as implementable.

The one new non-cache pair both reviewers regarded as potentially bridgeable was
C006, a stochastic constrained optimizer feeding a federated gradient compressor.
The [bounded prior-art check](C006_PRIOR_ART.md) found established compressed
primal-dual methods and a substantial contract gap. Reshaping a solver direction
does not create convolutional or cross-round gradient statistics. We did not
promote this pairing into an experiment merely because the reviewers agreed.

## Reproduce

From the engine root, choose a fresh output file:

```sh
../.venv/bin/python scripts/build_interface_graph.py \
  --selection-mode mechanism_diverse --max-paper-uses 3 \
  --out /tmp/interface-diverse-fresh.json
```

`scripts/build_graph_review_packet.py` rebuilds the shuffled packet from saved
inputs; `scripts/summarize_graph_review.py` validates and aggregates archived
labels. Rerunning a model reviewer may produce different labels. The deterministic
selection and aggregation, rather than repeated judge output, are reproducible.
No paid generation was used for this pilot.
