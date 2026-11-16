# Broader operational-note extraction pilot

2026-11-17. Eight non-KV abstracts were selected from the existing ML-first
corpus across fine-tuning, sparse networks, adaptation, diffusion, training
systems, distributed graph embedding and numerical optimization. All eight
generated notes passed schema and exact-quote checks. They remain explicitly
**not ready for composition**, with training eligibility and novelty claims off.

The selection takes the first eligible unseen arXiv record from each query's
top 20, excludes the prior 24-paper packet, requires an abstract of at least
400 characters and publication from 2023 onward, and preserves the corpus's
2026-11-16 cutoff. This is a development packet, not representative sampling,
frontier coverage or a novelty evaluation. No new papers were downloaded.

## What was extracted

Each note separates operational inputs, produced/transformed objects, reported
outcomes, and missing contract details. Examples include graph embeddings,
distilled model weights, an uncertainty signal and a low-rank numerical solution.
Analytical constructs and practitioner guidelines are identified as such; their
presence in an output field does not turn them into executable interfaces.
The extractor permits empty input/output lists when the abstract is insufficient.

Two Haiku calls returned eight quote-valid records with no truncation. These
checks establish source-span identity and structure, not that every selected
span entails the associated description. Unknown shapes, phases, exposure,
parameters and assumptions remain unresolved. The notes are not imported into
the existing atom graph, vocabulary or learned scorers.

## Semantic review and its limits

The first native-agent review wrongly treated non-executable objects as invalid
outputs. A data object need not be executable code: embeddings, numerical
solutions and analytical representations can be legitimate outputs. The first
review is preserved, not used as training labels.

After clarifying that distinction, the same reviewer judged seven notes
`faithful_with_limits` and one `needs_revision`, with no reported benchmark
metric misclassified as a produced object. This second pass is **not an
independent evaluation**, and there is no controlled comparison against the old
extractor.

The flagged SQP note cites “step decomposition” for a search-direction output.
The full abstract does discuss components of the search direction, so this is
not an invented object. The selected excerpt is weak in isolation, and whether
the direction is exposed as an interface remains unknown. A full-method check
is needed before using it as a composable port. None of the eight notes is
promoted on the strength of this review.

`notes.json` publishes descriptions and quotation hashes; exact spans and
original abstracts remain in local artifacts. `semantic-review.json` and
`semantic-review-v2.json` preserve both passes. `selection.json` records sources,
queries and hashes. Deterministic packet selection reproduced byte-identically.

## Reproduce

The subsequent [full-method triage](full-method-triage.md) records explicit
graphon assumptions and an algebraically redundant low-rank composition.
Its [structured disposition](full-method-triage.json) is scoped to that mapping;
it is not a training label or evidence that all such combinations fail.

From the engine root, with the existing corpus and previous packet available:

```sh
../.venv/bin/python scripts/select_operational_packet.py \
  --corpus results/corpus-ml-grounded-v4/corpus.sqlite \
  --exclude results/ml-graph-atoms-v1/papers.json \
  --output results/operational-packet-fresh
../.venv/bin/python scripts/extract_operational_notes.py \
  --papers results/operational-packet-fresh/papers.json \
  --output results/operational-notes-fresh
```

The second command plans offline. Adding `--execute` makes at most two paid
calls under a $0.10 run cap and the existing $10 shared phase cap. It does not
retry previous attempts. Failed records are preserved as errors; valid records
are not promoted to semantic truth. Hosted-model outputs may differ on rerun.

This run used 6,599 input and 4,404 output tokens. Usage cost was $0.002833281;
the conservative shared ledger accounted for $0.0028619. Phase total is now
**$2.17069 / $10**. See `receipts.json`.
