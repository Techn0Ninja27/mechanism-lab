# Problem-linked operational notes from the expanded corpus

2026-11-17. The extractor now records source-stated problems alongside operations,
inputs, outputs and reported outcomes. This addresses a representation gap:
knowing what a mechanism produces does not say which problem motivated it.
**A paper's stated motivation is not an unsolved research opportunity.** It may
already be addressed by that same paper or later work.

## Run and review

Eight previously unused sources were selected from the 739,964-record corpus
with the existing eight query slots and deterministic first-eligible rule. The
24-paper and earlier eight-paper packets were excluded. These sources span
2023–2025; they are not a current-frontier ranking or representative sample.
The packet includes PEFT design spaces, structured sparse training, VLM adaptation,
diffusion distillation, distributed checkpoint planning, GNN sampling, stochastic
SQP and randomized approximation analysis.

Two bounded Haiku calls produced **seven accepted notes and one rejected note**.
All accepted quotation spans match their source abstracts exactly. DDIL was
rejected because an input quote lowercased an initial capital letter; its raw
response and error are preserved locally. There was no retry, silent repair or
promotion of the rejected record. Both calls ended normally, without truncation.

Two separate native agents reviewed complementary fields against the abstracts:

- The operation/input/output review judged all seven accepted records
  `faithful_with_limits`. This does not make any interface executable.
- The problem review passed four and flagged three for weak selected evidence
  or scope: VLM adaptation overhead, checkpoint planning cost, and stochastic
  oracle inexactness. For the last, a modeling condition should not be presented
  as an unsolved gap. The full abstracts support parts of the flagged claims;
  the flags do not establish that those papers lack meaningful motivations.

These are same-family model reviews, not human adjudication or a measured quality
improvement over the earlier extractor. Root inspection confirmed the case-only
DDIL mismatch and the flagged checkpoint/SQP scope issues. Raw notes and reviews
remain unchanged. No record enters the atom graph, scorer training, or automatic
composition queue.

## Software and app behavior

Extraction protocol 2 adds `stated_problems`: up to four name/quote objects, or
an empty list when no explicit problem is stated. It requires exact source spans
and keeps `ready_for_composition`, `train_eligible`, and `novelty_established`
false. Prior protocol-1 artifacts remain historical records; they are not silently
migrated or revalidated as protocol 2. New runs need fresh output directories.

The workbench's Operator library now has a separate **Abstract-derived mechanism
notes** section. It displays the seven notes, problem-scope flags, missing details
and source links. These notes are not registered executable operators or typed
chains, and cannot be passed to the local operator executor as catalog entries.
The API bounds note size/count and excludes records with promotion flags enabled.

`notes.json` contains descriptions and quote hashes; exact source excerpts remain
in ignored local artifacts. `selection.json` records the fixed queries, exclusions,
source titles and hashes. `problem-review.json`, `operation-review.json` and
`receipts.json` preserve judgments and usage.

## Reproduction and cost

The local exclusion packet is `results/operational-notes-v2/exclude.json`, formed
by concatenating the prior two paper packets. From the engine root:

```sh
../.venv/bin/python scripts/select_operational_packet.py \
  --corpus results/corpus-ml-complete-mirror-v6/corpus.sqlite \
  --exclude results/operational-notes-v2/exclude.json \
  --output results/operational-packet-fresh
../.venv/bin/python scripts/extract_operational_notes.py \
  --papers results/operational-packet-fresh/papers.json \
  --output results/operational-notes-fresh
```

The second command plans offline. `--execute` makes at most two paid calls under
the existing $0.10 per-run and $10 shared caps. No automatic retries. Hosted
outputs can differ. Protocol, input and dependency hashes are frozen locally in
`results/operational-notes-v2/extraction/plan.json`.

Usage: 6,491 input and 4,582 output tokens. Conservative cost **$0.0029401**;
phase total **$2.5153296 / $10**. No new paper or model downloads were needed.

Deterministic packet selection reproduced byte-identically. Four extractor tests
and 37 webapp tests passed; these validate software behavior, not research quality.
