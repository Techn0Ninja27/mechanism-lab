# Dataset roadmap for evidence-grounded idea evaluation

Updated 2026-11-16. This roadmap is for the proposed Alien Science loop: generate atom combinations, retrieve prior work, and have a cheap reviewer (for example Haiku) produce a structured novelty/evidence assessment. The reviewer is a measurement component, not ground truth. In particular, we should not train a generator to imitate synthetic LLM novelty scores.

## Primary sources and what they can support

| Resource | What is actually labeled | Best use here | Important limitation |
|---|---|---|---|
| [RINoBench, arXiv:2603.10303](https://arxiv.org/abs/2603.10303), [code/data](https://github.com/TimSchopf/RINoBench) | 1,381 ideas with human-expert novelty judgments, plus automated metrics and rationales | Calibration/test set for a novelty judge; report macro-F1 and per-domain errors | The paper reports that automatic rationales diverge from human gold judgments. Verify the repository license and split before ingestion; do not assume its labels mean scientific validity or feasibility. |
| [Old Ideas, Novel Problems, arXiv:2610.02022](https://arxiv.org/abs/2610.02022), [code/data](https://github.com/allenai/novelty-eval) | OpenReview passages selected for unanimous extreme originality agreement, paired with vanilla-LLM ideas, and judge comparisons | Stress-test prompt/model/retrieval stability and construct adversarial pairs | It is an evaluation of novelty-judge behavior, not a corpus of experimentally validated ideas. The authors show prompt changes can substantially change pairwise accuracy; treat this as a warning against a single judge score. |
| [AgentIdeaBench, arXiv:2609.07611](https://arxiv.org/abs/2609.07611), [code/data](https://github.com/HKUST-KnowComp/AgentIdeaBench) | 40 subfields in five disciplines; static/active protocols, multidimensional scores, and literature-verified critic outputs | Reproduce active retrieval vs static evaluation; use field-balanced splits and feasibility/clarity/specificity dimensions | The release says hypotheses are model outputs scored by models and are not domain-expert validated. Its result that active exploration improves grounding but not measured originality is a reason to keep those dimensions separate. Check the data card: code and data have different obligations. |
| [Idea Novelty Checker, ACL/SDP 2025.sdp-1.9](https://aclanthology.org/2025.sdp-1.9/), [paper](https://arxiv.org/abs/2506.22026), [implementation](https://github.com/simra-shahid/idea_novelty_checker) | Small expert-annotated examples and a facet definition: purpose, mechanism, evaluation, or application difference | Blueprint for retrieval → rerank → facet comparison; seed our evidence schema | Its examples are also used as in-context guidance by the same annotation setup, so the reported agreement gain is not an independent gold standard. |
| [HorizonBench, ACL 2026.aacl-long.202](https://aclanthology.org/2026.aacl-long.202/) | Future discoveries with a strict pre-publication training/retrieval cutoff; soft-recall outcome and a 51-expert user study | Strongest prospective-style test: whether a system can recover work that was not yet available at decision time | Availability and licensing of the full benchmark must be checked before download. Future publication is evidence of eventual uptake, not proof that the earlier idea was good or causally useful. |

## Dataset tiers

1. **Gold human judgments.** RINoBench, the extreme OpenReview pairs, and a small locally adjudicated set. Keep the original label, annotator provenance, field, and source date. Split by idea family and source paper, not random rows.
2. **Silver evidence comparisons.** Our Haiku reviews and retrieved-paper comparisons. These are useful for triage and reranker training only after a human audit. Never promote a model's “novel” label to gold merely because it cites a paper.
3. **Prospective outcome tests.** HorizonBench-style temporal holdouts and our own frozen snapshots. Freeze the corpus and atom vocabulary at date `T`; evaluate only against papers first available after `T`. This is the only tier that tests recovery of later work rather than agreement with a reviewer.

## Required record format

Store one JSONL record per proposal/review with:

```text
proposal_id, atom_ids, proposal_text, claim_delta,
source_ids, source_urls, source_dates, source_quotes,
facet_overlap {purpose, mechanism, evaluation, setting},
closest_work, exact_difference, evidence_supported,
novelty_label, feasibility_label, falsification_label,
reviewer_model, reviewer_version, prompt_hash,
human_status, human_label, adjudication_notes
```

`source_quotes` must be exact excerpts (or a stable abstract field), and every overlap claim must point to at least one source ID. A citation without a supported quote is an unsupported review, not positive evidence. Preserve rejected proposals and retrieval failures.

## Pilot ingestion and adjudication

Start small: 60 proposals (the six current hybrid-search candidates, prior reviewed candidates, and atom-component controls), with metadata/abstracts only from RINoBench, AgentIdeaBench, OpenReview-derived data, and a frozen OpenAlex/arXiv snapshot. Do not download a huge corpus or make paid calls during ingestion.

For each proposal, retrieve a fixed top-k set, then run two blinded Haiku reviews with independently permuted source order. Ask for structured facets and evidence first, then a verdict (`covered`, `incremental`, `plausibly_distinct`, or `unsupported`) and confidence. A human sees the proposal plus the same evidence but not model scores. Route disagreements and low-evidence cases to a third adjudicator. Report agreement and disagreement by field; retain the adjudication queue as the highest-value local dataset.

The human form should separately record:

- whether the cited work actually supports the claimed overlap;
- whether the proposal changes purpose, mechanism, evaluation, or setting;
- whether the distinction is experimentally testable;
- whether the proposal is feasible under stated resources;
- whether the reviewer would search further.

This prevents “novel” from collapsing evidence quality, feasibility, and scientific importance into one unstable number.

## Training and evaluation order

1. Implement retrieval and the strict review schema; establish Haiku/model-swap and prompt-permutation baselines.
2. Measure retrieval recall and evidence precision before optimizing the judge. A novelty judge cannot recover a paper that was never retrieved.
3. Human-label a small, field-balanced disagreement set. Train a reranker or pairwise selector only on source-overlap and facet labels; do not train on raw synthetic verdicts.
4. Use temporal, field-held-out, and author-held-out splits. Keep the atom vocabulary and source index frozen for each cutoff.
5. Only after the evaluator is stable, use accepted distinctions/rejection reasons as a generator reranking signal. Compare against component-only and random atom controls, with matched search budgets.

## Metrics that matter

- retrieval recall of the closest prior work;
- evidence precision: fraction of cited sources whose text supports the claimed overlap;
- pairwise accuracy and macro-F1 on human-labeled novelty sets;
- calibration (Brier/ECE) for confidence, not just accuracy;
- robustness under prompt permutation, source-order permutation, and a second judge model;
- temporal soft recall of later discoveries on HorizonBench-style holdouts;
- diversity and falsifiability of the final proposal portfolio, measured separately from novelty.

Do not optimize a single “novelty score.” The Old Ideas study shows judge scores can move sharply under prompt changes, while AgentIdeaBench shows that better retrieval may improve grounding without improving measured originality. Those are different failure modes and should remain visible.

## License and contamination checklist

Before copying data into the repository, record the exact commit/release, license, fields used, and whether text is redistributed or only IDs/abstracts are cached. Check the data cards for RINoBench and AgentIdeaBench and the OpenReview/arXiv terms. For every temporal experiment, record the retrieval timestamp and exclude post-cutoff abstracts, citations, and model-generated summaries. A model with a later knowledge cutoff must not be used to claim a historical prediction without an explicit frozen-evidence setup.

### Recommended first command

Build `data/benchmarks/manifest.jsonl` containing only dataset URLs, commit hashes, licenses, split metadata, and download checksums. Then ingest a small, auditable subset into `data/reviews/` and run the 60-proposal adjudication pilot before any fine-tuning or large-scale corpus download.
