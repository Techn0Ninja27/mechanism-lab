# Corpus expansion and evaluation plan

## Why the current evidence is too weak

The first world-search pilot had a useful generation/control comparison, but
its literature check was only two lexical OpenAlex searches with four abstracts
per query. `insufficient_evidence` therefore means that this small search did
not decide the question. It does not mean that the proposal is novel or that it
is disproved.

The original Alien scorer has a separate provenance problem. The released
coherence rows match `papers/clusters_80.json` exactly, while the canonical
descriptions in `clusters.json` have only 0.88% exact numeric-label agreement
on shared non-noise units. The two files must never be joined by cluster ID.
Any expanded corpus must preserve the exact source artifact, paper ID, text,
date, and version lineage together. A larger contaminated corpus would make
the result less trustworthy, not more useful.

## What a source record contains

Store one immutable JSON record per work or released artifact:

```json
{
  "source_id": "doi:10... or arxiv:2401.01234",
  "canonical_id": "doi:...",
  "version": "v2",
  "title": "...",
  "authors": ["..."],
  "venue": "arXiv|OpenAI|conference|repository",
  "published_date": "2024-01-15",
  "updated_date": "2024-04-02",
  "landing_url": "https://...",
  "abstract": "...",
  "full_text_available": false,
  "license": "...",
  "retrieved_at": "2026-11-16",
  "retrieval_method": "arxiv_api|openalex|official_landing_page",
  "content_sha256": "...",
  "mechanism_units": [],
  "limitations": [],
  "evidence_level": "abstract_only"
}
```

`source_id` identifies the exact retrieved version; `canonical_id` joins
versions without erasing them. DOI, arXiv ID, OpenAlex ID, and repository URL
are aliases, not interchangeable primary keys. Keep withdrawn, superseded,
and duplicate records with an explicit status. A title match alone is not a
deduplication decision.

Each extracted mechanism unit should be a claim with provenance, rather than a
bag-of-words atom:

```json
{
  "unit_id": "sha256:...",
  "source_id": "arxiv:...",
  "span": "verbatim paragraph or table reference",
  "intervention": "what is changed",
  "causal_interface": "what receives the change and through what state",
  "assumptions": ["..."],
  "prediction": "what differs from the baseline",
  "baseline": "the closest comparison",
  "failure_modes": ["..."],
  "tasks_tested": ["..."],
  "evidence_level": "full_text|abstract_only|author_summary",
  "extractor": "human|model|rule",
  "adjudicated": false
}
```

An extracted unit is not automatically a new atom. The same intervention can
be represented by several papers, and a new label can be a synonym. Keep the
raw span so a reviewer can reject an extraction without losing the source.

## Retrieval coverage, not a single novelty score

For every proposal, run a fixed retrieval panel and save the query/result
ledger:

1. Exact mechanism phrases and aliases.
2. Component conjunctions and typed relation phrases.
3. Intervention plus target domain.
4. Failure mode, baseline, and measurement terms.
5. Citation and author/venue expansion from the strongest matches.
6. A date-bounded search for work published before the proposal's generation
   cutoff.

Report recall proxies instead of pretending that search absence is novelty:

- `query_count`, `source_count`, and abstract/full-text coverage;
- unique canonical works after version deduplication;
- agreement of top results across OpenAlex, arXiv, official lab pages, and
  citation expansion;
- rank stability under paraphrased queries;
- `known_control_recall`: whether deliberately supplied known papers are
  recovered;
- `citation_chaining_yield`: new canonical works per expansion step;
- exact and semantic overlap with the proposal, with human adjudication status;
- unresolved search terms and missing full text.

Use a small calibration set containing known covered ideas, near misses,
incremental ideas, and genuinely unrelated controls. The reviewer must recover
the covered controls before its “plausibly distinct” outputs are useful. Do not
train a novelty model from the same retrieval records used to generate the
proposal.

## Leakage and split rules

The corpus is versioned by publication date and source hash.

- For a historical experiment, freeze the corpus at a cutoff date before
  generation. Do not use updated abstracts or later papers in the generator.
- Keep a later period as a hidden evaluation set. A paper cited in the prompt,
  atom catalogue, retrieval context, or human calibration set cannot be a clean
  test example.
- Split by canonical paper family, author group when practical, and venue/topic
  cluster—not by abstract rows. All permutations of one paper stay together.
- Deduplicate DOI, arXiv versions, repository mirrors, and title/abstract near
  duplicates before assigning splits. Keep the alias map and the decision.
- Do not evaluate a “future” proposal against a full-corpus vocabulary. The
  existing `generation.temporal_data` split is explicitly a fixed-vocabulary
  diagnostic because the catalogue was built from all years.
- Keep source metadata, extracted mechanism units, and generated proposals in
  separate tables. A proposal may cite a source for review, but that source
  cannot also serve as an unseen transfer test.

The minimum honest split is: pre-cutoff source corpus for extraction and
generation; validation sources for prompt/retrieval design; post-cutoff or
held-out paper-family sources for evaluation; and a separate executable task
set for usefulness. Report overlap at every boundary.

## Cheap next experiment

Do not retrain the 273-atom scorer yet. First run a corpus-backed molecule
experiment on two tasks with the same 24-branch direct/random/world design:

1. Build a small, frozen source pack of 200–500 primary records from arXiv,
   official OpenAI/lab pages, and the existing corpus, with dates and hashes.
2. Extract typed mechanism units and their limitations. Give each generator
   only the source pack appropriate to its arm; the world builder remains blind
   to the target and corpus.
3. Ask the compiler to output a molecule: component units, directed relation
   types, assumptions, baseline, prediction, and cheapest ablation. Preserve
   the fictional-world rule and the reality bridge separately.
4. Retrieve prior art using the six-query panel above, including known controls.
5. Select two molecules per arm before seeing outcomes. Test component removal
   and the strongest existing-atom decomposition on a validation task.
6. Promote nothing. A candidate is only a reusable-abstraction hypothesis if
   it recurs across independently seeded tasks, compresses the mechanism ledger
   after charging for its definition, separates outcomes counterfactually, and
   transfers to a held-out task.

The cheapest kill rule is that the world arm does not produce more valid,
non-duplicate molecules with an executable difference than the matched random
seed arm after retrieval coverage is equalized. If it only produces stranger
language, stop expanding the world generator and improve the relation extractor.

## Ingestion tests before any model training

An ingestion command should fail closed unless all of these pass:

- every record has a stable source ID, source URL, date, retrieval timestamp,
  content hash, and declared evidence level;
- aliases map to exactly one canonical work or are marked ambiguous;
- duplicate title/abstract hashes are grouped without deleting versions;
- paper IDs and years join the assignment corpus without collisions;
- no source appears in both generation context and hidden evaluation;
- extracted spans are exact substrings of the stored source text;
- every mechanism unit resolves to one source hash and one span;
- atom/molecule IDs are namespaced by the artifact and revision;
- train, validation, and test manifests have disjoint canonical source IDs;
- re-running ingestion produces the same manifest and SHA-256 output.

The first implementation can use JSONL plus SQLite and standard-library
hashing. It does not require a vector database. A retrieval result that cannot
be reproduced from its URL, query, date, and response hash is a lead, not
training data.

## What “more input” should improve

More papers should improve coverage of mechanisms, alternatives, and failure
modes. It should not merely increase atom co-occurrence counts. Measure the
expanded system on held-out mechanisms and tasks, not on reconstruction of the
same papers used to build its vocabulary. Keep the old fixed-vocabulary scorer
as a baseline, compare it with a typed molecule ledger, and only then consider
retraining a scorer with a new vocabulary.
