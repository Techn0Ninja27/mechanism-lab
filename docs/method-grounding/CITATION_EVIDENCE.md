# Saved citation evidence in proposal generation

2026-11-17. Grounded research protocol 7 can attach selected abstracts from a
saved OpenAlex neighborhood to both proposal and review stages. This extends
the standalone citation retriever without fetching live graph data inside a
paid generation call. It does not implement Connected Papers' ranking algorithm.

First create a bounded snapshot using `generation.citation_neighborhood` (see
[the citation inspection](../../reports/citation-neighborhood-v1/README.md)).
Resolve the seed's identity deliberately; preprint and proceedings versions may
have different provider records. Select at most eight work IDs after inspecting
the neighborhood. Selection is explicit, not an automatic relevance judgment.

Create a configuration next to the snapshot:

```json
{
  "snapshot_path": "neighborhood.json",
  "selected_ids": ["W4407013152", "W4396628137", "W7160826828"]
}
```

The path is resolved relative to the configuration file. This CLI option reads
only a user-selected local artifact; it does not accept an URL to fetch. Selected
IDs must belong to the seed's one-hop neighborhood. Records without abstracts
remain explicitly omitted. Each included abstract is capped at 4,000 characters
and the total at 16,000; selection order determines which fit. Truncation is
recorded per evidence chunk. Edges are typed retrieval leads, not claim coverage.

Use the existing grounded pipeline:

```sh
python scripts/grounded_research.py --pack path/to/pack.json --question path/to/question.txt --citation-config path/to/citations.json --output results/new-citation-plan
```

Planning is offline and makes no model calls. It binds the raw snapshot hash,
selection, bounded evidence and dependency hashes. `--execute` uses the existing
shared model budget and requires unchanged inputs. Quotes in either model stage
may cite the added `citation:W…` chunk IDs and must match the selected text
exactly. The result saves a bounded provenance summary; prompts/call artifacts
retain the actual evidence. Older plans must use their recorded code revision;
protocol changes intentionally invalidate them.

The webapp displays citation evidence on completed runs that contain the new
summary. Historical runs remain `not_recorded`; malformed summaries are marked
`invalid`. The browser does not offer a new path input or silently fetch papers.
Generation from the existing UI still uses its existing source-pack workflow;
the optional citation configuration is currently a CLI capability.

## Verification and limits

Four new tests exercise omission versus missing evidence, size limits, selection
and edge validation, source changes after planning, and a fully mocked
proposal/review pass quoting the added evidence. No paid call is needed to test
that integration. A real offline plan in `results/citation-pipeline-v1/plan/`
includes four neighboring-paper abstracts totaling 5,735 characters alongside
SNI's selected methods. It is a plan, not a new generated proposal or experiment.

Hashes bind local bytes; they do not authenticate provider content or prove
scientific correctness. Current citation links are not historical evidence.
The selector does not infer relevance, deduplicate papers against other source
packs, resolve DOI/preprint aliases, or judge novelty/usefulness. Citation-only
provider IDs do not automatically establish matches to the curated arXiv
experiment-feedback registry. Keep primary methods and matching source URLs in
the main pack when needed. These limits remain relevant even when validation
passes. No framework benchmark improvement is claimed from wiring this feature.
