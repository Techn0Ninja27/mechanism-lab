# Corpus weights for ML research — 2026-11-16

The corpus is ML-first, with systems and mathematical papers included as
sources of methods, constraints, and transferable mechanisms. Paper influence,
evidence strength, and usefulness for a task are different fields. A citation
count must not overwrite them with a single quality score.

## Metadata to preserve

OpenAlex provides article-level citation counts, field-weighted citation
impact, and citation percentiles normalized for publication year, work type,
and subfield. Preserve the raw provider response and retrieval date because
these values change. See the [official work attributes](https://help.openalex.org/data/works/attributes/).
Missing metadata stays unknown; it is not a zero-citation or low-quality label.

Additional evidence should have source references: available code and data,
matched baselines, ablations, independent reproduction, measured effect,
resource requirements, and retraction/correction status. Code availability
alone does not prove reproducibility, and an LLM assessment is not an
independent replication. No such labels are inferred from citation counts.

## Where weights could help

- Retrieval: task relevance first, with influence as a separately inspectable
  feature and a reserved path for recent or lightly cited sources.
- Atom graph: distinguish how often an edge appears from how strong its
  supporting evidence is. Retain each supporting paper rather than only a sum.
- Feasibility: prioritize mechanisms whose requirements match the available
  data, hardware, and implementation environment.
- Utility learning: use measured experimental gains against declared controls,
  grouped by task and resource budget. Citations are a proxy to test, not the
  utility training target.

A future weighting experiment should freeze the same candidate pool and
compare unweighted, normalized-citation, evidence-only, and combined variants.
Report age/field coverage and task outcomes as well as link recovery. An
improvement in predicting popular connections would not establish improved
novelty. Do not retrospectively add present-day citation counts to a temporal
backtest: future citations leak information unavailable at its cutoff.

## Current status

The first atom-graph experiment uses no citation weights. Its nodes come from
the original atom catalog; its edges record co-occurrence, not causal or typed
mechanism relationships. Citation metadata collection is being added to the
larger corpus; evidence-weighted selection is not yet validated or deployed.
