# Research ledger export

`scripts/export_research_ledger.py` creates a read-only JSON view for the local workbench:

```sh
../.venv/bin/python scripts/export_research_ledger.py --output results/research-ledger.json
```

The export combines the 21 candidate studies and six literature-reconstruction studies. It preserves each artifact's raw verdict separately from the repository audit qualification. Primary metrics remain separate from the secondary harmonized reconstruction comparison; the latter is explicitly descriptive and must not be treated as a replacement for preregistered metrics.

The server can call `build_ledger(root)` directly without writing a file. Six reconstruction failure assessments are explicitly grounded in the reconstruction assessment note and study READMEs; these labels are evidence annotations, not automatic causal inference. Missing fields remain `unknown`.

Artifact references are repository-relative and include SHA-256 digests. Missing or malformed optional fields are represented as `unknown`, `null`, or an empty list. The exporter never edits study artifacts, promotes atoms, assigns novelty, or treats model-generated judgments as human adjudication. It is safe to regenerate after a study is added.

## Composition and utility experiments (2026-11-16)

The ledger also includes the frozen integer-composition study and the held-out
utility-ranker study. Their original verdicts remain separate from the check
that frozen source files and validation artifacts match their recorded hashes.
Hash agreement is not independent replication. A mismatch is exposed as
`artifact_mismatch`; it does not rewrite the historical verdict.

The workbench resolves each cohort's artifacts within its registered directory,
so the positive compression pilot and negative utility result can both be read
through `/api/studies`. Neither establishes novelty or promotes a model.

## Graph-walk codec development replay (2026-11-17)

The ledger now includes `graph-walk-codecs`: two public development graphs with
five seeds each, exact codec reconstruction and serialized-byte counts. The
candidate failed against fixed-width controls even with a free mapping table.
Same-code reproduction is recorded, but does not establish independent
replication or GPU/system performance. The originating generated proposal also
links to this result through its operator-evaluation entry.
