# Research ledger export

`scripts/export_research_ledger.py` creates a read-only JSON view for the local workbench:

```sh
../.venv/bin/python scripts/export_research_ledger.py --output results/research-ledger.json
```

The export combines the 21 candidate studies and six literature-reconstruction studies. It preserves each artifact's raw verdict separately from the repository audit qualification. Primary metrics remain separate from the secondary harmonized reconstruction comparison; the latter is explicitly descriptive and must not be treated as a replacement for preregistered metrics.

The server can call `build_ledger(root)` directly without writing a file. Six reconstruction failure assessments are explicitly grounded in the reconstruction assessment note and study READMEs; these labels are evidence annotations, not automatic causal inference. Missing fields remain `unknown`.

Artifact references are repository-relative and include SHA-256 digests. Missing or malformed optional fields are represented as `unknown`, `null`, or an empty list. The exporter never edits study artifacts, promotes atoms, assigns novelty, or treats model-generated judgments as human adjudication. It is safe to regenerate after a study is added.
