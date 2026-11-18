# Proposal-readiness replay

Frozen 2026-11-18. This replay applies the readiness classifier to four
historical local grounded results. It is a structural development consistency
check, not a blind scientific-quality benchmark.

Run from the repository root:

```sh
../.venv/bin/python scripts/replay_proposal_readiness.py \
  --manifest reports/proposal-readiness-v1/manifest.json \
  --output /tmp/proposal-readiness-replay.json
```

The manifest pins each original result by SHA-256 and records expected workflow
dispositions. The portfolio record also pins all three candidate dispositions.
The replay records the classifier SHA-256 and refuses changed source results or
changed expected statuses. It never edits the original results, calls a model,
or launches an experiment.

The historical result files are local development artifacts and are not present
in a fresh clone. Reproduction therefore requires the same local result tree;
the manifest is a provenance check, not a portable dataset. Coverage labels in
the underlying reviews are overlap accounting, not novelty or usefulness proof.

The output reports `auto_execute: false` for every record. A `test_design_candidate`
status would only mean that a human can inspect the proposed test design; it does
not authorize execution.
