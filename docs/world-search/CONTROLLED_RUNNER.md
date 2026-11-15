# Centralized experiment runner

The first native-agent studies used separate evaluation scripts. That exposed inconsistent metric aggregation, weak controls, and post-hoc protocol changes. Future implementations should be reviewed estimator plugins; the runner owns fixture access, splitting, scoring, selection, and the final gate.

A plugin implements `build(params, seed)` returning an object with `fit(masked_train_X, train_y)` and `predict_proba(masked_X)`. The runner never passes held-out labels or condition names to prediction. This is an API boundary, not a security sandbox: plugin source still requires review and must not read unrelated files or call private fixture helpers.

```sh
../.venv-benchmark/bin/python scripts/with_compute_slot.py -- ../.venv-benchmark/bin/python scripts/controlled_study.py validation --config examples/controlled-study/config.json --output results/runner-smoke
../.venv-benchmark/bin/python scripts/controlled_study.py freeze --config examples/controlled-study/config.json --output results/runner-smoke
../.venv-benchmark/bin/python scripts/with_compute_slot.py -- ../.venv-benchmark/bin/python scripts/controlled_study.py final --config examples/controlled-study/config.json --output results/runner-smoke
```

The example is ordinary extra-missingness augmentation and exists to exercise the harness. It is not a new research contribution or independent replication of an earlier result.

Validation selects a single candidate and strongest control using mean AP across the three non-clean validation conditions and all three seeds. Freeze records configuration, plugin, fixture and validation hashes. Final verifies them, evaluates only frozen choices, and refuses overwrite. Its primary metric is AP across the four new mask mechanisms; clean and MCAR are separate. The screen requires at least0.005 AP gain and at most0.01 clean AP loss. A failed validation screen remains failed even if descriptive final scores improve. No automatic novelty or atom promotion follows from a positive score.

Use a fresh output path for a new experiment. Preserve failures rather than deleting or silently resuming a final run. Check resource use and run through the shared three-slot gate. Results remain synthetic, small-sample evidence until tested on independently held-out datasets and reviewed against prior work.
