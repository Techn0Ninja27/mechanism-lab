# Archived-source reproducibility run — 2026-11-16

The current implementation reproduces v1's entire validation history, selected
step (175), validation NLL (4.035401291272088), and test NLL
(3.9867861903067254) exactly on this machine. This is a repeat on an exposed
split, not independent replication or an additional generalization result.

Before training, the runner archived the exact trainer, model implementation,
and token-mapping implementation; recorded the input hash, configuration and
environment; and refused a nonempty output directory. It saved the selected
checkpoint, mapping and their hashes before test evaluation. It checked the
source and input bytes again before writing the final summary.

The adjacent source_snapshot contains those archived files. The original v1
source-recovery limitation is preserved in the parent report; this new run does
not retroactively repair v1's provenance. No availability model was trained.

## Reproduce

Use Python 3.12.14, torch 2.15.0, and numpy 2.5.4. Obtain the original
`papers/clusters.json` from the documented upstream data assets; its SHA-256
must equal `99b157f1baedba7908fca0236fb7dca23b3a49da44054ada305035036aa27013`.
Run from the repository root in that environment:

```sh
python reports/source-coherence/reproduction-v2/source_snapshot/scripts/train_source_coherence.py --clusters papers/clusters.json --output results/source-coherence-reproduction --seed 42 --max-steps 400 --eval-every 25 --batch-size 64
```

The new output includes source snapshots, run-start manifest, canonical split
sets, source-unit hashes, validation history, checkpoint, mapping and selected
model test NLL. CPU execution uses two threads. Different library versions or
hardware may affect exact numerical agreement; compare recorded environments.

The corrected baseline audit is included as a receipt: test unigram NLL is
4.318623822259427, conditional pair NLL 4.662451109854124. These are co-occurrence
baselines, not feasibility, novelty, or proposal-quality metrics. No paid calls
were used.
