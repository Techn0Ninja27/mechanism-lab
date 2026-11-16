# Install the optional inference engine

Document projects work with Python 3.10+ alone. The optional sampler uses Python 3.12 and the pinned inference dependencies distributed with this fork.

From a fork checkout:

```sh
uv venv --python 3.12.14 .venv
uv pip sync --python .venv/bin/python local_runner/requirements.lock
.venv/bin/python workbench/download_assets.py
./workbench/start.sh
```

`uv` must be installed separately. The explicit asset download fetches public Hugging Face files at the revisions in `local_runner/assets-manifest.json`, verifies SHA-256, and extracts data archives. Expect hundreds of megabytes plus extracted files. The app does not download assets at startup.

Installed paths:

- `papers/clusters.json`: original source assignments and canonical atom descriptions.
- `papers/clusters_80.json`: legacy expanded catalogue with known provenance concerns.
- `models/coherence_model/`: coherence checkpoint and token mapping.
- `models/availability_model/`: availability checkpoint, author embeddings, and mappings.
- `data/`: released paper data and associated assets.

This restores the released inference assets, not a newly validated training corpus. Read the repository's `PROVENANCE.md` before interpreting scores. A fresh clone has no saved local candidate pool: run an offline sample first, then rerank that saved pool. A missing pool is an actionable error, not a reason to fabricate results.

For this research workspace, the existing engine is `experiments/alien-science/upstream/` and its environment is `experiments/alien-science/.venv/`. The app detects that layout. Avoid rerunning the older experiment setup script against a modified fork checkout because it deliberately checks a pinned revision.

Advanced location override:

```sh
ALIEN_REPO=/absolute/path/to/alien-space-of-science \
ALIEN_WORKSPACE=/absolute/path/to/my-research \
python3 /absolute/path/to/alien-space-of-science/workbench/server.py
```

Put a Python inference environment in `<ALIEN_WORKSPACE>/.venv` or `<ALIEN_REPO>/.venv`. Otherwise the server's Python interpreter is used. Files & setup shows the chosen interpreter and expected model paths.
