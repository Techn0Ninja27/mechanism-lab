# Application architecture

`server.py` serves a local HTTP API and static UI. `workspace.py` manages project files and a SQLite index. `static/` contains the dependency-free browser application. `start.sh` starts the server; it binds to loopback and rejects remote mutation origins.

The project library is independent of the inference engine. Documents become indexed source material, not model inputs. Transforming a new corpus into atoms requires the engine's extraction, clustering, provenance checking, dataset construction, and training workflows. Those steps may require paid services and additional dependencies and are not launched by importing a folder.

Engine operations are allowlisted subprocesses, never arbitrary shell commands. Sampling runs the repository's bounded local runner with decoding disabled. Reranking calls `alien.py rerank` on the selected saved pool. Results go to unique job directories; saved results are not overwritten. One heavy engine job runs at a time. Cancellation signals the subprocess group. Server shutdown stops owned children. Interrupted jobs are retained for inspection rather than automatically retried.

Human adjudications remain distinct from generated reviews and are not automatically treated as training labels. Legacy scorer validity warnings reflect the repository's provenance audit; the UI does not claim to have repaired model semantics.

Project files, SQLite metadata, job records, model checkpoints, credentials, and local research results are excluded from publication. GitHub contains source code, tests, and documentation. Never put API keys into imported source documents; imports preserve bytes by design.

Important endpoints: `/api/projects` for projects; `/api/projects/{id}/documents` for import/indexing; `/api/projects/{id}/folders` for paths; `/api/projects/{id}/notes` for notes/reviews; `/api/projects/{id}/export` for ZIP creation; `/api/setup` for local readiness; `/api/jobs` for engine operations; `/api/candidates` and `/api/evidence/{id}` for inspection. All mutations use JSON and same-origin requests.

Grounded generation is now an explicit allowlisted job for a saved source pack in the selected project. The server constructs the command, revalidates integrity and containment, and prevents duplicate execution. Each run is capped at $2 within the shared $10 phase ledger. The Generated research view separates machine review from empirical validation. Imported documents are sent to OpenRouter only after the Generate action. Legacy paid decoder execution remains disabled.
