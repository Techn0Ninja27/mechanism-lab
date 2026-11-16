# Mechanism Lab Workbench

A local research application for organizing source documents, running the Alien Science sampler, inspecting candidates against original evidence, recording reviews, and exporting handoffs.

## Start

In this research workspace:

```sh
./experiments/alien-science/webapp/start.sh
```

In a clone of the GitHub fork:

```sh
./workbench/start.sh
```

Open **http://127.0.0.1:8765**. Keep the terminal running. Stop with Ctrl+C. Use `--port 8766` if that address is occupied. The UI and document workspace need Python 3.10+ only: no npm, account, API key, or model download.

## Work on a project

1. Create a project and give it a useful name.
2. Import files or a folder. Supported files: PDF, Markdown, text, JSON, CSV, and Python source (read as text). Folder structure is preserved inside `sources/`; imports are copies. Identical file contents are deduplicated. Conflicting filenames are rejected rather than overwriting originals.
3. Read source documents and create notes or evidence reviews. PDF originals can be downloaded; PDFs are not automatically parsed, summarized, or embedded.
4. Open Engine runs to configure an offline run or rerank an existing scored pool. Sampling uses the installed engine corpus, **not the newly imported documents**. Choose the project to keep a run reference alongside your sources.
5. Inspect candidates, their exact atom descriptions, original source units, saved proposal artifacts, and validity warnings. Record a human decision.
6. Check selected text documents and use **Build source pack** to create a bounded local evidence pack and an offline grounded-research plan. Optional prior-art queries are saved with the run (up to three, 200 characters each). They remain local unless **Search arXiv metadata using these queries** is explicitly checked; without queries, network search is disabled. The plan can start one fixed grounded generation/review run (up to $2, subject to the shared $10 phase cap); it accepts no model, budget, output-path, or arbitrary-command overrides.
7. Export the project as a ZIP containing source files, notes, reviews, run references, README, and a machine-readable manifest.

See [where documents go](docs/FOLDERS.md), [engine installation](docs/ENGINE_SETUP.md), and [architecture and boundaries](docs/ARCHITECTURE.md).

## Engine requirements

The Files & setup view checks local engine paths. Browsing documents works before installing model dependencies. Sampling requires the pinned inference environment and source/model assets. Existing results are discovered without importing or moving them.

The Studies view reads the fork's 27 committed study records through a read-only audit ledger. It keeps raw verdicts, audit qualification, primary metrics, secondary metrics, and limitations separate. A study verdict never becomes an automatic novelty claim.

Legacy released assignments have a confirmed provenance mismatch. Their scores and draft interpretations are diagnostic, not evidence of scientific novelty. The app keeps that warning attached to candidate inspection. Repaired source links do not certify checkpoint lineage. Paid generation/review and corpus training remain explicit CLI workflows; this app never makes automatic paid retries.

## Verify

From this research repository:

```sh
python3 -m unittest discover -s experiments/alien-science/webapp -p 'test_*.py'
node --check experiments/alien-science/webapp/static/app.js
```

From the fork, substitute `workbench` for `experiments/alien-science/webapp`. Tests use temporary project storage and do not call paid services. Local runtime data is excluded from Git.

## Operator library and local checks

The Operator Library displays known typed operators and compatible chains from the engine catalog, with source references. Its local checker accepts bounded integer inputs and verifies an exact encode/decode round trip using registered operators. It makes no model/API calls. A passing check demonstrates that input round trip only; it does not establish novelty or general correctness. See the engine repository README for frozen benchmark results and the negative utility-ranker result.
