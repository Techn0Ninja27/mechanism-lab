# Where documents and outputs go

The app's **Files & setup** view shows the absolute paths on your computer. Create a project before importing documents. Names entered in the app are display names; a stable project ID identifies the folder.

```text
<workspace root>/
├── workspace/
│   ├── workspace.sqlite3          Project and document index
│   ├── .app/
│   │   ├── jobs/                 Durable job records
│   │   └── adjudications.json    Candidate decisions
│   └── projects/<project-id>/
│       ├── README.md             Folder guide for this project
│       ├── sources/              Imported originals, including folder structure
│       ├── notes/                Reading notes and working observations
│       ├── reviews/              Evidence checks and written decisions
│       ├── runs/                 JSON references to engine jobs and parameters
│       └── exports/              Downloadable project ZIPs
└── results/
    └── webapp-jobs/<job-id>/      Engine outputs, logs, scored pools, rankings
```

| You have | Put it here | How |
| --- | --- | --- |
| Papers, source PDFs, Markdown, text, JSON or CSV | Project `sources/` | Import files or Import folder |
| Personal observations, reading notes | Project `notes/` | New note |
| Prior-art comparisons, assessments | Project `reviews/` | New review |
| Candidate decision | `workspace/.app/adjudications.json` | Save decision in candidate inspector |
| A new engine run | `results/webapp-jobs/<job-id>/` | Start run; optional project gets a JSON reference in `runs/` |
| A portable handoff | Project `exports/` | Export project |
| Existing frozen results | Their original `results/` directories | Select the saved run; no need to move them |
| Engine source catalogue | Engine `papers/clusters.json` | Install engine assets, not project import |
| Learned checkpoints | Engine `models/` | Install engine assets, not project import |

Do not drag files into the folders behind the app to register them. Use import so the database records hashes and paths. Moving indexed files manually will make the document unavailable; originals outside the project are never moved or deleted by import.

Limits per import: 100 files, 10 MiB per file, 40 MiB total. Split larger folders into batches. Hidden files and unsupported extensions are rejected. Source paths cannot contain traversal, absolute paths, backslashes, or symlinks. An existing path with different content is rejected; rename the new version before importing.

An export includes document files and run **references**, not large checkpoint files or entire candidate pools. Preserve `results/` separately for full engine reproducibility. Back up the whole `workspace/` directory, including `workspace.sqlite3`, to preserve the app index; ZIP exports are document handoffs rather than database backups.

The workspace root defaults to the experiment directory here and the repository root in a standalone fork clone. `ALIEN_WORKSPACE` changes storage location; `ALIEN_REPO` selects the engine checkout. Set these explicitly when storing work outside the source checkout.
