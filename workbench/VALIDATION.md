# Application verification — 2026-11-16

- 18 automated tests pass, including document import/read/export over HTTP, isolated persistent decisions/jobs, path and symlink rejection, duplicate handling, filename-conflict preservation, PDF originals, and project run references.
- JavaScript syntax check passes.
- Clean standalone checkout without model assets: server starts; engine readiness is false; project creation, nested document import/read, and ZIP export succeed.
- Browser: project creation, Markdown import/read, note creation, export creation, exact folder map, run configuration, run completion, and source inspection checked. No browser console errors; no horizontal overflow in the tested narrow layout.
- Real engine job 08532c4c478a: 100 uniform draws, seed 484, top-k 5, beta 0 and 0.4, completed with a saved project run reference. No paid calls.

Boundaries: project documents are a local library, not automatic atomization/training inputs. Original model assets are optional and require explicit installation. Legacy scorer interpretation remains unvalidated; paid model calls are not launched from this app. ZIP exports contain run references, not large checkpoints or complete scored pools.
