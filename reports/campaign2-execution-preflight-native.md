# Campaign 2 execution-path preflight

Date: 2026-11-19. Read-only inspection; no Cora training, seed runs, network
calls, or source edits were performed.

The post-generation path has the expected split. `select_development` in
`generation/campaign_selection.py` requires one complete run for every
inventory candidate at each development seed 100–115, scores only the exact
window 88–119, rejects any failed/non-finite candidate seed, and selects one
candidate per call using the development mean and candidate-ID tie break.
`validate_selection` re-derives that selection from the bound development
aggregate before a final run is allowed. It checks the inventory hash, code and
dependency hashes, and the development split.

`run_causal_campaign.py::execute` adds only the frozen selected candidates,
hold, and the hash-bound stronger control for final execution. Final seeds are
200–231. It writes a freeze record before loading Cora, and each cached job is
checked for candidate ID, seed, call ID, and arm. `final_report` requires the
selection, final aggregate, hold runs, stronger-control manifest, final seeds,
dependency hashes, and exact inventory/run keys; it rejects foreign or missing
candidate-seed rows and checks program hashes. The downstream Campaign 2
decision classifier is applied after final scoring and preserves the legacy
decision for diagnostics.

The new Campaign 2 inventory schema is compatible with this path: every valid
candidate has a canonical program hash, call ID, arm, and compiled AST. The
inventory loader accepts up to 128 rows; the frozen Campaign 2 inventory is
124 (96 generated candidates, 24 seeded random controls, and 4 fixed linear
controls). The new random controls use the same AST hash convention and are
validated by `load_inventory` before selection.

Two integration limitations remain observational rather than newly introduced
blockers. The generic runner does not itself enforce that a Campaign 2
inventory has exactly 124 rows or that every expected arm has six calls; the
Campaign 2 plan/generation layer and `campaign2_decisions` enforce those
denominators. Also, `report_final` is a report-only CLI path: it requires the
already recorded final aggregate and never launches training itself. These are
consistent with the frozen workflow, where generation and local evaluation are
separate stages.

Existing mocked campaign-run tests cover final selection before data loading,
development-only selection, failed candidates, cached metadata, program-hash
validation, final hold runs, and final report binding. No post-generation
execution was run in this preflight.
