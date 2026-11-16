# Grounded research workflow

Date: 2026-11-16.

This workflow converts selected source documents into one narrow proposal, a separate critical model review, and an executable operator test when feasible. Model acceptance is not a novelty or utility verdict.

## Evidence and generation

1. In the workbench, select project documents and build a source pack with a research question. This is offline. UTF-8 Markdown/text/JSON/Python sources are supported; PDFs remain document attachments until separately extracted.
2. Inspect the selected sources and truncation metadata. The pack records byte hashes, source levels, line spans and a canonical digest. Imported content is untrusted evidence, never code to execute.
3. Generate and review with the app action or `scripts/grounded_research.py --pack PACK --question QUESTION_FILE --output NEW_RUN --execute`. Without `--execute`, the command only writes a plan.
4. Read the proposed operator delta, assumptions, observability paths, matched control and kill threshold. Source quotes are checked exactly, but that is not proof of entailment.
5. Verify the reviewer’s objection with an algebraic or numerical check before training. If useful, design a fresh preregistered study using the central runner. Previously exposed final fixtures cannot confirm an adaptive improvement.

Opus 5.5 is currently used for both proposal and review. Their judgments are correlated; the second call is a critique, not independent experimental validation. Source and code retrieval are separate from the model, and the model cannot fetch additional papers itself. External novelty review is still required.

## Spending and failure behavior

Each run is capped at $2 in conservative reservations. All new runs share `results/improvement-phase-2026-11-16/budget.json`, capped at $10. No BYOK exemption is assumed. Provider price ceilings, UTF-8-byte input bounds and output-token limits determine reservations before requests. Failed/unknown calls retain reservations unless a verified receipt establishes actual usage. Actual receipts are retained even if a provider overruns a reservation.

Calls are cached by request identity; changed inputs need a new run directory. There are no automatic paid retries. Existing final results are not overwritten. A failure can therefore require a new, explicitly recorded experiment rather than an invisible retry. Parallel clients share a file-locked ledger.

This application is local-only. A Generate action sends the explicitly selected evidence pack and question to OpenRouter. Credentials are read by the subprocess from the existing local configuration, never placed in the browser or project exports.

## Reproducing the official-source diagnostics

Run `python scripts/fetch_method_sources.py` to fetch the exact pinned source snapshots with hash verification into ignored local storage, then run the commands in `operator-studies/*/README.md`. The downloaded implementation is not bundled into public study artifacts. Source hashes and commit URLs are in `docs/method-grounding/sourcepack-manifest.json`.

## Iteration evidence

The first code-grounded proposal changed a random-feature normalization axis. A reviewer predicted that binary mask features would collapse. A direct numerical audit confirmed the collapse and constant-column NaNs; the idea was rejected before training. A repaired paired sine/cosine representation passes structural tests, but is affine in the raw binary mask. Raw indicators remain the simpler baseline; no new atom is promoted.

The second proposal’s apparent operator advantage disappeared against stronger Richardson and Chebyshev controls. Its matrix-sharing premise also overlapped NeuMiss 2021. Future evidence packs must include method-version lineage and the strongest simple classical control, not only the first implementation encountered.

## Retrieval before generation

Protocol v2 accepts `--search-config FILE` containing `queries` (up to three explicit strings, each at most 200 characters) and `allow_arxiv` (boolean). Planning is always offline. Paid execution first searches the local corpus, and optionally arXiv, before constructing the model client. The same controls are exposed in the app’s source-pack dialog.

The default corpus is the integrated-v2 snapshot when present, otherwise the earlier expanded snapshot. Missing corpus access is recorded as an error, not a successful zero-hit search. Retrieved abstracts receive separate chunk IDs and evidence-level labels; they do not become full method implementations. Query strings are compiled into literal-token searches. Cache files have request-parameter and response-hash checks. Results are balanced across query/provider groups within a fixed limit.

The curated method-lineage registry currently covers only NeuMiss/NeuMiss_sota and StableMiss/StableMiss+. It distinguishes exact pinned implementation files from README or abstract metadata. A known successor’s absence is a coverage warning, not evidence that the original is current or novel. This registry is agent-checked, not human adjudicated, and is not a comprehensive literature graph.

A real protocol-v2 run recorded an arXiv HTTP 429 and continued with one local abstract plus the original code pack and a missing-successor warning. It explicitly recognized the published shared-weight successor. That is one observed case, not a controlled estimate of retrieval’s effect on idea quality. The alternative proposal still failed a numerical fixed-point test.
