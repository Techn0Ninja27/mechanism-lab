# Bounded proposal pipeline

These scripts are snapshots of the local research harness. Set `ALIEN_WORKSPACE` to a working directory and `ALIEN_REPO` to this fork checkout. Credentials come from OPENROUTER_API_KEY or ~/.config/openrouter/env and are never committed. Install dependencies from the pinned local requirements.lock; obtain source/model assets using the asset manifest.

The workspace must contain results/proposal-v2-review-lessons.md and results/proposals-v3/prior-art/{candidate-id}.md. `proposal_v2.py select` reads the saved results/hybrid-50000-seed45/k3/scored_candidates.json pool under ALIEN_REPO. It uses candidate IDs only as **unranked concept seeds**, strips scores, and attaches original clusters.json evidence. This is deliberately not an end-to-end reproduction of semantic scorer ranking.

`proposal_v2.py decode` is paid: six candidates, at most two concurrent requests, no automatic retries, default Opus5.5 (low reasoning effort) and16384 maximum output tokens. DECODE_MODEL and DECODE_MAX_TOKENS can override these within the bounded limit. Full responses and receipts are persisted before checking completion. PROPOSAL_OUTPUT selects a fresh result directory relative to the workspace (default results/proposals-v3). Existing complete outputs are skipped.

`retrieve_v2.py` caches bounded OpenAlex queries (currently the six v2 seed topics); `review_v2.py proposals` runs evidence comparisons on complete outputs. Model judgments remain ineligible for training without human adjudication. The retrieval script retains its explicit v2 cache path; copy evidence into the active output directory only when topics match.

Do not copy old scored interpretation claims into new proposals. Run the provenance audit described in ../PROVENANCE.md before any new scorer-to-description join.
