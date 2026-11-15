# Literature reconstruction study — parent-reviewed execution

User authorizes full studies with one Luna agent per candidate. Work in `/Users/altaiavdil/Proga/research/experiments/alien-science/upstream`. Your ID is given in your task message. Read your `reconstruction-studies/<ID>/proposal.json` and referenced primary sources.

This is a new comparison of raw abstracts, flat extracted atoms, and typed molecule composition from the SAME32-paper packet. Source papers and atoms are in `results/atomization-study-v1/expanded-papers.json` and atom JSONL under `atoms/`, `added-atoms/`; proposed component IDs resolve through `context-atoms.json`. These are research hypotheses, not new findings.

Own ONLY your assigned directory. No git changes, commits, global dependency changes, paid API calls, extra agent spawns or external messages. Use primary web research for the closest prior work. Do not read other agents' scores or prior MCAR experiment results.

Use the shared `scripts/missingness_shift_fixture.py` without modification. Seeds9401,9403,9407. Training inputs are already masked; hidden clean values must never be used by your learner. Anchors0,1 always observed. Validation cases are clean/MCAR/MAR; final cases are reserved for one held-out evaluation later. Do not import private fixture helpers or use `test_x`/clean test values to repair other conditions. The prediction function gets only a masked feature matrix, not a condition name or labels. If target-batch adaptation is proposed, declare it and never access target labels.

Environment `../.venv-benchmark/bin/python` has pinned sklearn, NumPy and SciPy. PyTorch only in `../.venv/bin/python`; use NPZ handoff if needed. No new dependencies or models. Execute heavy compute through:
`../.venv-benchmark/bin/python scripts/with_compute_slot.py -- ../.venv-benchmark/bin/python reconstruction-studies/<ID>/experiment.py ...`
Use one CPU thread; initially limit total validation compute to300seconds. Keep all candidate/control training data, corruption draws, sample counts and model capacity matched except for the intervention under test.

Deliver a faithful implementation (or explain precisely why the proposal is not implementable), strongest constituent-method controls, and component-removal ablations. Check algebraic cancellation, source identifiability assumptions, undefined losses/targets, and access to unavailable values BEFORE fitting anything. A gain against weak median imputation is not evidence when a stronger known control matches it.

## Phase one — do now
1. Write README with exact mechanism, primary prior art URLs, assumptions, intended intervention/control, feasible implementation details and any deviations. No claim of novelty.
2. Implement experiment.py with separate `validation` and `final` modes and an explicit refusal to overwrite final output. Prediction must not receive held-out labels or condition names.
3. Run ALL three seeds in validation mode only. Save per-seed/per-condition AP and Brier, runtimes, error logs, split/data hashes, and ablations to validation.json.
4. Save preregistration.json with the fixed metric, effect threshold (predeclare at least0.005meanAP over strongest matched control, no clean AP decrease>0.01), choice rule, selected parameters, source SHA256 and validation artifact hash. Missingness-shift gain must be evaluated across new mechanisms, not just MCAR.
5. Send parent a concise implementation summary, validation findings, and any mechanism problems. STOP BEFORE loading final fixture mode or final test labels. Parent will review the code/controls and explicitly tell you to execute phase two. This is an internal review gate, not a user permission request.

## Phase two — only after parent's follow-up
Run the frozen final evaluation once for each seed and save results.json; no tuning afterward. Write verdict.json containing candidate_id,status(supported_pilot/no_gain/inconclusive/invalid_mechanism/blocked),faithful_implementation,mean_candidate,mean_strongest_control,paired_deltas,source_hash,prior_art_status,novelty_established=false,human_adjudicated=false,limitations. Distinguish implementation validity, utility and external novelty. Never promote an atom.

Fixture clarification: final structural blocks are named block_mcar_2_8 and block_mcar_8_14 because their row-selection is independent of feature values. Only the two own-value conditions are MNAR. Final `test_x` stays None; use final_cases only for evaluation. Do not use the clean condition to repair other conditions.
