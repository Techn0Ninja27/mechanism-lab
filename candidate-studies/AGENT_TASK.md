# Full candidate study instructions — 2026-11-16

You are one native Luna worker explicitly requested by the user. Your assigned candidate ID is in your task message. Work in `/Users/altaiavdil/Proga/research/experiments/alien-science/upstream`. Read `candidate-studies/<ID>/proposal.json` first.

Goal: carry this candidate through prior-art investigation, faithful implementation, controlled validation, ablations, frozen held-out evaluation, and a clear reproducible verdict. An honest negative or an unimplementable mechanism is a useful result. Do not invent novelty or substitute a different technique while calling it the original proposal.

## Ownership and coordination
- Own only `candidate-studies/<ID>/`. Other agents work concurrently; do not modify shared scripts, docs, dependencies, git state, or other candidates. Do not commit or push.
- No paid API calls, no credentials inspection, no external messages, no Kaggle submissions, no new model weights. Use web research on primary papers/docs for prior art.
- Read shared `scripts/world_search_benchmark.py` for fixture conventions, but do not read earlier benchmark scores or other agents' reports. Fresh candidate-specific validation should determine choices.
- Heavy compute MUST run through the shared three-slot gate, from fork root:
  `../.venv-benchmark/bin/python scripts/with_compute_slot.py -- ../.venv-benchmark/bin/python candidate-studies/<ID>/experiment.py ...`
  Set models to one CPU thread. Limit total actual experiment CPU work to roughly 120 seconds initially; if infeasible, report the limitation instead of silently replacing the method. The gate may wait: use exec with yield_time_ms=1000 and poll, not blocking tools indefinitely.
- Python 3.12.14, benchmark environment `../.venv-benchmark/bin/python` has NumPy2.2.6, SciPy1.16.3, sklearn1.7.2, joblib1.5.2, threadpoolctl3.6.0. Main `../.venv/bin/python` has PyTorch2.15.0 and NumPy2.5.4 but not sklearn. If PyTorch is necessary, export arrays to NPZ using the benchmark environment, train with main environment, then score outputs with benchmark environment. Do not downgrade/install into either environment.

## Scientific procedure
1. Identify the exact proposed causal mechanism. Separate factual errors or contradictory equations from implementation details. Search primary prior work for its mechanism and nearest simpler alternative; keep exact URLs and distinguish known components from a possibly different relationship.
2. Implement faithfully with matched controls: same splits, corruption draws, classifier capacity, sample counts and preprocessing fit on training only, except the factor explicitly under test. Add component removal or replacement ablations. Document any deviation; a material approximation is not a test of the original idea.
3. Use three predeclared seeds 137,271,389. Digits: sklearn.load_digits, stratified60/20/20. Tabular: make_classification5000rows20features5informative5redundant, weights0.8/0.2, dataset seed and stratified60/20/20 split from each seed. Tabular metric average precision over clean +10%,30%,50% MCAR. Digits mean accuracy over clean, noise0.2/0.4/0.6, dropout0.05/0.15/0.30, joint(0.4,0.15),(0.6,0.30), normalized pixel intensity noise (scale16). Same corruption draws across compared methods.
4. Complete validation for ALL seeds before reading any final test score. Write `preregistration.json` with chosen hyperparameters, control, metric, pass/fail rule, source-code SHA256 and split hashes. Then run each held-out split once. Do not tune after looking at held-out results. These are exploratory replications, not proof of significance or external novelty.
5. Report per-seed metrics, paired differences, train/inference runtime and ablations. If some variants fail, record exceptions and retain counts. Do not drop failures to make the average look better. Never promote a new atom on your own.

## Deliverables inside your directory
- `README.md`: exact proposal, mechanism, prior art with verified links, implementation and deviations, commands, results, smallest falsification test, verdict and limitations.
- `experiment.py` (and small helper files if needed): reproducible offline experiment; separate validation/final modes, explicit output paths. Avoid overwriting frozen test outputs without an explicit rerun flag.
- `preregistration.json`, `validation.json`, `results.json` (or explain exactly why evaluation could not run).
- `verdict.json`: candidate_id, status (`supported_pilot`, `no_gain`, `inconclusive`, `invalid_mechanism`, `blocked`), faithful_implementation boolean, seeds, primary_metric, mean_candidate, mean_control, paired_deltas, prior_art_status, novelty_established=false, human_adjudicated=false, limitations.

Send a concise result to parent, identifying any blockers early. Work through the full authorized experiment rather than only writing a plan.
