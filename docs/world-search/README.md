# World-search pilot: runbook

This directory documents the runnable pilot in `scripts/world_search.py`. The
older `WORLD_SEARCH_METHOD.md` describes the fuller 24-case research protocol;
this file is the exact local runbook for the current implementation.

The pilot compares three arms on two fixed sklearn problems (`digits_shift` and
`tabular_missing`), four seed words (`porous`, `mirrored`, `tidal`, `braided`),
and seed `42`. That is 24 branches. Every arm uses three staged calls: a first
analysis/world, a solution, and a compiler that emits the same JSON mechanism
schema. The current pilot intentionally fixes the decoder to
`anthropic/claude-sonnet-4.6` so the arm comparison has no model confound. The
future long-form decoder preference is Opus 5.5; changing to it requires a
fresh plan/output directory and is a new experiment, not a continuation.

The world arm is target-blind at its first stage: its builder sees only the seed
word. The solver then sees the problem and world text, but not the seed or study
purpose. Direct and random-seed controls use the same three-stage shape so the
call budget is visible. No learned Alien scorer is used.

## Environments

Use Python 3.12.14. The normal Alien environment contains PyTorch and the
released scorers. The benchmark environment is a separate lightweight install
from `requirements-benchmark.lock` (NumPy, SciPy, scikit-learn and their pinned
dependencies). Keep them separate: the offline benchmark must remain runnable
without model or scorer downloads. If the benchmark environment has not been
created, use a fresh `.venv-benchmark` directory and install the lock file; do
not modify the main `.venv` while reproducing a run.

## Plan and run

From the fork root:

```sh
source ../.venv/bin/activate  # or use the absolute Python 3.12.14 path
python scripts/world_search.py plan --protocol-version 1 \
  --output results/world-search-pilot
python scripts/world_search.py run --protocol-version 1 \
  --output results/world-search-pilot
```

`plan` writes the immutable `plan.json`. `run` resumes complete calls but
refuses to silently retry failed or changed requests. It uses at most three
concurrent generation branches and a conservative **$3.00** generation reserve
(prompt and completion prices are recorded in `plan.json`). The key is read
from `OPENROUTER_API_KEY` or `~/.config/openrouter/env`; it is never written to
the results. Generation output is under `calls/` and `candidates/`, including
failed calls, request hashes, usage and provider finish reasons.

The saved pilot has 24 candidate records: 21 complete and 3
failed. Persisted call accounting should be read from the JSON receipts rather
than inferred from the number of candidates. The last observed calls used
about 49k input and 56k output tokens, with roughly $2.11 reserved; upstream
billing may differ. Do not launch another paid run into this directory.

## Offline benchmark and validation

Use `../.venv-benchmark/bin/python` for the following benchmark commands. Run the baseline fixture without network access:

```sh
python scripts/world_search_benchmark.py validation \
  --output results/world-search-benchmark/validation-baseline.json
```

Validation constructs the fixed train/validation/test indices and hashes all
three, but computes metrics only on validation. It compares the ordinary
classifier with the existing augmentation baseline on clean and predefined
shift conditions. The test split must remain untouched until a plan is frozen.

The current selected-candidate validation harness is:

```sh
python scripts/world_search_candidates.py \
  --output results/world-search-benchmark/candidates-validation.json
```

It evaluates the fixed candidate IDs declared in the script against the shared
validation conditions. Validation mode records deviations and does not score test labels. The corrected v3 configuration was frozen and evaluated once on the test split. Reproduce it with `--mode final --selection reports/world-search-2026-11-16/benchmark/candidates-validation-v3.json --output results/replay-final.json`. Copy the published candidate JSON files into `results/world-search-pilot/candidates/` first. Preliminary v0/v2 benchmark outputs have confounded controls and are excluded from findings.

## Retrieval and blinded review

Once generation is complete, retrieve bounded OpenAlex abstracts and run
blinded evidence triage:

```sh
python scripts/evaluate_world_search.py \
  --run results/world-search-pilot
```

This exports `evaluation/blind-cases.json`, caches two top-four abstract
searches per candidate, and uses `deepseek/deepseek-v3.2` with a **$0.25**
review reserve. The reviewer sees the normalized proposal and retrieved source
records, not arm labels or seed metadata. Automated decisions remain
`human_adjudicated=false` and `train_eligible=false`. “Insufficient evidence”
means the bounded search did not decide novelty; it does not mean novel.

The evaluator reports exact mechanism-string duplicates and review decisions.
These are descriptive checks. External novelty requires broader search and a
human reading the cited papers; usefulness requires the executable validation.

## Cost and reproducibility rules

Record model name, request hash, seed, stage caps, actual usage, provider
finish reason, and output directory. Generation is capped at $3 and review at
$0.25. Failed or refused calls count as findings and should not be retried in
place; make a new run ID with a changed configuration. Keep all 24 branches in
the analysis even when a provider refuses one.

No Kaggle credentials or leaderboard submissions are needed for this pilot.
The next Kaggle step is a separate, locked validation task with a private local
holdout and three implementation seeds; see `KAGGLE.md`. Do not interpret a
local digits gain as external research novelty.

Run unit tests before changing the harness:

```sh
python -m unittest discover -s tests -q
```


Generation seeds control case ordering and dataset splits, not provider sampling: the API requests do not set a provider random seed. The direct arm has four stochastic replicates; its nominal seed word is unused. Exact cached outputs are preserved, but fresh API calls are not bitwise reproducible.

## Truncation fix for future runs
The saved v1 pilot used1024/1536/1536 stage caps. All three failures occurred at the solver stage when it wrote full Python code. Default protocolv2 now separates mechanism prose from later code generation, forbids full code in these stages, and uses1024/4096/2048 caps. Use a fresh output directory. The$3 reserve remains; settled actual usage frees unused reservations, while pending/unknown calls retain their full reserve. No v1 failures were retried or erased.
