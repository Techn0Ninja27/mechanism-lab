# Three-candidate serving portfolio

2026-11-17. The new optional `--portfolio-size 3` path completed one two-call
Opus generation/comparison run. Both outputs ended with `stop`. Conservative
receipt cost: **$0.539892**; shared spending: **$4.2840365/$20**. No experiment
was launched from the selection.

| Candidate | Proposed central change | Comparison review |
|---|---|---|
| cand-00 | Lower-bound admission with retained-prompt restart | Selected for inspection; needs explicit memory/transfer semantics |
| cand-01 | Geometric slicing with retained-prompt restart | Revise: shares cand-00's retention mechanism, with a different trigger |
| cand-02 | Pool reallocation triggered by retained-cache pressure | Reject: reviewer reports overlap with Arrow's load trigger |

The generator produced three full candidate records in one call; these are
not independent samples. A separate call reviewed every candidate and all
three pairs, citing the same bounded source evidence. The two retention
candidates have different declared-field fingerprints and no lexical flag,
yet the reviewer calls their central mechanism the same. Native review agrees
on that overlap. This is a useful diagnostic of the comparison path, not a
benchmark proving improved creativity, usefulness, or novelty.

The selected candidate remains underspecified. [Transition accounting](transition-accounting.md)
distinguishes move-back, persistent duplication, and discard/re-prefill. Its
capacity and cost implications must be resolved before any efficacy test.
[Audit notes](audit-notes.md) preserve additional limits.

## Engine behavior

Protocol10 adds the optional mode; the default remains one proposal plus one
review. Portfolio mode keeps the same $2 run cap and shared $20 ledger, with
12,288 output tokens per stage. Stage0 emits three complete proposals, then
local code assigns IDs cand-00..02. Every proposal and review uses the existing
claim and exact-evidence validators. Stage1 covers each candidate and unordered
pair once and may select one candidate or none. All candidates remain stored.

Fingerprinting normalizes whitespace and JSON-key order in baseline,
operator_delta and data_paths, preserving case and data-path order. Case is
significant for mathematics. Exact duplicate declarations are flagged, and a
later exact duplicate cannot be selected over its first occurrence. Lexical
Jaccard overlap flags are only inspection hints. Neither fingerprint matching
nor reviewer judgments establish algorithm equivalence. All automatic
experiment-eligibility flags remain false; there is no novelty score.

The workbench now displays the complete portfolio, review decisions, pairwise
comparison and nullable selection. Malformed metadata is labeled while valid
candidate cards remain visible. Existing single-candidate results are unchanged.
The selected proposal is also projected into the existing detailed view.

## Evidence and reproduction

The run used the existing five-paper source pack plus ten retrieved local
abstract leads and source-scoped experiment feedback. It did not retrain the
legacy atom rankers or atomize the entire corpus. Sources and feedback are
bounded; a proposal selection is not research validation.

[Redacted result](result.json), [plan](plan.json), [receipts](call-receipts.json),
[question](question.txt), and [post-run audit](factual-audit.json) are saved.
Verbatim quotes remain in the local validated artifact
`results/serving-portfolio-v1/decode/result.json`; the public export replaces
them with hashes and is not an exact-span replay fixture.

Offline planning with a fresh output path:

```sh
../.venv/bin/python scripts/grounded_research.py --pack results/serving-problem-v2/pack.json --question reports/serving-portfolio-v1/question.txt --search-config reports/serving-portfolio-v1/search.json --portfolio-size 3 --output /tmp/portfolio-plan
```

Adding `--execute` makes bounded paid calls; do not use it merely to reproduce
validation. Unit tests use mocked calls, including no-selection retention,
all-pair coverage, duplicate identity, case-sensitive math and malformed UI
records. The real run demonstrates one successful execution, not a controlled
comparison against the one-proposal method.

## Follow-up source and interface audit

The [retention prior-art audit](retention-prior-art.md) identifies existing
recompute, prefix reuse, distributed KV retention and admission/retention
co-design. No distinct, complete retention operator is established by the
selected candidate. It remains a scoped accounting question, not ready for
an efficacy run. The [source-contract check](../serving-contracts-v1/README.md)
separately makes the lower-bound versus exact-peak interface difference
explicit; it does not reject an adaptation that supplies a valid eviction rule.
