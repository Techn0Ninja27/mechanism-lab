# Fixed-operation extraction on three ML methods

2026-11-17. Three operations were named before model extraction from an inspected
[primary-source packet](../contract-eval-packet-v1/README.md). Opus extracted drafts
and a separate Opus call reviewed them. Both calls finished with `stop`; all
three cards passed exact-quote, source-ID and schema validation.

| Method | Model review | Observed issue |
|---|---|---|
| GradLite Algorithm 1 | Revise | Draft names the missing residual estimator as unknown, but does not expose low-rank factors as required dependencies. |
| PacTrain communication branch | Revise | Review questions the mask boundary and finds an unsupported dense representation assigned to the full synchronization branch. |
| PERA polynomial update | Accept limited draft | Draft preserves the different B/A expansions and coefficient initialization; executable completeness remains unproven. |

The model reviewer is also fallible. Whether a value is an external input or
internal state depends on the chosen operation boundary; a reference to an exact
gradient in an approximate residual equation does not prove the algorithm can
access that gradient. See the [independent audit](independent-audit.md) for an
assessment of both stages. No card was curated or promoted into a molecule.

The extractor independently identified the undefined GradLite residual estimator,
unspecified basis construction and accumulator initialization, PacTrain's undefined
mask stability criterion and compression format, and PERA's unclear materialization
and scaling. These are useful missing-detail findings, not a contract accuracy
score or evidence of novelty/usefulness. The three selected papers are new to this
local workflow, not demonstrated unseen model-training data. The packet is small,
exposed and not representative. There is no controlled comparison with the earlier
unscoped extraction runs.

## Reproduce and inspect

The local source pack has 26,261 included characters in eight chunks. Rebuild it
with `../.venv/bin/python reports/contract-eval-packet-v1/build_packet.py` using
the retained local snapshots. Those full texts are not redistributed here.

```sh
../.venv/bin/python scripts/extract_contract_drafts.py \
  --pack results/contract-eval-packet-v1/pack.json \
  --operation-scopes results/contract-eval-packet-v1/scopes.json \
  --output /tmp/contract-plan-fresh --extractor opus
```

This creates an offline plan; `--execute` makes paid calls. A changed scope or
code hash refuses the old plan. Existing attempts are retained without automatic
retry. The two calls share a $1 run ceiling and the $20 phase ledger.

[Receipts](call-receipts.json) record 36,813 input and 7,526 output tokens, including
reported reasoning tokens. Conservative charge was **$0.297772**; the shared phase
ledger reached **$4.8168796/$20**, with all charges reconciled. [Plan](plan.json)
records the exact input hashes, requested operations and code hashes.

The [public result](result.json) replaces evidence quotes with hashes. The original
local artifact supports exact-span replay; the redacted export does not. The
webapp's draft library displays all three records separately from executable
operators. Model acceptance never sets curation, novelty or training eligibility.
