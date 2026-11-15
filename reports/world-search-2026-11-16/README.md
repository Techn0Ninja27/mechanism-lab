# World search and molecules: first falsification pilot — 2026-11-16

Question: does a target-blind invented-world detour yield useful mechanisms beyond equally staged direct generation or random-word inspiration?

24 branches, two tasks, four repeats/seed words, three arms. Sonnet4.6 was fixed across the experiment; future long-form decoding defaults to Opus5.5 at low effort. All arms receive the same requested completion-token caps; actual input/output tokens and costs differ. API sampling is not seeded, so the saved artifacts are reproducible offline but fresh generations are not bitwise reproducible.

| Arm | Complete | Truncated/failed |
|---|---:|---:|
| direct | 7 | 1 |
| random_seed | 6 | 2 |
| world | 8 | 0 |

All failures are retained; no automatic retries. Completion is not quality. This is a pipeline pilot, not a powered comparison of creative ability.

## Executed digits tests

One already-completed proposal from each arm was fixed before performance was examined. We then repaired implementation confounds on validation, froze the configuration, and ran one held-out evaluation. The v0/v2 exploratory benchmark outputs are excluded.

| Mechanism | Held-out mean accuracy | Matched control |
|---|---:|---:|
| Gaussian smoothing | Validation selected sigma=0 (no smoothing) | 0.73642 |
| Sinusoidal severity schedule | 0.65309 | Uniform schedule: 0.65401 |
| Structured block dropout | 0.73549 | Matched random dropout: 0.73642 |
| World-derived margin-weighted ensemble | 0.73426 | Uniform voting: 0.73457; single random-dropout model: 0.73642 |

The selected digits interventions did not improve over their simpler controls. These tiny differences are descriptive, not statistically significant claims. No macro-atom was promoted. Only three generated candidates received executable digits tests. The separate tabular screen below tests two other generated mechanisms with stated modifications; remaining proposals are not empirically validated. Baseline fixtures also cover synthetic tabular missingness; any separate tabular candidate results are included in benchmark/.

## Tabular mechanism screen

Held-out mean AP: median-only0.81919; median+augmentation0.86326; native NaN handling+Betaaugmentation0.86879; adding mask indicators0.86879 (no gain); native NaN+fixedaugmentation+mask0.86650; modified world soft-imputation alpha0.7+mask+fixedaugmentation0.87033. Alpha was chosen on validation before one held-out evaluation. The modified world composition gained0.00383 AP over the fixed-rate control and0.00155 over the stronger Beta baseline. This is a small single-split signal requiring independent replication.

The direct trainable-MLP proposal was not implemented. The world formula was tested with an explicit training-augmentation extension; its prose incorrectly equated alpha0 with median imputation, whereas the formula yields zero imputation. This is a modified mechanism screen, not exact reproduction or scientific novelty.

## Evidence reviews

{"incremental": 2, "insufficient_evidence": 17, "plausibly_distinct": 2}

DeepSeekV3.2 compared proposals to two OpenAlex searches, at most four abstracts each. The generator was Sonnet, reducing same-model judging but not establishing independence or truth. Citation IDs/quotes are mechanically checked. Insufficient evidence means this bounded retrieval did not resolve novelty. No labels are training-eligible or human-adjudicated. Initial cached reviews used the raw common-schema proposal; subsequent reviews use the normalized blind export. Neither includes arm metadata, but this version difference is retained.

## What was built

- Target-blind world builder, solver and real-world compiler with staged controls, locks, caches, token/cost receipts and bounded spending.
- Molecule records with typed relations, immutable definitions and preserved decomposition.
- Promotion checks for artifact existence/hash, positive validation, independent held-out task, ablations and human-adjudicated prior art. These checks cannot certify that a human assertion is truthful; they prevent accidental unsupported promotion.
- Local executable benchmarks, component ablations, blind literature triage, mechanism-family census, and a research plan for learned abstraction libraries.

## Costs

```json
{
  "generation": {
    "calls": 69,
    "prompt_tokens": 48978,
    "completion_tokens": 55874,
    "upstream_cost_usd": 0.985044,
    "reserved_usd": 2.105808
  },
  "review": {
    "calls": 21,
    "prompt_tokens": 71066,
    "completion_tokens": 93914,
    "upstream_cost_usd": 0.0439143552,
    "reserved_usd": 0.10728818000000001
  }
}
```

Reported upstream inference costs come from persisted receipts; reserved amounts are conservative request estimates, not charges. No Kaggle submission or account access was used.

## Reproduce and inspect

See [runbook](../../docs/world-search/README.md), [molecules](../../docs/world-search/MOLECULES.md), [mechanism census](../../docs/world-search/MECHANISM_CENSUS.md), and [new primitives research](../../docs/world-search/NEW_PRIMITIVES.md). Original source inspiration: Henrik Westerberg’s [ontology-of-the-alien](https://github.com/emergent-wisdom/ontology-of-the-alien) and [issue4](https://github.com/alejandrohdez00/alien-space-of-science/issues/4). This is a new adapted experiment, not a replication of his published study.

The next justified improvement is a stronger mechanism extraction and prior-art comparison stage, followed by matched tests of relations between components. More elaborate prose or extra components do not earn admission into the atom vocabulary.

Review-provider caveat: despite `reasoning.enabled=false` and a requested2048-token cap, DeepSeek receipts include reasoning tokens and total completions above2048. We report actual usage; the request settings were not an enforced total-token limit on this route. The three generation truncations instead had zero reasoning tokens and hit their1536 visible-output limit. Future provider budget enforcement must be verified per route.
