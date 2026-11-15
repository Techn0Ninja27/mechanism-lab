# Candidate source ablation — 2026-11-16

Frozen released scorers, 10,000 attempted draws/source, seed42, three atoms per set, MPS. Source pools contain 8,124 autoregressive, 9,482 hybrid and 9,985 uniform distinct triplets. Hybrid uses half model/half uniform draws. Comparing top30 with z-scores from the shared deduplicated union:

| beta | source | effective atoms | Jaccard Vendi | absent from released paper sets |
|---|---|---:|---:|---:|
| 0.4 | autoregressive | 69.9 | 29.3 | 0 |
| 0.4 | hybrid | 67.8 | 29.2 | 0 |
| 0.4 | uniform | 67.0 | 29.2 | 22 |
| 0.7 | autoregressive | 65.6 | 29.1 | 18 |
| 0.7 | hybrid | 68.5 | 29.2 | 25 |
| 0.7 | uniform | 72.1 | 29.4 | 30 |

Absence means no single released corpus paper contains all three atoms. It is not an external literature novelty claim. Model scores are proxies; lower availability can trade away coherence. The selected 30-candidate portfolios already approach the Vendi ceiling of30, so tiny differences should not be oversold. No new paid decoder calls or human quality evaluations were made for these candidate pools.

Reproduction commands are in FORK.md. Machine-readable results: results/source-comparison.json. Primary conclusion: search source materially changes which combinations are considered and retained; quality superiority remains untested. These runs are separate from the earlier six Fable-decoded proposals, which are frozen in the parent experiment's handoff folder.

At beta0.7, mean raw coherence was -5.349 (autoregressive), -6.700 (hybrid), and -7.234 (uniform); higher is better under this scorer. Mean unavailability was -0.240, -0.208 and -0.198 respectively; again higher is preferred. The broader pools bought lower availability by sacrificing considerable modeled coherence. This tradeoff is why corpus absence alone cannot pick the winner.
