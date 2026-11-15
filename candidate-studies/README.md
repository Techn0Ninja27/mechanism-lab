# Candidate studies — 2026-11-16

Each of the21 complete world-search candidates received a separate native Luna study. These reports are exploratory. Agent-authored `supported_pilot` labels are not discoveries and can compare against a weak baseline; read the independent audit and component ablations.

All21 studies have result/verdict files and matching preregistered source hashes. Matching hashes do not repair post-hoc preregistration, prior test exposure, or a weak control.

| Candidate | Agent label | Audit classification |
|---|---|---|
| [digits_shift-direct-0](digits_shift-direct-0/README.md) | no_gain | clean_negative |
| [digits_shift-direct-1](digits_shift-direct-1/README.md) | supported_pilot | provisional_or_compromised |
| [digits_shift-direct-2](digits_shift-direct-2/README.md) | no_gain | clean_negative |
| [digits_shift-direct-3](digits_shift-direct-3/README.md) | supported_pilot | provisional_or_compromised |
| [digits_shift-random_seed-0](digits_shift-random_seed-0/README.md) | no_gain | clean_negative |
| [digits_shift-random_seed-1](digits_shift-random_seed-1/README.md) | no_gain | clean_negative |
| [digits_shift-random_seed-2](digits_shift-random_seed-2/README.md) | no_gain | clean_negative |
| [digits_shift-random_seed-3](digits_shift-random_seed-3/README.md) | no_gain | clean_negative |
| [digits_shift-world-0](digits_shift-world-0/README.md) | supported_pilot | provisional_or_compromised |
| [digits_shift-world-1](digits_shift-world-1/README.md) | supported_pilot | provisional_or_compromised |
| [digits_shift-world-2](digits_shift-world-2/README.md) | invalid_mechanism | invalid_mechanism |
| [digits_shift-world-3](digits_shift-world-3/README.md) | supported_pilot | provisional_or_compromised |
| [tabular_missing-direct-0](tabular_missing-direct-0/README.md) | inconclusive | inconclusive |
| [tabular_missing-direct-1](tabular_missing-direct-1/README.md) | invalid_mechanism | invalid_mechanism |
| [tabular_missing-direct-2](tabular_missing-direct-2/README.md) | supported_pilot | provisional_or_compromised |
| [tabular_missing-random_seed-1](tabular_missing-random_seed-1/README.md) | supported_pilot | provisional_or_compromised |
| [tabular_missing-random_seed-2](tabular_missing-random_seed-2/README.md) | invalid_mechanism | invalid_mechanism |
| [tabular_missing-world-0](tabular_missing-world-0/README.md) | no_gain | clean_negative |
| [tabular_missing-world-1](tabular_missing-world-1/README.md) | inconclusive | inconclusive |
| [tabular_missing-world-2](tabular_missing-world-2/README.md) | no_gain | provisional_or_compromised |
| [tabular_missing-world-3](tabular_missing-world-3/README.md) | supported_pilot | provisional_or_compromised |

[Audit narrative](AUDIT.md) · [Structured audit](audit.json) · [Artifact hashes](artifact-integrity.json)

Main findings: constant feature weights can be canceled by StandardScaler; explicit mask features can be redundant with native missing-value routing; absent stable anchors make anchor reconstruction untestable; ensembles can lose to their strongest member. Useful component effects are retained, but no new atom is promoted.

Data arrays/model weights are excluded from Git. Each study includes its code, seed/split description and environment assumptions. Some studies document protocol violations; those outputs remain labeled exploratory and must not be pooled as clean evidence.
