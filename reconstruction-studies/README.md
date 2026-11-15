# Literature atomization and reconstruction study — 2026-11-16

We tested a bounded version of the hypothesis: can literature-derived atoms and explicit relationships reconstruct useful new implementations? The source corpus has 16,346 deduplicated metadata/abstract records plus 15 separately labeled OpenAI research summaries. A focused 32-paper packet produced 91 quotation-checked atoms. These are abstract-level units, not full-method extractions.

Six Opus 5.5 proposals compared raw abstracts, flat atoms, and molecule-oriented composition (two outputs per condition). One Luna agent implemented each proposal on the same missingness-shift fixture. Parent review preceded each frozen final run. This is a small feasibility pilot, not a powered comparison of the representations.

## Results

All six retained their preregistered `no_gain` verdict. Below is a secondary descriptive alignment to the same four new mask mechanisms and three seeds; it does not replace the original study-specific primary metrics or select models using test performance.

| Study | Candidate AP | Evaluated control AP | Difference |
|---|---:|---:|---:|
| [abstracts-0](abstracts-0/README.md) | 0.9002 | 0.9014 | -0.0012 |
| [abstracts-1](abstracts-1/README.md) | 0.8124 | 0.8578 | -0.0454 |
| [atoms-0](atoms-0/README.md) | 0.8801 | 0.8994 | -0.0193 |
| [atoms-1](atoms-1/README.md) | 0.5333 | 0.8499 | -0.3166 |
| [molecules-0](molecules-0/README.md) | 0.8482 | 0.8524 | -0.0043 |
| [molecules-1](molecules-1/README.md) | 0.8262 | 0.8383 | -0.0121 |

The best useful constituents were often simpler: conditional imputation with augmentation, native tree missing-value handling, or uniform weighting. Added gates, pseudo-targets, propensity weights, and expert pooling did not establish additional benefit in these implementations. More source material made richer proposals, but it did not by itself validate the new relationships.

## Scientific limits

- Three synthetic dataset seeds and a narrow binary-classification task; no Kaggle submission or real-world SOTA claim.
- Only two generated proposals per representation; different implementation capacities and control families prevent a general causal ranking of abstracts versus atoms versus molecules.
- Several implementations refined ambiguous proposals, including internal training calibration in place of using external validation labels to estimate method parameters; exact deviations are in each README.
- Atoms quote abstracts, and quote checking does not establish semantic entailment or implementation completeness. Published algorithms such as MIRRAMS were not faithfully reproduced as full baselines.
- All validation screens failed. Final results are descriptive and cannot reverse those predeclared outcomes. No new atom or molecule was promoted.

## Input and protocol records

- [Source manifest](source-manifest.json): titles, source URLs, dates, evidence level, abstract hashes.
- [Atom provenance](atom-provenance.json): content IDs and source-span hashes, without redistributing the source abstracts. Full working snapshots remain in the local handoff.
- [Pre-final review](PREFINAL_AUDIT.md), [harmonized results](harmonized-results.json), each study’s preregistration and parent-approved-final record.
- [Generation configuration](generation-plan.json), [accounting](accounting.json), [validation recovery](validation-recovery.json).

One malformed source quotation was rejected. Seven valid atoms from that otherwise complete response were individually recovered without changing their text or making another API call. Four generated proposals initially failed an overly restrictive ID-prefix check; deterministic revalidation accepted arbitrary proposed-component names and normalized known arXiv aliases. Those were schema repairs, not semantic rewrites or additional generations.

## Engine improvements

The downstream checks now include [algebraic/observability preflight](../docs/world-search/PREFLIGHT.md) and a [central controlled runner](../docs/world-search/CONTROLLED_RUNNER.md). Future workers implement an estimator plugin; the runner owns data access, common metrics, validation selection, immutable hashes and final-output protection. This addresses evaluation inconsistencies seen in the first 21 native-agent studies.

Lacuna is the upstream authors’ existing production research map. Mechanism Lab uses bounded research lookups rather than rebuilding or mirroring it; see [Lacuna integration](../docs/corpus-expansion/LACUNA.md).

## Recorded model usage

Extraction plus six Opus reconstructions cost $0.7562 in recorded upstream OpenRouter usage. Native-agent usage and local compute are separate. No paid retries were made.

The next scientific improvement should add full method/code evidence and verified operator semantics, then test a narrow proposed difference against the strongest published implementation. Increasing abstract volume alone is not supported as a solution by this pilot.

Artifact verification: all six code and fixture hashes match their final-approval records. Five validation records use file-byte hashes; atoms-1 uses canonical JSON excluding its self-hash field. Both its content hash and recorded hash match. See [integrity report](artifact-integrity.json).
