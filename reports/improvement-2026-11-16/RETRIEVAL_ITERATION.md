# Retrieval and method-version iteration — 2026-11-16

The previous source-code generation rediscovered NeuMiss’s already-published shared-weight successor. This iteration adds retrieval and explicit version coverage before generation.

## Implemented

- Bounded local FTS retrieval, optional arXiv metadata queries, response-hash caching, explicit errors, and provider-balanced retention. No paid calls during retrieval.
- An offline merge of the focused 32-paper packet with the prior corpus: 16,355 deduplicated works, preserving all 16,346 previous canonical IDs. The earlier corpus and an intermediate legacy-only rebuild are retained; the integrated-v2 snapshot is the default.
- A curated two-family lineage registry. Exact code file + commit + evidence type is required for implementation coverage. READMEs and abstracts do not count as code reproduction.
- App controls for explicit query strings and arXiv opt-in. Source-pack creation remains offline; generation performs the saved search before its first model call.
- Immutable retrieval artifacts, unchanged-plan checks, a whole-run file lock, and source/evidence provenance in the output.

## Actual run

[grounded-03](grounded-03.json) used the same original NeuMiss pack and question as grounded-02, with the new pipeline. It found the 2021 paper locally. arXiv returned HTTP 429; the error was retained and no automatic retry occurred. The output explicitly treated missing NeuMiss_sota code as a baseline gap and shared weights as published prior art.

It proposed a conditional diagonal-preconditioning diagnostic instead. The model reviewer identified a wrong fixed point. An [executable numerical test](../../operator-studies/jacobi-fixed-point/README.md) confirmed the literal recurrence’s relative inverse error of 0.297–0.409 on three fresh small SPD cases. Allowing a global scale did not remove the error. The corrected iteration matched ordinary standardized-coordinate Richardson. No method was promoted or trained.

This is one adaptive case study, not a controlled ablation or evidence of improved scientific discovery rates. Retrieval remains incomplete; the lineage registry was curated from the preceding failure and is not an independent benchmark.

## Spending and next boundary

Six model calls across grounded-01/02/03 cost $0.963008 in conservatively counted API usage, leaving $9.036992 of the original shared $10 phase cap. Native-agent usage and local compute are separate.

The source-grounded path currently provides a direct-generation reference, alongside the original atom sampler. The next important integration is to turn verified executable mechanisms into typed atoms and compose them under operator contracts, then compare against this direct-generation reference. A generic LLM critique alone is not the Alien methodology and is not enough to establish useful atom recombination.
