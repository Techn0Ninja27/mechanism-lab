# Grounded generation and workbench iteration — 2026-11-16

This iteration connects project documents, primary code sources, generated proposals, critical reviews, and executable operator checks. It also merges the separate local-workbench branch into Mechanism Lab so the application and research engine ship together.

## What changed

- The workbench shows all 27 earlier studies with original verdicts and recorded audit qualifications separately. Primary and harmonized secondary metrics stay distinct.
- Selected project documents can become bounded, line-referenced source packs. Text/Python/JSON is treated as data, never executed on import. Unsupported PDF extraction is not silently claimed.
- An allowlisted job generates and reviews one proposal using Opus. A shared cross-process $10 phase ledger and $2 run ceiling reserve cost before requests; there are no automatic paid retries or BYOK exemptions assumed.
- Generated research shows machine critiques separately from hash-verified executable operator diagnostics.
- Pinned official implementation files, rather than only abstracts, now ground this pilot. The source fetcher verifies SHA-256 before use. Public manifests retain provenance without bundling unlicensed source snapshots.

## Two actual generated proposals

**Normalization axis:** the generator proposed changing the random-feature normalization axis in StableMiss. The reviewer identified binary-feature collapse. Direct numerical checks confirmed the collapse and NaNs on constant columns, so the idea was rejected before training. See [proposal/review summary](grounded-01.json) and [executable audit](../../operator-studies/normalization-axis/README.md).

A subsequent paired sine/cosine representation passed structural contracts, but the audit also verified that any such per-element binary embedding is affine in raw indicators. Raw masks are the simpler baseline. This is an engineering check, not a new research primitive. See [contract audit](../../operator-studies/mask-embedding-contract/README.md).

**Shared NeuMiss recurrence:** the generator proposed shared matrix powers with learned scalar coefficients. Parent review corrected its oracle approximation: the exact recurrence cannot represent an arbitrary polynomial. A development-only numerical test fit the constrained coefficients across masks and measured inverse approximation error. This does not establish trained predictive gain. A fuller prior-art search then found that matrix sharing was already introduced by the authors in 2021. The initial fitted-polynomial advantage did not survive stronger classical controls: at degree 3, relative inverse error was 0.0603 for the candidate versus 0.0062 for Chebyshev; at degree 5, 0.0044 versus 0.0004. These are supplementary numerical diagnostics on the same development covariance family, not new predictive experiments. Network training is not justified by this gate. See [proposal/review summary](grounded-02.json), [operator diagnostic](../../operator-studies/neumiss-polynomial/README.md), and [prior art](../../docs/method-grounding/neumiss-polynomial-prior-art.md).

## Cost and limits

The four actual model calls cost $0.67222 in conservatively counted API usage, leaving $9.32778 of the phase ceiling. Native-agent usage and local compute are separate. [Receipts](api-usage.json) preserve actual usage and response identifiers without credentials or full prompts.

There is no established novel method or predictive improvement from this iteration. The evidence instead improves the engine: exact code grounding exposes invalid operations; executable tests can verify an LLM critique; and method-version coverage prevents a plausible proposal from rediscovering an author's later implementation. Two proposals are not a controlled comparison of generation quality.

## Run

Start the local app with `./workbench/start.sh`. The [workflow guide](../../docs/GROUNDED_RESEARCH.md) covers document selection, paid execution, and boundaries. Fetch pinned source snapshots with `python scripts/fetch_method_sources.py`; operator-study READMEs list their numerical commands.

The ongoing goal remains active. Next iterations should require stronger published controls and method-version coverage before expensive training, and should test additional problem families rather than treating this small missingness pilot as general evidence.
