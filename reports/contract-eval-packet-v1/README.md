# Full-method contract evaluation packet

Prepared 2026-11-17. This is an exposed development packet for testing the
contract-draft extractor. Packet preparation made no model calls; the subsequent
[scoped extraction run](../full-contract-extraction-v3/README.md) is recorded separately.

The deterministic local selection used exactly three queries against
`corpus-ml-complete-mirror-v6` (739,964 records), selected the first eligible
unseen arXiv IDs from each local FTS top-20 list, and froze the choices before
fetching primary HTML:

* [Backward-Friendly Optimization, arXiv:2510.22467](https://arxiv.org/html/2510.22467v1)
* [PacTrain, arXiv:2505.18563](https://arxiv.org/html/2505.18563v1)
* [Polynomial Expansion Rank Adaptation, arXiv:2604.11841](https://arxiv.org/html/2604.11841v1)

The selected excerpts total 26,269 characters in a validated 26,261-character
source pack with no pack truncation. Each source has inspected normalized HTML
line ranges covering a named method substep, its assumptions or algorithm, and
limitations/evaluation context where available. Every excerpt includes an
omission notice; it is not a complete paper or a completeness claim. HTML
snapshots are retained locally under `results/contract-eval-packet-v1/` and
are not redistributed by this report.

The packet is intentionally separate from the previous DMRS/Chen contract
pilot. It supplies three narrow targets to a later extractor/reviewer:
GradLite's error-feedback update, PacTrain's mask-stability communication
branch, and PERA's polynomial Delta-W construction. The operation scopes were
written after reading the selected primary text and before model extraction;
they are frozen review metadata, not model-generated labels or evidence of
novelty. Following a pre-extraction audit, the labels were narrowed to the
named algorithmic operations: GradLite Algorithm 1's projected-gradient and
error-feedback step, PacTrain Algorithm 1's stable/unstable `model_update`
communication branch, and PERA's distinct `Poly²(B)PolyH²(A)` update. These
labels are targets for extraction/review, not answer keys or claims that the
paper uses any stronger interpretation.

[Selection](selection.json), [source receipts](receipts.json), [manifest](manifest.json),
and [pack summary](summary.json), plus the frozen [operation scopes](scopes.json), retain hashes, URLs, source versions, line
ranges, operation scopes, and the corpus/exclusion provenance. The public
artifacts contain metadata and receipts only; selected text and HTML remain
local. The extractor takes this packet with `--operation-scopes` pointing to the
frozen scopes file. The separate run records both extraction and review.

This packet does not establish novelty, usefulness, representativeness,
unseen-data generalization, or contract accuracy. It is a reproducible input
for testing whether full-method extraction preserves interfaces and unknowns.

[build_packet.py](build_packet.py) replays the normalized line-range selection
from the local HTML snapshots, checks scope IDs against the manifest, verifies source
and excerpt hashes, rebuilds the bounded source pack, and refuses silent
truncation. Run it from the repository root with
`python3 reports/contract-eval-packet-v1/build_packet.py`. It performs no
network access and does not run extraction.

The frozen selection and manifest contain a shortened PERA title (“Low-Fine-Tuning”).
The primary arXiv HTML title verified on 2026-11-17 is “Polynomial Expansion Rank
Adaptation: Enhancing Low-Rank Fine-Tuning with High-Order Interactions.” This is
a metadata transcription discrepancy, not evidence of a different paper version.
The original frozen files remain intact.

The builder replays excerpt construction from retained snapshots. It does not
re-run the FTS selection or prove that selection preceded the fetch; selection
order is documented in the receipt rather than independently timestamp-attested.
