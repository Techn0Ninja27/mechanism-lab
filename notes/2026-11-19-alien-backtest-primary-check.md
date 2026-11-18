# ALIEN temporal backtest: primary-source check

2026-11-19. Source check, not a rerun or generation-prompt input.
[arXiv:2603.01092v3](https://arxiv.org/pdf/2603.01092v3), §5.2 and Appendix C.2.

- The backtest rebuilds 191 atoms from pre-2025 material and trains both scorers through 2024. Frozen centroids and threshold precede 2025 projection.
- Candidate space: C(191,3) = 1,143,135. The 2,424 targets comprise 299 recurring and 2,125 exact-new triples; 1,411 are strictly novel.
- At β=0 (coherence-only), top-10,000 recovers 762 targets: all 299 recurring, 463 exact-new, and 29 strictly novel. Overall recall is 31.4%; uniform expectation is 21.2 hits, giving 35.9× enrichment.
- Strictly-novel recall is 29/1,411 = 2.1%. Its uniform expectation is 12.3 hits, giving 2.35× enrichment. The fraction 29/762 = 3.8% describes recovered-hit composition, not recall.
- At β=.7: 10 total hits, approximately 0.5× chance; Table 12's seven exact-new hits are not automatically a strictly-novel count.
- 26.4% of 40,934 2025 units are unmappable and excluded. Uniform ranking is the temporal chance baseline.

Strict novelty means no prior co-occurrence, even within larger atom sets; it is corpus-relative. Upstream Gemini atomization's independent knowledge-cutoff control remains unspecified. Released full-corpus artifacts and the paper's separately described temporal reconstruction must not be conflated.
