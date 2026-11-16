# Integer-composition telemetry study

This is a preregistered, CPU-cheap mechanism check for lossless signed-integer telemetry compression. The object is a raw stream of 4096 signed 64-bit integers. It tests whether composing a source-aware integer representation (first differences, zigzag, and varints) with ordinary compressors is useful, while charging every scheme identifier, length, block header, fallback payload, and other framing byte.

The fixture is in `fixture.py`. `load("development", seed)` and `load("validation", seed)` return only the input tuple. Each of the three seeded streams is a mixed composition of four source families—linear/smooth random walks, bursty stationary runs, periodic signals with sparse spikes, and broad uniform values—not a separate replicate for each family. The validation seeds are independent of development seeds. Final mode changes the distribution using piecewise mixed blocks, drift resets, geometric jump regimes, and alternating run/random sections. Final values must not be generated or read before parent approval; the API makes that explicit with `allow_final=True`.

## Protocol

The parent benchmark runner supplies the identical input stream to every chain and owns implementations, timing, and result files. Candidate chains are raw signed-64 little-endian bytes; zlib and lzma over that representation; delta-1 or delta-2 + zigzag + unsigned varint followed by zlib; constant-block packing; and a deterministic adaptive block codec. Every method pays the identical 9-byte header (magic 2, version 1, method id 2, count 4); method-specific framing is additional. A method receives no fixture labels or family name and cannot use a target or training labels. This is an enumerated search over known compositions, not a novelty claim.

The primary metric is bytes/sample including all framing. Every output must round-trip exactly to the original signed-int64 stream. Encode and decode latency are descriptive CPU measurements (warmup, repeated runs, median/IQR), not claims of hardware-independent speed. Validation may select at most one chain; selection cannot inspect final data. The selected chain must be lossless, stable across all three validation seeds, and reduce mean validation bytes/sample by at least 1% against the strongest known validation baseline (raw, zlib, lzma, delta-1/2+zigzag-varint+zlib, block packing, or adaptive block codec). Otherwise retain raw signed-64 as the operational fallback. The frozen winner must also clear the same 1% mean threshold on final data.

This is an engineering/composition benchmark, not a novelty claim. It can show a useful regime-dependent compression chain or a negative result; it does not establish a universal telemetry law. No final values or result numbers are recorded in this document.


## Completed result — 2026-11-16

The registry enumerated 115 type-compatible known chains, plus two separately specified block heuristics. All 702 development/validation round-trips passed. Validation selected `for.lzma` on each of the three seeds: 1.4273 bytes/sample versus 1.5011 for the strongest predeclared control, `delta.block_for.zlib` (4.9% reduction).

After source/validation hashes were frozen and reviewed, the selected chain ran once on the three shifted final streams. It used **1.9680 bytes/sample versus 2.5982 for the frozen control** (24.25% reduction). All round-trips remained exact. It also used fewer bytes than every evaluated final control; the best of those *descriptively selected after evaluation* was `delta.zigzag.varint.zlib` at 2.1312 bytes/sample. That latter comparison is not the preregistered primary comparison.

Encoding was approximately 2.0 ms and decoding 1.3 ms for 4096 values on this machine, using medians of three timed repeats per stream. The frozen control encoded in about 1.6 ms and decoded in 1.3 ms. The candidate is not faster at every operation; these timings are descriptive and implementation-specific.

This is a useful composition of established packing and LZMA, **not a novel compression algorithm**, not a published-codec benchmark, and not evidence that the Alien learned scorer has been repaired. The atoms were manually grounded in primary specifications; the search was bounded deterministic enumeration, not LLM decoding or highest-beta sampling. Three synthetic mixed-stream seeds provide a small utility pilot, not broad task generalization. No macro-atom was promoted.

Protocol deviations: the plan listed `delta.zigzag.varint.lzma`; it was evaluated in the search but omitted from the designated baseline list. Its validation size (1.5432 bytes/sample) was worse than the selected control, so adding it would not change selection. Timing dispersion is min/max rather than IQR. The existing pre-final audit records these details. Failed or positive labels should not be generalized beyond this scope.

Re-run in a fresh checkout/output copy: `python experiment.py validation`, inspect it, then `python experiment.py freeze` and `python experiment.py final`. Existing result files refuse overwrite. Fixture generation needs the Python standard library; the operator engine imports no ML models. No paid calls were used.
