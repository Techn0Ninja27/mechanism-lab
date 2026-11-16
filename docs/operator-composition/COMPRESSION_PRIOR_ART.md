# Prior art: lossless signed-integer telemetry compression

Date checked: 2026-11-16. Scope: lossless streams of signed 32/64-bit telemetry values. This is a prior-art boundary map for the operator-composition experiment, not a claim of novelty or a benchmark result.

## Bottom line

The individual atoms needed by a typed composition are all established:

`predict/delta -> signed-to-unsigned (ZigZag) -> variable-byte or bit-pack -> RLE`

and, for regular clocks, `timestamp delta-of-delta` is established as well. Block-local selection or interleaving of RLE and bit-packed runs is explicitly standard in Parquet; online predictor choice and block bit widths are explicit in Sprintz. Therefore, “dynamic switching between simple codecs” is not a novelty claim. Any new claim must be narrower (for example, a particular typed atom grammar, overhead accounting, or transfer protocol) and must beat strong fixed/composed baselines under untouched distributions.

## Source-defined operator units

### 1. Signed mapping and varints (Protobuf; specification)

For a signed integer `n`, ZigZag maps small magnitudes to small unsigned integers: nonnegative `p -> 2p`, negative `n -> 2|n|-1`; the 32-bit implementation formula is `(n << 1) ^ (n >> 31)` (and analogously for 64-bit). The result is then encoded as an unsigned base-128 varint: seven payload bits per byte and one continuation bit. Unsigned 64-bit values take 1–10 bytes; a negative two's-complement `int64` without ZigZag takes 10 bytes, so signed telemetry residuals should not be sent as plain `intN` varints.

Operator atoms:

* `zigzag32/64`: signed residual -> unsigned magnitude-order code.
* `uleb128`: unsigned integer -> 7-bit continuation-byte stream.
* `packed-varints`: omit per-element field tags for a homogeneous sequence (a serialization framing optimization, not a compression guarantee).

Assumptions/limits: fixed integer width and agreed endianness/overflow semantics are required. Varint size is byte-quantized; it can lose to bit packing when residuals fit a narrow but non-byte-aligned width. Protobuf is a wire-format specification, not a telemetry block format; framing, count, and corruption recovery remain the caller's responsibility.

### 2. Delta binary packing and RLE/bit-pack hybrid (Apache Parquet; specification)

Parquet's `DELTA_BINARY_PACKED` supports INT32/INT64. It stores header values with ULEB128 (signed header values use ZigZag), then groups delta values into blocks, subtracts each block's minimum delta (a frame of reference), and bit-packs miniblocks at their required widths. The current format specifies a block size that is a multiple of 128 and miniblocks whose value count is a multiple of 32. This is a concrete block operator: compute first-order deltas, find a block-local frame and sub-block widths, pack the nonnegative residuals, and retain enough metadata to reconstruct.

Parquet's RLE/bit-packing hybrid uses a run header whose low bit distinguishes an RLE run from a bit-packed run. RLE stores one repeated value at a fixed byte-rounded width; bit-packed runs store groups of eight values at a known bit width. Implementations may interleave the two run types. Run lengths are constrained to `[1, 2^31-1]` in the current spec; bit-packed runs are multiples of eight values. The encoding is formally used for booleans, dictionary indices, and repetition/definition levels, but the mechanism is a direct baseline for residual integers.

Operator atoms:

* `delta1`: `x[i] - x[i-1]`, with a seed.
* `block_bitpack(w)`: choose a width from a bounded block and pack each integer in `w` bits.
* `frame_of_reference(min)`: subtract a block-local minimum before packing (the Parquet delta variant).
* `rle(value, count)`: repeated-value run.
* `hybrid_rle_bitpack`: switch between the previous two using explicit run headers.

Assumptions/limits: block width is controlled by the largest magnitude in the block, so one outlier inflates all peers unless exceptions or smaller blocks are added. Headers and padding matter for short blocks. This is strong prior art against any claim that “typed atoms + an adaptive RLE/bit-pack selector” is itself new.

### 3. Delta-of-delta timestamps and stateful residual coding (Gorilla; paper)

Gorilla partitions timestamp/value pairs into time blocks. It stores an initial timestamp and first timestamp delta, then encodes the change in timestamp delta (delta-of-delta). For regular sampling, the delta-of-delta is zero and has a one-bit representation; small signed deviations use short prefixed codes. For values, Gorilla is floating-point-specific: XOR the current 64-bit representation with the previous value, then encode leading/trailing-zero windows and the meaningful bits, reusing the previous window when possible.

Relevant integer-telemetry lesson: `delta2` is a stateful predictor whose win depends on near-constant increments; its state/seed and block reset are part of the format. The value-XOR operator is not a signed-integer residual codec and should not be silently counted as one. Gorilla's own paper says it keeps full resolution and compresses each series separately; it is not evidence that arbitrary channels should be mixed.

Operator atom:

* `delta2`: `(x[i]-x[i-1]) - (x[i-1]-x[i-2])`, with two seeds and a signed small-code layer.

Known overlap: delta-of-delta plus small-code prefixes is established; only a measured, typed composition around integer residuals could be new.

### 4. Online forecasting + block packing (Sprintz; paper and Apache-2.0 reference code)

Sprintz targets multivariate integer time series. Per sample it predicts from the prior sample and forecaster state, computes an error, updates the forecaster, then for each column chooses the maximum required error bit width over a block of eight samples and bit-packs that column. All-zero error blocks are run-length encoded; the paper also describes optional Huffman coding of headers/payload. The default/simple forecaster is delta; FIRE is an online learned predictor that can outperform delta on some series. The paper explicitly reports poor behavior on sparse/discrete AMPD gas/water data, a useful warning against only testing smooth traces.

Operator atoms:

* `predict_delta` and `predict_FIRE`: stateful predictor -> signed error.
* `column_block_width`: maximum required error bits over a block.
* `block_bitpack`: fixed-width residual packing.
* `zero_block_rle`: RLE of blocks whose residual widths are all zero.

Assumptions/limits: multivariate layout and block alignment are important; predictor state creates serial dependence. Results are not a license to claim general superiority on univariate or randomly ordered streams. The authors' public repository is marked Apache-2.0; the ACM paper itself remains publisher-copyrighted.

### 5. Control/data separated byte varints (Stream VByte; paper)

Stream VByte groups four unsigned integers. A control byte records each integer's byte length; all control bytes are stored in a separate stream from payload bytes. A 256-entry table maps control bytes to payload lengths/shuffle masks. Differential coding may be applied before the format, but the codec itself is byte-oriented and unsigned. The paper reports speed advantages from SIMD and explicitly notes a storage cost of roughly 0.5–2 extra bits/int versus VByte in its tests; this is a speed baseline, not a minimum-size baseline.

Operator atoms:

* `byte_length_control`: choose 1–4 bytes per 32-bit integer and emit a control stream.
* `stream_vbyte_payload`: little-endian byte payload stream.
* optional `delta1` before unsigned encoding.

Assumptions/limits: the published speed path relies on SIMD; a stdlib-only implementation should be treated as a scalar format baseline, not as a reproduction of the throughput claim. For signed values, ZigZag (or a nonnegative residual transform) is needed first.

## What counts as a strong smallest baseline

Implement reproducible, dependency-light baselines in Python stdlib (and NumPy only for array generation/measurement), with explicit seeds and length/count framing:

1. `raw_le32`/`raw_le64` fixed-width little-endian values.
2. `zigzag + ULEB128` per value.
3. `delta1 + zigzag + ULEB128` (streaming, one seed).
4. `delta2 + zigzag + ULEB128` (two seeds).
5. `delta1 + block_bitpack`, with block sizes 8, 32, and 128 and a width header per block.
6. `hybrid_rle_bitpack` over residuals, with an explicit deterministic run decision and header cost.
7. A Sprintz-shaped baseline: delta predictor, 8-sample block width, zero-block RLE; this isolates Sprintz's structural advantage without reimplementing FIRE/Huffman.
8. Optional format baseline: scalar Stream-VByte-shaped control/payload after ZigZag; report bytes and decode cost separately.

All decoders must round-trip adversarial extremes (`INT_MIN`, `INT_MAX`, alternating signs, empty and one-element streams) and reject truncation/overflow. Report total bytes including seeds, count, headers, block widths, and padding; “payload bits only” is not an honest format comparison.

## Meaningful development and transfer regimes

Use separate generated streams (fixed seeds) for development and untouched transfer. At minimum:

* sequential: constant/near-constant increments, ramps, bounded random walks;
* bursty: long constant/error-zero periods with short high-amplitude bursts;
* heavy-tail: residual magnitudes from a signed heavy-tailed distribution, including occasional extremes;
* random: iid full-range and iid narrow-range signed integers.

Vary length, block phase, bit width (8/16/32/64), and sign balance. Train or choose any block policy only on development streams. On transfer, freeze the policy and report bytes/value, encode/decode time, and exact round-trip. Include channel-wise and interleaved multivariate layouts because Sprintz's advantage depends on column structure. Do not infer success from a smooth toy ramp: sequential data is the easiest case for delta coding and does not test robustness.

## Combination-overlap ledger

Already directly established: delta, delta-of-delta, ZigZag, ULEB128/varints, block bit-packing, RLE, adaptive RLE-vs-bit-pack runs, online prediction, and optional predictor/codec switching. A composition that merely chains these operators is prior art in mechanism, even if its implementation is new.

Potentially nontrivial but unproven: a precisely typed atom interface with bounded state, a selector whose metadata is included in the cost, or a transfer-safe rule that chooses among these atoms without peeking at held-out data. These are experiment questions, not novelty claims. The prior-art search has not established that any such rule is unpublished; it only establishes that the component mechanisms and adaptive block selection are old.

## Source and license cautions

The sources below are primary specifications/papers. Parquet's format repository and Sprintz reference repository are Apache-2.0 according to their repository metadata. The Protobuf page is Google's official documentation; Gorilla, Sprintz, and Stream VByte papers are scholarly publications with publisher terms, not blanket permission to copy code. Reimplement algorithms from the definitions and keep any copied implementation separate with its license.

