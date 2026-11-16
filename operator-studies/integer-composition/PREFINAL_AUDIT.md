# Pre-final audit

Date: 2026-11-16. This is a read-only audit of the frozen validation artifacts. It did not call the final fixture. The cheap guard check confirmed that `load("final", 19311)` raises `PermissionError` unless the caller explicitly passes `allow_final=True`; no final stream was generated or read.

## Findings

- **No blocking leakage found.** Validation uses only the development and validation seeds, and each method is given the same raw stream. The fixture API returns only values, not family names or labels. Selection is computed from validation rows; final mode is only reached through the explicit final branch after reading a freeze record.
- **Framing is counted consistently.** `HEADER = <2sBHI` is 9 bytes (magic 2, version 1, method id 2, count 4), and `measure()` uses `len(payload)` including that header. Block codecs put their count, per-block selector, block length, and payload inside the measured payload. The decoder checks magic/version/method/count, rejects malformed/trailing block bytes, and every one of the 702 validation rows reports `lossless: true`.
- **Decoder/round-trip audit passes.** The recorded validation run has 702 rows (117 catalog methods × 6 streams), zero errors, and all rows lossless. The selected `for.lzma` is stable on all three validation seeds. Its recorded mean is 1.4273274739583333 bytes/sample versus 1.5011393229166667 for `delta.block_for.zlib`, a ratio of 0.9508294481188333 (about 4.9% lower).
- **Source provenance matches.** Recomputed SHA-256 values for `experiment.py`, `fixture.py`, `plan.json`, `generation/operator_composition.py`, `generation/integer_codecs.py`, and `generation/molecules.py` all match `validation.json`.
- **Known-method scope is explicit.** The catalog has 115 enumerated composition chains plus the two known block heuristics. This is a search over known compositions, not a novel-algorithm claim. The three seeds are mixed-composition streams, not four separate task-family replications; family-level diagnostics, if shown, are secondary.

## Protocol deviations / wording notes

`plan.json` lists `delta.zigzag.varint.lzma` as a candidate and the validation catalog contains and evaluates it (its recorded mean is worse than the selected control), but `BASELINE_CHAINS` intentionally omits it from the named baseline-control set. This is a preregistered baseline-set deviation worth reporting; it does not alter the selected candidate or result because the omitted chain was evaluated as search and did not win. Do not silently relabel it as a control in the final report.

The timing fields are medians over three repeats plus min/max ranges. They are not IQRs; describe them as median-and-range timing, not median/IQR.

No blocking issue was found for final-read approval, subject to parent review of the frozen `freeze.json` and the ordinary final-run write-once guard. This audit intentionally contains no final values.

