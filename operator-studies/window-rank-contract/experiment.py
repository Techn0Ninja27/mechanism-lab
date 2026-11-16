"""CPU-only contract checks for the FIER-to-WindowQuant transfer.

This is a numerical and accounting diagnostic.  It is not a VLM result and
does not measure usefulness, novelty, or a research improvement.
"""
from __future__ import annotations

import hashlib
import json
import platform
import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "results.json"
SEEDS = (4811, 4813, 4817)


def _finite(name, x):
    if not np.isfinite(x).all():
        raise AssertionError(f"{name} contains non-finite values")


def stable_softmax(x):
    x = np.asarray(x, dtype=np.float64)
    _finite("softmax input", x)
    y = x - np.max(x, axis=-1, keepdims=True)
    out = np.exp(y)
    out /= np.sum(out, axis=-1, keepdims=True)
    _finite("softmax output", out)
    return out


def rank_fill(scores, counts, *, first_fp16=True):
    """Fill windows by score while preserving exact precision counts."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("scores must be a finite vector")
    n = scores.size
    if set(counts) != {"fp16", "int4", "int2"} or any(
        isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in counts.values()
    ) or sum(counts.values()) != n:
        raise ValueError("precision counts must be nonnegative integers summing to windows")
    if first_fp16 and (n == 0 or counts["fp16"] < 1):
        raise ValueError("mandatory first window requires an FP16 slot")
    order = sorted(range(n), key=lambda i: (-scores[i], i))
    assignment = [None] * n
    remaining = dict(counts)
    if first_fp16:
        assignment[0] = "fp16"
        remaining["fp16"] -= 1
        order.remove(0)
    slots = ["fp16"] * remaining["fp16"] + ["int4"] * remaining["int4"] + ["int2"] * remaining["int2"]
    for i, precision in zip(order, slots):
        assignment[i] = precision
    if any(x is None for x in assignment) or {p: assignment.count(p) for p in counts} != counts:
        raise AssertionError("rank fill violated precision counts")
    return assignment


def attention(q, key, value, mask=None):
    if q.ndim == 1:
        logits = np.einsum("d,kd->k", q, key) / np.sqrt(q.shape[-1])
    else:
        logits = np.einsum("qd,kd->qk", q, key) / np.sqrt(q.shape[-1])
    if mask is not None:
        logits = logits + np.asarray(mask, dtype=np.float64)
    weights = stable_softmax(logits)
    if q.ndim == 1:
        return np.einsum("k,kv->v", weights, value), weights
    return np.einsum("qk,kv->qv", weights, value), weights


def grouped_binary(key, group=4):
    if key.shape[0] % group:
        raise ValueError("sequence length must be divisible by group")
    grouped = key.reshape(key.shape[0] // group, group, key.shape[1])
    lo = np.min(grouped, axis=1, keepdims=True)
    hi = np.max(grouped, axis=1, keepdims=True)
    scale = np.maximum(hi - lo, 1e-12)
    levels = np.rint((grouped - lo) / scale).clip(0, 1)
    return (lo + levels * scale).reshape(key.shape), (lo, hi)


def grouped_two_bit(key, group=4):
    if key.shape[0] % group:
        raise ValueError("sequence length must be divisible by group")
    grouped = key.reshape(key.shape[0] // group, group, key.shape[1])
    lo = np.min(grouped, axis=1, keepdims=True)
    hi = np.max(grouped, axis=1, keepdims=True)
    scale = np.maximum(hi - lo, 1e-12)
    levels = np.rint(3.0 * (grouped - lo) / scale).clip(0, 3)
    return (lo + levels * scale / 3.0).reshape(key.shape), (lo, hi)


def window_score(query, key, mode, windows, window_tokens):
    if mode == "full":
        key_used = key
    elif mode == "1bit":
        key_used, _ = grouped_binary(key)
    elif mode == "2bit":
        key_used, _ = grouped_two_bit(key)
    else:
        raise ValueError(mode)
    raw = np.einsum("qd,kd->qk", query, key_used)
    raw_means = np.array([raw[:, i * window_tokens:(i + 1) * window_tokens].mean() for i in range(windows)])
    # This is intentionally a different observable: mean attention probability,
    # not mean raw logits.  Neither is called an upper bound.
    probabilities = stable_softmax(raw / np.sqrt(query.shape[-1]))
    prob_means = np.array([probabilities[:, i * window_tokens:(i + 1) * window_tokens].mean() for i in range(windows)])
    return raw_means, prob_means


def accounting(assignment, *, tokens, dim, group=4, perm_bytes=4):
    bits = {"fp16": 16, "int4": 4, "int2": 2}
    payload = sum(tokens * dim * 2 * bits[p] / 8 for p in assignment)
    quantized_windows = sum(p != "fp16" for p in assignment)
    groups_per_window = int(np.ceil(tokens / group)) * dim
    # Per quantized K/V group: float32 min and max (two values), for both K and V.
    metadata = quantized_windows * groups_per_window * 2 * 2 * 4
    permutation = len(assignment) * perm_bytes
    return {"payload_bytes": payload, "quantizer_metadata_bytes": metadata, "permutation_bytes": permutation,
            "persistent_bytes": payload + metadata + permutation}


def run_case(seed):
    rng = np.random.default_rng(seed)
    windows, window_tokens, dim, queries = 8, 4, 8, 3
    key = rng.normal(size=(windows * window_tokens, dim))
    value = rng.normal(size=(windows * window_tokens, dim))
    query = rng.normal(size=(queries, dim))
    _finite("key", key); _finite("value", value); _finite("query", query)
    counts = {"fp16": 2, "int4": 3, "int2": 3}
    scores = window_score(query, key, "1bit", windows, window_tokens)[0]
    assignment = rank_fill(scores, counts)
    assert assignment[0] == "fp16"
    baseline = accounting(assignment, tokens=window_tokens, dim=dim)
    candidate = accounting(rank_fill(scores[::-1], counts), tokens=window_tokens, dim=dim)
    assert baseline == candidate, "equal precision counts must have equal accounting"

    window_order = [i for p in ("fp16", "int4", "int2") for i, x in enumerate(assignment) if x == p]
    permutation = np.array([window * window_tokens + offset for window in window_order for offset in range(window_tokens)])
    restored = np.argsort(permutation)
    mask = np.linspace(-1.3, 0.7, key.shape[0], dtype=np.float64)
    original_out, _ = attention(query[0], key, value, mask)
    reordered_out, _ = attention(query[0], key[permutation], value[permutation], mask[permutation])
    restored_out = reordered_out  # attention is permutation invariant; values are already summed
    reorder_error = float(np.max(np.abs(original_out - restored_out)))
    if reorder_error > 1e-12:
        raise AssertionError("K/V permutation changed attention output")
    wrong_mask_out, _ = attention(query[0], key[permutation], value[permutation], mask)
    wrong_mask_error = float(np.max(np.abs(original_out - wrong_mask_out)))
    if wrong_mask_error <= 1e-8:
        raise AssertionError("nonuniform mask did not expose an unpermuted-mask error")

    raw1, prob1 = window_score(query, key, "1bit", windows, window_tokens)
    raw2, prob2 = window_score(query, key, "2bit", windows, window_tokens)
    rawf, probf = window_score(query, key, "full", windows, window_tokens)
    for name, arr in (("raw1", raw1), ("prob1", prob1), ("raw2", raw2), ("prob2", prob2), ("rawf", rawf), ("probf", probf)):
        _finite(name, arr)
    transient = {"query_array_bytes": int(query.nbytes),
                 "raw_score_matrix_bytes": int(queries * windows * window_tokens * 8),
                 "softmax_matrix_bytes": int(queries * windows * window_tokens * 8),
                 "temporary_dequantized_key_array_bytes_lower_bound": int(key.nbytes),
                 "accounting_scope": "explicit arrays only; not a peak-memory estimate"}
    return {"seed": seed, "windows": windows, "window_tokens": window_tokens, "dim": dim,
            "counts": counts, "assignment": assignment, "scores": raw1.tolist(),
            "raw_logit_means": {"1bit": raw1.tolist(), "2bit": raw2.tolist(), "full": rawf.tolist()},
            "softmax_probability_means": {"1bit": prob1.tolist(), "2bit": prob2.tolist(), "full": probf.tolist()},
            "accounting": baseline, "transient_scoring": transient,
            "reorder_max_abs_error": reorder_error, "wrong_unpermuted_mask_error": wrong_mask_error,
            "interpretation": "Contract-only synthetic diagnostic; score differences do not establish usefulness or superiority."}


def invalid_input_checks():
    checks = {}
    for name, fn in {
        "nan_scores": lambda: rank_fill([0.0, np.nan], {"fp16": 1, "int4": 1, "int2": 0}),
        "boolean_count": lambda: rank_fill([0.0], {"fp16": True, "int4": 0, "int2": 0}),
        "missing_first_fp16": lambda: rank_fill([0.0, 1.0], {"fp16": 0, "int4": 1, "int2": 1}),
    }.items():
        try:
            fn()
        except (ValueError, TypeError):
            checks[name] = True
        else:
            checks[name] = False
    if not all(checks.values()):
        raise AssertionError(f"invalid input contract failed: {checks}")
    return checks


def quantizer_contract_checks():
    known = np.array([[-3.0], [-1.0], [1.0], [3.0]])
    two, _ = grouped_two_bit(known, group=4)
    if not np.allclose(two, known, atol=0.0, rtol=0.0):
        raise AssertionError("2-bit affine surrogate did not reproduce four-level endpoints")
    one_input = np.array([[-1.0], [0.0], [0.0], [1.0]])
    one, _ = grouped_binary(one_input, group=4)
    if not np.isfinite(one).all() or one[1, 0] != one[2, 0]:
        raise AssertionError("1-bit affine zero handling is not deterministic")
    return {"two_bit_four_level_known_vector": True, "one_bit_zero_deterministic": True}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args(argv)
    output = args.output
    if output.exists():
        raise FileExistsError("preserve prior result")
    checks = invalid_input_checks()
    quantizer_checks = quantizer_contract_checks()
    rows = [run_case(seed) for seed in SEEDS]
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    paper = ROOT / "results/interface-proposal-v1/windowquant-method.txt"
    recorded_paper_hash = "5ca509416e31af4bbd96d6971fe319f20bb1d96b7681f11e64fe405baa875e40"
    paper_present = paper.exists()
    if paper_present and hashlib.sha256(paper.read_bytes()).hexdigest() != recorded_paper_hash:
        raise AssertionError("local WindowQuant extract does not match recorded hash")
    result = {"schema": "window-rank-contract.v2", "seeds": list(SEEDS), "rows": rows,
              "invalid_input_checks": checks,
              "quantizer_contract_checks": quantizer_checks,
              "python": platform.python_version(), "numpy": np.__version__,
              "source_hashes": {"experiment.py": script_hash, "windowquant_method.txt": recorded_paper_hash},
              "paper_extract_verified_if_present": paper_present,
              "scope": "Synthetic contract checks only; no model, task, latency, novelty, or usefulness claim.",
              "quantization_surrogate": "Generic affine min/max 2**bits-level surrogate; not the FIER implementation."}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"cases": len(rows), "output": str(output), "script_sha256": script_hash}, indent=2))


if __name__ == "__main__":
    main()
