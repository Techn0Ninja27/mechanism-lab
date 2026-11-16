"""CPU-only contract checks for the FIER-to-WindowQuant transfer.

This is a numerical and accounting diagnostic.  It is not a VLM result and
does not measure usefulness, novelty, or a research improvement.
"""
from __future__ import annotations

import hashlib
import json
import platform
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


def attention(q, key, value):
    if q.ndim == 1:
        logits = np.einsum("d,kd->k", q, key) / np.sqrt(q.shape[-1])
    else:
        logits = np.einsum("qd,kd->qk", q, key) / np.sqrt(q.shape[-1])
    weights = stable_softmax(logits)
    if q.ndim == 1:
        return np.einsum("k,kv->v", weights, value), weights
    return np.einsum("qk,kv->qv", weights, value), weights


def grouped_binary(key, group=4):
    if key.shape[0] % group:
        raise ValueError("sequence length must be divisible by group")
    grouped = key.reshape(key.shape[0] // group, group, key.shape[1])
    scale = np.mean(np.abs(grouped), axis=1, keepdims=True)
    scale = np.maximum(scale, 1e-12)
    out = np.sign(grouped) * scale
    return out.reshape(key.shape), scale


def grouped_two_bit(key, group=4):
    if key.shape[0] % group:
        raise ValueError("sequence length must be divisible by group")
    grouped = key.reshape(key.shape[0] // group, group, key.shape[1])
    scale = np.maximum(np.max(np.abs(grouped), axis=1, keepdims=True) / 3.0, 1e-12)
    levels = np.clip(np.rint(grouped / scale), -3, 3)
    return (levels * scale).reshape(key.shape), scale


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
    groups = sum(int(np.ceil(tokens * dim / group)) for _ in assignment)
    # two float32 metadata values (scale/offset) per group, for both K and V
    metadata = groups * 2 * 4
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
    original_out, _ = attention(query[0], key, value)
    reordered_out, _ = attention(query[0], key[permutation], value[permutation])
    restored_out = reordered_out  # attention is permutation invariant; values are already summed
    reorder_error = float(np.max(np.abs(original_out - restored_out)))
    if reorder_error > 1e-12:
        raise AssertionError("K/V permutation changed attention output")

    raw1, prob1 = window_score(query, key, "1bit", windows, window_tokens)
    raw2, prob2 = window_score(query, key, "2bit", windows, window_tokens)
    rawf, probf = window_score(query, key, "full", windows, window_tokens)
    for name, arr in (("raw1", raw1), ("prob1", prob1), ("raw2", raw2), ("prob2", prob2), ("rawf", rawf), ("probf", probf)):
        _finite(name, arr)
    transient = {"query_bytes": int(query.nbytes), "score_bytes": int(raw1.nbytes + prob1.nbytes),
                 "temporary_quantized_key_bytes": int(key.nbytes),
                 "peak_scoring_bytes": int(query.nbytes + raw1.nbytes + prob1.nbytes + key.nbytes)}
    return {"seed": seed, "windows": windows, "window_tokens": window_tokens, "dim": dim,
            "counts": counts, "assignment": assignment, "scores": raw1.tolist(),
            "raw_logit_means": {"1bit": raw1.tolist(), "2bit": raw2.tolist(), "full": rawf.tolist()},
            "softmax_probability_means": {"1bit": prob1.tolist(), "2bit": prob2.tolist(), "full": probf.tolist()},
            "accounting": baseline, "transient_scoring": transient,
            "reorder_max_abs_error": reorder_error,
            "interpretation": "Contract-only synthetic diagnostic; score differences do not establish usefulness or superiority."}


def main():
    if OUT.exists():
        raise FileExistsError("preserve prior result")
    rows = [run_case(seed) for seed in SEEDS]
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    paper = ROOT / "results/interface-proposal-v1/windowquant-method.txt"
    result = {"schema": "window-rank-contract.v1", "seeds": list(SEEDS), "rows": rows,
              "python": platform.python_version(), "numpy": np.__version__,
              "source_hashes": {"experiment.py": script_hash, "windowquant_method.txt": hashlib.sha256(paper.read_bytes()).hexdigest()},
              "scope": "Synthetic contract checks only; no model, task, latency, novelty, or usefulness claim."}
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"cases": len(rows), "output": str(OUT), "script_sha256": script_hash}, indent=2))


if __name__ == "__main__":
    main()
