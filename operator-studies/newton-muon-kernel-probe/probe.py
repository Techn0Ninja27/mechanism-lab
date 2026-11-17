"""Capability-only MPS linear-algebra probe; no training or performance claim."""
from __future__ import annotations

import argparse
import json
import os
import platform
import warnings
from pathlib import Path

import torch


SEED = 73031
N = 256
DTYPE = torch.float32


def _error(exc: BaseException) -> dict[str, str]:
    return {"type": type(exc).__name__, "message": str(exc)[:500]}


def _run(name: str, fn, residual_fn) -> dict:
    row = {"name": name, "status": "unsupported", "fallback_disabled": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "0"}
    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            value = fn()
        warning_text = [str(item.message)[:500] for item in captured]
        row["warnings"] = warning_text
        if any("falling back to CPU" in message or "fallback to CPU" in message for message in warning_text):
            row.update(status="internal_cpu_fallback", error={"type": "InternalFallbackWarning", "message": "MPS operation warned that it fell back to CPU"})
            return row
        if value is not None:
            torch.mps.synchronize()
        residual = residual_fn(value) if residual_fn is not None else None
        row.update(status="available", residuals=residual or {})
    except Exception as exc:  # Capability probing must preserve each failure.
        row["error"] = _error(exc)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        raise SystemExit("Run with PYTORCH_ENABLE_MPS_FALLBACK=0; implicit CPU fallback is disallowed")
    if not torch.backends.mps.is_available():
        result = {"schema": "newton-muon-kernel-probe.v1", "status": "mps_unavailable", "passed": False}
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result)); return

    torch.manual_seed(SEED)
    base = torch.randn((N, N), dtype=DTYPE)
    cpu_a = base @ base.T + 0.25 * torch.eye(N, dtype=DTYPE)
    cpu_eye = torch.eye(N, dtype=DTYPE)
    device = torch.device("mps")
    a = cpu_a.to(device)
    b = torch.randn((N, N), dtype=DTYPE, device=device)

    def matmul_residual(value):
        expected = cpu_a @ b.detach().cpu()
        return {"max_abs_vs_cpu": float((value.detach().cpu() - expected).abs().max())}

    def chol_residual(value):
        lower, info = value
        l = lower.detach().cpu()
        return {"info_max_abs": float(info.detach().cpu().abs().max()), "factorization_max_abs": float((l @ l.T - cpu_a).abs().max())}

    def chol_inv_residual(value):
        inv = value.detach().cpu()
        return {"identity_max_abs": float((cpu_a @ inv - cpu_eye).abs().max()), "symmetry_max_abs": float((inv - inv.T).abs().max())}

    def eigh_residual(value):
        eigenvalues, eigenvectors = value
        w, q = eigenvalues.detach().cpu(), eigenvectors.detach().cpu()
        return {"reconstruction_max_abs": float((q @ torch.diag(w) @ q.T - cpu_a).abs().max()), "orthogonality_max_abs": float((q.T @ q - cpu_eye).abs().max())}

    def qr_residual(value):
        q, r = value
        q, r = q.detach().cpu(), r.detach().cpu()
        return {"reconstruction_max_abs": float((q @ r - cpu_a).abs().max()), "orthogonality_max_abs": float((q.T @ q - cpu_eye).abs().max())}

    rows = [
        _run("synchronize", lambda: (torch.mps.synchronize() or True), None),
        _run("dense_matmul", lambda: a @ b, matmul_residual),
        _run("cholesky_ex", lambda: torch.linalg.cholesky_ex(a, check_errors=False), chol_residual),
        _run("cholesky_inverse", lambda: torch.cholesky_inverse(torch.linalg.cholesky(a)), chol_inv_residual),
        _run("eigh", lambda: torch.linalg.eigh(a), eigh_residual),
        _run("qr", lambda: torch.linalg.qr(a), qr_residual),
    ]
    result = {
        "schema": "newton-muon-kernel-probe.v1",
        "status": "completed",
        "scope": "MPS capability and numerical residuals only; no timing, optimizer, training, utility or novelty claim.",
        "seed": SEED, "matrix_size": N, "dtype": str(DTYPE), "device": str(device),
        "torch_version": torch.__version__, "platform": platform.platform(),
        "fallback_disabled": True, "operations": rows,
        "passed": all(row["status"] == "available" for row in rows if row["name"] in {"synchronize", "dense_matmul"}),
        "interpretation": "Unsupported MPS kernels are capability findings; CPU residuals verify outputs only where an operation executes. This probe does not justify choosing an optimizer implementation.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "available": [r["name"] for r in rows if r["status"] == "available"], "unsupported": [r["name"] for r in rows if r["status"] != "available"]}))


if __name__ == "__main__":
    main()
