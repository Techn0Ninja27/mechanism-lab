"""Run the reproducible PERA algebra and tiny synthetic matrix diagnostic."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import torch

from pera_algebra import delta_expanded, delta_sum, nonlinear_component, expanded_width, parameter_count, rank_bound, pair_indices

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/contract-eval-packet-v1/sources/2604.11841v1.selected.txt"
DEFAULT_PROTOCOL = Path(__file__).with_name("protocol.json")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def arm(seed: int, rank: int, protocol: dict) -> dict:
    torch.manual_seed(seed)
    m, n = protocol["algebra"]["m"], protocol["algebra"]["n"]
    B = torch.randn(m, rank, dtype=torch.float64)
    A = torch.randn(rank, n, dtype=torch.float64)
    h = torch.randn(rank * (rank + 1) // 2, dtype=torch.float64)
    expanded = delta_expanded(B, A, h)
    summed = delta_sum(B, A, h)
    return {
        "seed": seed, "rank": rank, "m": m, "n": n,
        "expanded_width": expanded_width(rank),
        "parameter_count": parameter_count(m, n, rank),
        "max_factorization_error": float((expanded - summed).abs().max()),
        "delta_rank": int(torch.linalg.matrix_rank(expanded)),
        "rank_bound": rank_bound(m, n, rank),
        "attains_bound": bool(int(torch.linalg.matrix_rank(expanded)) == rank_bound(m, n, rank)),
        "special_identity_max_error": special_identity_error(B, A, protocol["algebra"]["special_lambda"]),
        "gradient_equivalence_max_error": gradient_equivalence_error(B, A, h),
    }


def special_identity_error(B: torch.Tensor, A: torch.Tensor, lam: float) -> float:
    rank = B.shape[1]
    coeffs = torch.tensor([lam if i == j else 2 * lam for i, j in pair_indices(rank)], dtype=B.dtype)
    expected = B @ A + lam * (B @ A).square()
    return float((delta_expanded(B, A, coeffs) - expected).abs().max())


def gradient_equivalence_error(B: torch.Tensor, A: torch.Tensor, h: torch.Tensor) -> float:
    B = B.detach().requires_grad_(); A = A.detach().requires_grad_(); h = h.detach().requires_grad_()
    left = torch.autograd.grad(delta_expanded(B, A, h).square().sum(), (B, A, h), retain_graph=True)
    right = torch.autograd.grad(delta_sum(B, A, h).square().sum(), (B, A, h))
    return max(float((x - y).abs().max()) for x, y in zip(left, right))


def training(seed: int, protocol: dict) -> dict:
    torch.manual_seed(seed)
    m, n = protocol["algebra"]["m"], protocol["algebra"]["n"]
    rank = protocol["training"]["rank"]
    target = torch.randn(m, n, dtype=torch.float64)
    B = torch.randn(m, rank, dtype=torch.float64, requires_grad=True)
    A = torch.zeros(rank, n, dtype=torch.float64, requires_grad=True)
    h = torch.zeros(rank * (rank + 1) // 2, dtype=torch.float64, requires_grad=True)
    rows = []
    updates = protocol["training"]["updates"]
    lr = protocol["training"]["learning_rate"]
    for step in range(updates + 1):
        delta = delta_expanded(B, A, h)
        loss = (delta - target).square().mean()
        gB, gA, gh = torch.autograd.grad(loss, (B, A, h))
        rows.append({"step": step, "loss": float(loss.detach()), "grad_b_norm": float(gB.norm().detach()),
                     "grad_a_norm": float(gA.norm().detach()), "grad_h_norm": float(gh.norm().detach()),
                     "nonlinear_norm": float(nonlinear_component(B, A, h).norm().detach())})
        if step < updates:
            with torch.no_grad():
                B -= lr * gB
                A -= lr * gA
                h -= lr * gh
    return {"seed": seed, "steps": rows, "learning_rate": lr, "updates": updates, "objective": "synthetic matrix MSE"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    if protocol["source"]["sha256"] != sha(SOURCE):
        raise ValueError("source hash differs from frozen protocol")
    if platform.python_version() != protocol["environment"]["python"] or torch.__version__ != protocol["environment"]["torch"]:
        raise RuntimeError("pinned Python/Torch environment does not match protocol")
    if args.output.exists():
        raise FileExistsError("refusing to overwrite existing diagnostic output")
    args.output.mkdir(parents=True, exist_ok=True)
    protocol_sha = sha(args.protocol)
    code_path = Path(__file__).resolve()
    result = {
        "schema": "pera-algebra-diagnostic.v2", "status": "completed",
        "source": {"path": str(SOURCE), "sha256": sha(SOURCE)},
        "protocol": {"path": str(args.protocol.resolve()), "sha256": protocol_sha},
        "code": {"path": str(code_path), "sha256": sha(code_path),
                 "module_sha256": sha(Path(__file__).with_name("pera_algebra.py"))},
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "dtype": "float64", "device": "cpu"},
        "arms": [arm(seed, rank, protocol) for seed in protocol["algebra"]["seeds"] for rank in protocol["algebra"]["ranks"]],
        "training": [training(seed, protocol) for seed in protocol["training"]["seeds"]],
        "limitations": ["Algebra and synthetic matrix diagnostic only; no transformer task quality.",
                        "Random rank attainment is not a proof of useful adaptation or novelty.",
                        "The selected excerpt does not establish the full implementation or benchmark reproducibility."],
    }
    checks = {
        "factorization_within_tolerance": all(x["max_factorization_error"] <= protocol["algebra"]["factorization_atol"] for x in result["arms"]),
        "gradient_equivalence_within_tolerance": all(x["gradient_equivalence_max_error"] <= protocol["algebra"]["gradient_atol"] for x in result["arms"]),
        "special_identity_within_tolerance": all(x["special_identity_max_error"] <= protocol["algebra"]["factorization_atol"] for x in result["arms"]),
        "initial_b_and_h_gradients_zero": all(x["steps"][0]["grad_b_norm"] <= 1e-14 and x["steps"][0]["grad_h_norm"] <= 1e-14 for x in result["training"]),
        "initial_a_gradient_nonzero": all(x["steps"][0]["grad_a_norm"] > 1e-10 for x in result["training"]),
        "post_update_h_gradient_observed": all(x["steps"][1]["grad_h_norm"] > 1e-14 for x in result["training"]),
        "rank_bounds_hold": all(x["delta_rank"] <= x["rank_bound"] for x in result["arms"]),
    }
    result["checks"] = checks
    result["checks_pass"] = all(checks.values())
    result["status"] = "completed" if result["checks_pass"] else "checks_failed"
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    if not result["checks_pass"]:
        raise AssertionError("diagnostic checks failed; result preserved")
    print(json.dumps({"status": result["status"], "output": str(args.output), "arms": len(result["arms"])}))


if __name__ == "__main__":
    main()
