"""Audit the proposed random-feature normalization-axis change.

The proposal and review are archived in results/improvement-phase-2026-11-16.
Only selected functions are AST-extracted from the official metric.py snapshot;
the repository module is never imported.
"""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.autograd import Variable

SEED = 16019
ROOT = Path(__file__).parents[2]
SOURCE = ROOT / "results/improvement-phase-2026-11-16/sources/stablemiss/metric.py"


def extract_official():
    tree = ast.parse(SOURCE.read_text())
    wanted = {"random_fourier_features", "cov", "intra_feature_correlation",
              "intra_mask_correlation", "inter_correlation"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    if {n.name for n in nodes} != wanted:
        raise RuntimeError("official function extraction incomplete")
    ns = {"np": np, "torch": torch, "Variable": Variable}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), ns)
    return ns


def proposed_features(x, w, b, *, sum=True):
    """Same body as official function, changing only min/max dimension 1 -> 0."""
    n, r = x.shape
    x3 = x.view(n, r, 1)
    mid = torch.matmul(x3, w.t()) + b
    mid -= mid.min(dim=0, keepdim=True)[0]
    mid /= mid.max(dim=0, keepdim=True)[0]
    mid *= np.pi / 2.0
    z = torch.sqrt(torch.tensor(2.0 / w.shape[0]))
    return z * (torch.cos(mid) + torch.sin(mid)) if sum else z * torch.cat((torch.cos(mid), torch.sin(mid)), dim=-1)


def mask(n, d, rng):
    out = (rng.random((n, d)) > 0.4).astype(np.float32)
    out[:, 0] = 1.0  # explicit fully observed constant-column case
    return torch.tensor(out)


def stats(z):
    return {"shape": list(z.shape), "nan_count": int(torch.isnan(z).sum()),
            "finite": bool(torch.isfinite(z).all()),
            "column_variances": [float(v) if torch.isfinite(v) else None for v in torch.var(z[..., 0], dim=0, unbiased=False)]}


def call_shape(ns, name, x, m, w):
    try:
        torch.manual_seed(SEED)
        value = ns[name](m, w) if name == "intra_mask_correlation" else ns[name](x, m, w)
        return {"status": "returned", "shape": list(value.shape) if hasattr(value, "shape") else [],
                "finite": bool(torch.isfinite(value).all())}
    except Exception as exc:
        return {"status": "raised", "type": type(exc).__name__, "message": str(exc)}


def main(output: Path):
    rng = np.random.default_rng(SEED)
    ns = extract_official()
    records = {}
    for n, d in ((64, 5), (2048, 20)):
        m = mask(n, d, rng)
        w = torch.ones((1, 1))
        # Match the source's random phase construction while keeping both paths
        # on the same inputs; b has shape (n,d,1).
        torch.manual_seed(SEED)
        b = 2 * np.pi * torch.rand((d, 1)).repeat((n, 1, 1))
        original = ns["random_fourier_features"](m, w=w, b=b, num_f=1, sum=True)
        proposed = proposed_features(m, w, b, sum=True)
        uniform = torch.full((n,), 1.0 / n)
        proposed_var = torch.var(proposed[..., 0], dim=0, unbiased=False)
        finite_var = proposed_var[torch.isfinite(proposed_var)]
        records[f"n{n}_d{d}"] = {
            "original": stats(original), "proposed": stats(proposed),
            "constant_column_proposed_nan": bool(torch.isnan(proposed[:, 0, :]).any()),
            "finite_binary_columns_proposed_constant": bool(torch.all(finite_var < 1e-10)),
            "source_loss_calls": {name: call_shape(ns, name, m, m, uniform)
                                  for name in ("intra_feature_correlation", "intra_mask_correlation", "inter_correlation")},
        }
    result = {
        "seed": SEED,
        "serialization_note": "Nonfinite column variances are null; nan_count and finite flags retain the numerical finding. Original nonstandard JSON retained locally.",
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "source_functions": "AST-extracted random_fourier_features, cov, intra_feature_correlation, intra_mask_correlation, inter_correlation; no module import",
        "records": records,
        "verdict": {
            "normalization_axis_claim": "supported for the exact sum=True, num_f=1 binary-column construction: proposed outputs are constant (or NaN for constant columns)",
            "shape_claim": "supported if n != d for intra_feature_correlation and inter_correlation in the supplied function path; intra_mask_correlation is separately shape-safe",
            "mechanistic_interpretation": "The proposed axis removes mask-feature variation from the mask penalty rather than demonstrating better decorrelation. It deletes the mask-dependent covariance signal; it is not a validated improvement.",
            "novelty_established": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(Path(__file__).with_name("results.json"))
