"""Capture real K/G tensors for offline inverse-method comparisons.

This deliberately performs no timing or optimizer comparison.  It reuses the
frozen cost-screen model and update order, and saves only first/last block
covariances plus raw gradients at refresh steps.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "operator-studies/newton-muon-cost-screen/cost_profile.py"
TEXT = ROOT / "results/newton-muon-cost-screen-v1/text.txt"
STEPS = (32, 96)


def load_cost_profile():
    spec = importlib.util.spec_from_file_location("newton_cost_profile", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen cost profile")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def selected(model):
    chosen = []
    for block_index in (0, len(model.blocks) - 1):
        block = model.blocks[block_index]
        for name in ("qkv", "proj", "fc1", "fc2"):
            chosen.append((f"block{block_index}.{name}", getattr(block, name)))
    return chosen


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture(width: int, device_name: str, output: Path) -> dict:
    cp = load_cost_profile()
    device = cp.resolve_device(device_name)
    raw_text = TEXT.read_bytes()
    batches = [(x.to(device), y.to(device)) for x, y in cp.make_batches(
        raw_text.decode("utf-8"), 256, seq=64, batch=4, steps=96, seed=0
    )]
    torch.manual_seed(0)
    model, optimizer, adam = cp._make_arm(width, device, 0, "newton_cholesky", 32)
    chosen = selected(model)
    records = {}
    losses = []
    fallback_warnings = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for step_number, (x, y) in enumerate(batches, start=1):
            for module in optimizer.modules:
                module.capture_input = ((optimizer.step_index + 1) % optimizer.refresh == 0)
            model.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
            if not torch.isfinite(loss):
                raise ValueError(f"nonfinite loss at step {step_number}")
            losses.append(float(loss.detach().cpu()))
            loss.backward()
            if step_number in STEPS:
                before = {}
                for name, module in chosen:
                    if module.weight.grad is None:
                        raise RuntimeError(f"missing raw gradient for {name} at step {step_number}")
                    before[name] = {
                        "G": module.weight.grad.detach().cpu().numpy().copy(),
                        "block_count": module.block_count,
                        "in_features": module.in_features,
                        "out_features": module.out_features,
                    }
                optimizer.step()
                adam.step()
                adam.zero_grad(set_to_none=True)
                for name, module in chosen:
                    state = optimizer.states[id(module.weight)]
                    before[name]["K"] = state.k.detach().cpu().numpy().copy()
                records[step_number] = before
            else:
                optimizer.step()
                adam.step()
                adam.zero_grad(set_to_none=True)
        fallback_warnings = [str(item.message)[:500] for item in caught]
    arrays = {}
    metadata = {}
    for step_number, modules in records.items():
        for name, record in modules.items():
            key = name.replace(".", "_")
            arrays[f"step{step_number}_{key}_K"] = record["K"]
            arrays[f"step{step_number}_{key}_G"] = record["G"]
            metadata[f"step{step_number}_{name}"] = {
                "K_key": f"step{step_number}_{key}_K",
                "G_key": f"step{step_number}_{key}_G",
                "K_shape": list(record["K"].shape),
                "G_shape": list(record["G"].shape),
                "block_count": record["block_count"],
                "in_features": record["in_features"],
                "out_features": record["out_features"],
                "gradient_timing": "raw G copied after backward and before optimizer.step",
                "covariance_timing": "K copied after optimizer.step at source refresh step",
            }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    return {
        "width": width,
        "device": str(device),
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "seed": 0,
        "steps": list(STEPS),
        "loss_finite": True,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "fallback_warnings": fallback_warnings,
        "fallback_env": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
        "npz": str(output),
        "npz_sha256": sha(output),
        "npz_bytes": output.stat().st_size,
        "arrays": metadata,
        "source": {
            "cost_profile": str(SOURCE),
            "cost_profile_sha256": sha(SOURCE),
            "text": str(TEXT),
            "text_sha256": sha(TEXT),
        },
        "scope": "Offline K/G capture only; no timing, optimizer-quality, held-out, usefulness or novelty claim.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, choices=(256, 512), required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.manifest.exists():
        raise FileExistsError("preserve existing capture; choose fresh output and manifest")
    result = capture(args.width, args.device, args.output)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"width": args.width, "npz_sha256": result["npz_sha256"], "bytes": result["npz_bytes"], "warnings": result["fallback_warnings"]}))


if __name__ == "__main__":
    main()
