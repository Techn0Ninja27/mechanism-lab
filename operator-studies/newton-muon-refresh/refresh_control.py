"""Matched refresh-frequency control for the frozen Newton--Muon harness.

One invocation runs one fresh `(width, refresh)` arm.  Keeping the arm as a
single-process command makes process order and allocator carry-over explicit;
the shell driver in the README runs each arm separately.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import sys
import time
import warnings
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FROZEN = ROOT / "operator-studies/newton-muon-cost-screen/cost_profile.py"


def load_frozen():
    spec = importlib.util.spec_from_file_location("frozen_newton_cost", FROZEN)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen cost profile")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def expected_refreshes(refresh: int, warmup: int = 8, steps: int = 128) -> int:
    if refresh <= 0 or warmup < 0 or steps <= 0:
        raise ValueError("refresh, warmup and steps must be positive")
    return sum((step % refresh) == 0 for step in range(warmup + 1, warmup + steps + 1))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.steps != 128 or args.warmup != 8 or args.window != 32:
        raise ValueError("the frozen protocol requires warmup=8, steps=128, window=32")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        raise ValueError("set PYTORCH_ENABLE_MPS_FALLBACK=0")
    cp = load_frozen()
    device = cp.resolve_device(args.device)
    text = args.text.read_text(encoding="utf-8")
    batches = [(x.to(device), y.to(device)) for x, y in cp.make_batches(
        text, 256, seq=64, batch=4, steps=args.warmup + args.steps, seed=0
    )]
    torch.manual_seed(0)
    model, opt, adam = cp._make_arm(args.width, device, 0, "newton_cholesky", args.refresh)
    losses = []
    measured_windows = []
    warning_text = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for i in range(args.warmup):
                loss, _ = cp._train_step(model, opt, adam, *batches[i])
                losses.append(loss)
            for window in range(args.steps // args.window):
                cp.synchronize(device)
                start = time.perf_counter()
                begin = args.warmup + window * args.window
                for i in range(begin, begin + args.window):
                    loss, _ = cp._train_step(model, opt, adam, *batches[i])
                    losses.append(loss)
                cp.synchronize(device)
                measured_windows.append({
                    "window": window,
                    "steps": args.window,
                    "seconds": time.perf_counter() - start,
                    "refreshes": expected_refreshes(args.refresh, begin, args.window),
                })
            warning_text = [str(item.message)[:500] for item in caught]
        values = [float(value.cpu()) for value in losses]
        infos = [int(info.max().cpu()) for info in opt.factor_infos]
        refresh_counts = sorted({state.refresh_count for state in opt.states.values()})
        expected = expected_refreshes(args.refresh, args.warmup, args.steps)
        if not all(torch.isfinite(x).item() for x in losses):
            raise ValueError("non-finite loss")
        if any(infos):
            raise ValueError(f"Cholesky failure info={max(infos)}")
        if refresh_counts != [expected]:
            raise ValueError(f"unexpected refresh counts {refresh_counts}; expected {expected}")
        result = {
            "schema": "newton-muon-refresh-control.v1",
            "scope": "Matched short end-to-end cost control; loss is a training-stream sanity trace, not optimizer-quality or time-to-target evidence.",
            "protocol": {"width": args.width, "refresh": args.refresh, "warmup": args.warmup, "steps": args.steps, "window": args.window, "windows": len(measured_windows), "seed": 0, "batch": 4, "seq": 64, "vocab": 256},
            "device": str(device), "torch_version": torch.__version__, "platform": platform.platform(),
            "text": str(args.text), "text_sha256": sha(args.text),
            "frozen_cost_profile": str(FROZEN), "frozen_cost_profile_sha256": sha(FROZEN),
            "fallback_env": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
            "warnings": warning_text, "internal_cpu_fallback_detected": any("cpu" in x.lower() and "fall" in x.lower() for x in warning_text),
            "windows": measured_windows, "ms_per_step": [1000 * x["seconds"] / x["steps"] for x in measured_windows],
            "loss_finite": True, "loss_first": values[0], "loss_last": values[-1], "losses": values,
            "expected_refreshes_per_matrix": expected, "refresh_counts": refresh_counts,
            "cholesky_info_max": max(infos, default=0),
            "covariance_state_bytes": sum(x.k.numel() * x.k.element_size() for x in opt.states.values()),
            "inverse_state_bytes": sum(x.inverse.numel() * x.inverse.element_size() for x in opt.states.values()),
            "memory_snapshot": ({"current_allocated_bytes": torch.mps.current_allocated_memory(), "driver_allocated_bytes": torch.mps.driver_allocated_memory(), "scope": "end-of-arm snapshots, not peak"} if device.type == "mps" else {}),
        }
    finally:
        del model, opt, adam, batches, losses
        if device.type == "mps":
            torch.mps.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, choices=(256, 512), required=True)
    parser.add_argument("--refresh", type=int, choices=(16, 32, 64), required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--window", type=int, default=32)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({"width": args.width, "refresh": args.refresh, "ms_per_step": result["ms_per_step"], "refreshes": result["refresh_counts"]}))


if __name__ == "__main__":
    main()
