"""Run one provenance-bound held-out Muon/Newton--Muon arm."""
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
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FROZEN = ROOT / "operator-studies/newton-muon-cost-screen/cost_profile.py"
SEQ = 64
BATCH = 4
STEPS = 256
CHECKPOINTS = tuple(range(0, STEPS + 1, 32))


def load_frozen():
    spec = importlib.util.spec_from_file_location("heldout_frozen_cost", FROZEN)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen cost profile")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_hash(values) -> str:
    digest = hashlib.sha256()
    for value in values:
        data = value.detach().cpu().contiguous().numpy().tobytes()
        digest.update(len(data).to_bytes(8, "little"))
        digest.update(data)
    return digest.hexdigest()


def model_hash(model) -> str:
    return tensor_hash([p for p in model.parameters()])


def eval_batches(text: str, *, vocab: int = 256, seq: int = SEQ, batch: int = BATCH):
    ids = torch.tensor(list(text.encode("utf-8")), dtype=torch.long)
    # A window consumes seq input bytes plus its next-byte target. Exact
    # multiples are valid; only a final incomplete transition is discarded.
    usable = ids.numel() // (seq + 1)
    windows = []
    for index in range(usable):
        start = index * (seq + 1)
        windows.append((ids[start:start + seq], ids[start + 1:start + seq + 1]))
    batches = []
    for index in range(0, len(windows), batch):
        group = windows[index:index + batch]
        x = torch.stack([item[0] for item in group])
        y = torch.stack([item[1] for item in group])
        batches.append((x, y))
    return batches, {"bytes": ids.numel(), "candidate_windows": len(windows), "batches": len(batches),
                    "used_windows": len(windows), "tail_windows": 0,
                    "tail_bytes": int(ids.numel() - len(windows) * (seq + 1)),
                    "seq": seq, "batch": batch, "nonoverlapping": True}


def optimizer_signature(optimizer):
    values = [torch.tensor([optimizer.step_index], dtype=torch.int64)]
    for module in optimizer.modules:
        values.append(optimizer.momentum[id(module.weight)])
    for state in optimizer.states.values():
        values.extend([state.k, state.inverse] if state.inverse is not None else [state.k])
        values.append(torch.tensor([state.last_refresh, state.refresh_count], dtype=torch.int64))
    return tensor_hash(values)


@torch.no_grad()
def evaluate(model, batches, device, optimizer=None):
    was_training = model.training
    captures = [module.capture_input for module in model.matrix_modules()]
    before = model_hash(model)
    optimizer_before = optimizer_signature(optimizer) if optimizer is not None else None
    try:
        model.eval()
        for module in model.matrix_modules():
            module.capture_input = False
        total = 0.0
        tokens = 0
        for x, y in batches:
            logits = model(x.to(device))
            total += float(F.cross_entropy(logits.reshape(-1, 256), y.to(device).reshape(-1), reduction="sum").cpu())
            tokens += y.numel()
        after = model_hash(model)
    finally:
        for module, capture in zip(model.matrix_modules(), captures):
            module.capture_input = capture
        model.train(was_training)
    if before != after:
        raise AssertionError("evaluation mutated model parameters")
    if optimizer is not None and optimizer_before != optimizer_signature(optimizer):
        raise AssertionError("evaluation mutated optimizer state")
    return {"mean_cross_entropy": total / tokens, "tokens": tokens, "model_hash_before": before, "model_hash_after": after}


def verify_split(train_text: Path, heldout_text: Path, manifest: Path):
    data = json.loads(manifest.read_text())
    files = data.get("files", {})
    expected_train = files.get("text.txt", {}).get("sha256")
    expected_heldout = files.get("unused-validation.txt", {}).get("sha256")
    if sha(train_text) != expected_train or sha(heldout_text) != expected_heldout:
        raise ValueError("text fixture hash does not match frozen manifest")
    source_rows = data.get("sources", [])
    train_ids = [row["id"] for row in source_rows if row.get("split") == "profile_train"]
    heldout_ids = [row["id"] for row in source_rows if row.get("split") == "reserved_unused"]
    train_hashes = [row.get("text_sha256") for row in source_rows if row.get("split") == "profile_train"]
    heldout_hashes = [row.get("text_sha256") for row in source_rows if row.get("split") == "reserved_unused"]
    if (not train_ids or len(set(train_ids)) != len(train_ids) or len(set(heldout_ids)) != len(heldout_ids)
            or set(train_ids) & set(heldout_ids) or set(train_hashes) & set(heldout_hashes)
            or any(not isinstance(item, str) for item in train_hashes + heldout_hashes)
            or len(train_ids) != 96 or len(heldout_ids) != 32):
        raise ValueError("frozen source split is missing, overlapping, or wrong-sized")
    return {"manifest_sha256": sha(manifest), "train_ids_sha256": hashlib.sha256(json.dumps(train_ids, separators=(",", ":")).encode()).hexdigest(),
            "heldout_ids_sha256": hashlib.sha256(json.dumps(heldout_ids, separators=(",", ":")).encode()).hexdigest(),
            "train_records": len(train_ids), "heldout_records": len(heldout_ids), "disjoint": True}


def run(args):
    if args.output.exists():
        raise FileExistsError("refusing to overwrite output")
    if args.width != 256 or args.refresh != 32 or args.steps != 256 or args.checkpoint != 32:
        raise ValueError("frozen protocol requires width=256, refresh=32, steps=256, checkpoint=32")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        raise ValueError("set PYTORCH_ENABLE_MPS_FALLBACK=0")
    split = verify_split(args.train_text, args.heldout_text, args.manifest)
    cp = load_frozen()
    device = cp.resolve_device(args.device)
    heldout, heldout_spec = eval_batches(args.heldout_text.read_text(encoding="utf-8"))
    train_text = args.train_text.read_text(encoding="utf-8")
    train_batches = cp.make_batches(train_text, 256, seq=SEQ, batch=BATCH, steps=STEPS, seed=args.seed)
    train_batch_hash = tensor_hash([v for pair in train_batches for v in pair])
    train_batches = [(x.to(device), y.to(device)) for x, y in train_batches]
    torch.manual_seed(args.seed)
    model, optimizer, adam = cp._make_arm(args.width, device, args.seed, args.mode, args.refresh)
    initial_hash = model_hash(model)
    warnings_text = []
    checkpoints = []
    training_losses = []
    factor_infos = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            checkpoints.append({"step": 0, "training_seconds": 0.0, "training_loss": None, "heldout": evaluate(model, heldout, device, optimizer)})
            start = time.perf_counter()
            cumulative = 0.0
            for step in range(1, STEPS + 1):
                loss, _ = cp._train_step(model, optimizer, adam, *train_batches[step - 1])
                training_losses.append(loss.detach())
                if step % args.checkpoint == 0:
                    cp.synchronize(device)
                    cumulative += time.perf_counter() - start
                    factor_infos = [int(info.max().cpu()) for info in optimizer.factor_infos]
                    if not torch.isfinite(torch.stack(training_losses[-args.checkpoint:])).all().item():
                        raise ValueError("non-finite training loss")
                    held = evaluate(model, heldout, device, optimizer)
                    checkpoints.append({"step": step, "training_seconds": cumulative, "training_loss": float(loss.cpu()), "heldout": held})
                    start = time.perf_counter()
            warnings_text = [str(item.message)[:500] for item in caught]
        if any(factor_infos):
            raise ValueError(f"Cholesky failure info={max(factor_infos)}")
        if warnings_text:
            raise RuntimeError("warnings observed: " + " | ".join(warnings_text))
        if not torch.isfinite(torch.stack(training_losses)).all().item():
            raise ValueError("non-finite training loss")
        if not all(torch.isfinite(torch.tensor(row["heldout"]["mean_cross_entropy"])) for row in checkpoints):
            raise ValueError("non-finite held-out loss")
        return {"schema": "newton-muon-heldout-arm.v1", "status": "completed",
                "scope": "Width-256 local held-out control; no universal optimizer or novelty claim.",
                "protocol": {"mode": args.mode, "seed": args.seed, "order_index": args.order_index, "width": args.width, "refresh": args.refresh,
                             "steps": STEPS, "checkpoint": args.checkpoint, "seq": SEQ, "batch": BATCH},
                "command": " ".join(sys.argv), "python_version": sys.version, "device": str(device),
                "torch_version": torch.__version__, "platform": platform.platform(), "initial_model_sha256": initial_hash,
                "final_model_sha256": model_hash(model), "train_batch_stream_sha256": train_batch_hash,
                "train_text_sha256": sha(args.train_text), "heldout_text_sha256": sha(args.heldout_text),
                "frozen_cost_profile_sha256": sha(FROZEN), "runner_sha256": sha(Path(__file__)),
                "split": split, "heldout_spec": heldout_spec, "fallback_env": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
                "warnings": warnings_text, "internal_cpu_fallback_detected": any("cpu" in x.lower() and "fall" in x.lower() for x in warnings_text),
                "cholesky_info_max": max(factor_infos, default=0), "training_losses": [float(x.cpu()) for x in training_losses],
                "training_loss_finite": True, "checkpoints": checkpoints,
                "optimizer_refresh_counts": sorted({state.refresh_count for state in optimizer.states.values()})}
    finally:
        del model, optimizer, adam, train_batches
        if device.type == "mps":
            torch.mps.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("muon", "newton_cholesky"), required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1), required=True)
    parser.add_argument("--order-index", type=int, required=True)
    parser.add_argument("--train-text", type=Path, required=True)
    parser.add_argument("--heldout-text", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--refresh", type=int, default=32)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--checkpoint", type=int, default=32)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite output")
    try:
        result = run(args)
    except Exception as exc:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        failure = {"schema": "newton-muon-heldout-arm.v1", "status": "failed", "command": " ".join(sys.argv),
                   "python_version": sys.version, "protocol": {k: str(v) for k, v in vars(args).items()},
                   "error_type": type(exc).__name__, "error": str(exc),
                   "train_text_sha256": sha(args.train_text) if args.train_text.exists() else None,
                   "heldout_text_sha256": sha(args.heldout_text) if args.heldout_text.exists() else None}
        args.output.write_text(json.dumps(failure, indent=2) + "\n")
        raise
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"mode": args.mode, "seed": args.seed, "status": result["status"]}), flush=True)


if __name__ == "__main__":
    main()
