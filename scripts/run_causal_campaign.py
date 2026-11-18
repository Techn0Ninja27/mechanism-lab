"""Run the local closed-loop candidate campaign with a frozen dev/final boundary.

No model calls are made here.  The default command writes a plan only; local
training requires ``--execute``.  Final evaluation refuses to start unless a
validated development selection manifest is supplied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_causal_candidates import run_one
from generation.campaign_selection import select_development, canonical_hash

SCHEMA = "causal-candidate-inventory.v1"
DEV = tuple(range(100, 116))
FINAL = tuple(range(200, 232))
WINDOW = tuple(range(88, 120))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def load_inventory(path: Path) -> tuple[dict[str, Any], str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema") != SCHEMA or not isinstance(raw.get("candidates"), list):
        raise ValueError("inventory schema or candidates are malformed")
    if not 1 <= len(raw["candidates"]) <= 128:
        raise ValueError("candidate count outside bounds")
    ids: set[str] = set()
    for candidate in raw["candidates"]:
        if not isinstance(candidate, dict) or not isinstance(candidate.get("id"), str):
            raise ValueError("candidate ID is malformed")
        cid = candidate["id"]
        if cid in ids:
            raise ValueError(f"duplicate candidate ID: {cid}")
        ids.add(cid)
        if not isinstance(candidate.get("call_id"), str) or not isinstance(candidate.get("arm"), str):
            raise ValueError(f"candidate metadata is malformed: {cid}")
        status = candidate.get("status", "valid")
        if status not in {"valid", "invalid"}:
            raise ValueError(f"candidate status is malformed: {cid}")
        if status == "valid":
            program = candidate.get("program")
            if not isinstance(program, dict):
                raise ValueError(f"valid candidate has no program: {cid}")
            digest = hashlib.sha256(json.dumps(program, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            if candidate.get("program_sha256") != digest:
                raise ValueError(f"program hash mismatch: {cid}")
    return raw, sha(path)


def _failure(candidate: dict[str, Any], seed: int, error: str) -> dict[str, Any]:
    return {"id": candidate.get("id"), "call_id": candidate.get("call_id"), "arm": candidate.get("arm"),
            "seed": seed, "status": "failed", "error": error}


def _hold_candidate() -> dict[str, Any]:
    program = {"predict": {"var": "last_sync"}, "update": {"var": "memory"}}
    digest = hashlib.sha256(json.dumps(program, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"id": "__hold__", "call_id": "__baseline__", "arm": "hold", "status": "valid",
            "program": program, "program_sha256": digest}


def _job_id(candidate: dict[str, Any], seed: int) -> str:
    return hashlib.sha256(f"{candidate['id']}\0{seed}".encode()).hexdigest()[:20]


def _canonical_inventory_hash(inventory: dict[str, Any]) -> str:
    return canonical_hash(sorted(inventory["candidates"], key=lambda row: row["id"]))


def _dependency_hashes() -> dict[str, str]:
    from scripts import evaluate_causal_candidates as evaluator
    files = [Path(__file__), ROOT / "scripts/evaluate_causal_candidates.py", ROOT / "generation/causal_program.py", ROOT / "generation/campaign_selection.py",
             ROOT / "operator-studies/gnn-closedloop-gap/gap.py", ROOT / "operator-studies/gnn-sync-gate/screen.py",
             evaluator.gap.DATA / "cora.content", evaluator.gap.DATA / "cora.cites"]
    return {str(path.relative_to(ROOT)): sha(path) for path in files}


def validate_selection(selection_path: Path, inventory_sha: str, inventory: dict[str, Any], stage: str,
                       *, dev_result_path: Path | None = None, code_sha256: str | None = None) -> dict[str, Any]:
    if stage != "final":
        raise ValueError("selection is only required for final stage")
    data = json.loads(selection_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema") != "causal-candidate-selection.v1":
        raise ValueError("selection manifest schema is invalid")
    if data.get("source_inventory_sha256") != inventory_sha or data.get("split") != "dev" or data.get("status") != "complete":
        raise ValueError("selection does not bind to this inventory")
    if data.get("candidate_inventory_sha256") != _canonical_inventory_hash(inventory):
        raise ValueError("selection candidate inventory hash mismatch")
    if code_sha256 is not None and data.get("code_sha256") != code_sha256:
        raise ValueError("selection code hash mismatch")
    if dev_result_path is None or not dev_result_path.is_file():
        raise ValueError("final evaluation requires a development result")
    dev_result = json.loads(dev_result_path.read_text(encoding="utf-8"))
    if dev_result.get("schema") != "causal-campaign-result.v1" or dev_result.get("stage") != "dev":
        raise ValueError("development result is not a campaign result")
    if dev_result.get("inventory_sha256") != inventory_sha or dev_result.get("dependency_hashes") != _dependency_hashes():
        raise ValueError("development result inventory hash mismatch")
    # Re-derive the selection from the bound development aggregate.  A supplied
    # selected list alone is never sufficient to open final data.
    derived = select_development(inventory["candidates"], dev_result.get("runs"), expected_seeds=DEV,
                                 inventory_sha256=inventory_sha, code_sha256=code_sha256)
    if derived.get("status") != "complete" or derived.get("selected") != data.get("selected"):
        raise ValueError("selection does not match re-derived development picks")
    candidates = {x["id"]: x for x in inventory["candidates"]}
    selected = data.get("selected")
    if not isinstance(selected, list):
        raise ValueError("selection list is missing")
    chosen: list[dict[str, Any]] = []
    seen_calls: set[str] = set()
    for row in selected:
        if not isinstance(row, dict) or not isinstance(row.get("call_id"), str):
            raise ValueError("selection row malformed")
        call = row["call_id"]
        if call in seen_calls: raise ValueError("duplicate selected call")
        seen_calls.add(call)
        cid = row.get("candidate_id")
        if cid is None and row.get("status") == "no_eligible_candidate":
            continue
        if not isinstance(cid, str) or cid not in candidates:
            raise ValueError("selection candidate is unknown")
        if row.get("status") != "selected":
            raise ValueError("final selection contains an unselected call")
        candidate = candidates[cid]
        if candidate.get("call_id") != call or candidate.get("status") != "valid":
            raise ValueError("selected candidate does not bind to its call")
        chosen.append(candidate)
    return {"manifest": data, "candidates": chosen}


def plan(inventory_path: Path, output: Path, stage: str, selection_path: Path | None = None,
         dev_result_path: Path | None = None, code_sha256: str | None = None) -> dict[str, Any]:
    if output.exists(): raise FileExistsError("output exists; choose a fresh directory")
    inventory, digest = load_inventory(inventory_path)
    selected = None
    if stage == "final":
        if selection_path is None: raise ValueError("final stage requires a selection manifest")
        selected = validate_selection(selection_path, digest, inventory, stage, dev_result_path=dev_result_path, code_sha256=code_sha256)
    output.mkdir(parents=True)
    result = {"schema": "causal-campaign-plan.v1", "stage": stage, "inventory_sha256": digest,
              "dependency_hashes": _dependency_hashes(),
              "candidate_count": len(inventory["candidates"]), "seeds": list(DEV if stage == "dev" else FINAL),
              "selection_sha256": sha(selection_path) if selection_path else None,
              "development_result_sha256": sha(dev_result_path) if dev_result_path else None,
              "selection_required": stage == "final", "selected_count": len(selected["candidates"]) if selected else None,
              "execute_required": True, "paid_calls": 0, "auto_execute": False,
              "novelty_established": False, "train_eligible": False}
    (output / "plan.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def execute(inventory_path: Path, output: Path, stage: str, selection_path: Path | None = None,
            *, dev_result_path: Path | None = None, code_sha256: str | None = None, seed_override: int | None = None) -> dict[str, Any]:
    inventory, digest = load_inventory(inventory_path)
    if stage == "final":
        if selection_path is None: raise ValueError("final stage requires selection")
        candidates = validate_selection(selection_path, digest, inventory, stage, dev_result_path=dev_result_path, code_sha256=code_sha256)["candidates"]
        candidates = [*candidates, _hold_candidate()]
        seeds = FINAL if seed_override is None else (seed_override,)
    else:
        candidates = inventory["candidates"]
        seeds = DEV if seed_override is None else (seed_override,)
    output.mkdir(parents=True, exist_ok=True)
    dependencies = _dependency_hashes()
    freeze = {"inventory_sha256": digest, "selection_sha256": sha(selection_path) if selection_path else None,
              "development_result_sha256": sha(dev_result_path) if dev_result_path else None, "stage": stage,
              "seeds": list(seeds), "dependency_hashes": dependencies}
    freeze_path = output / "freeze.json"
    if freeze_path.exists() and json.loads(freeze_path.read_text()) != freeze:
        raise ValueError("existing campaign freeze does not match")
    if not freeze_path.exists(): freeze_path.write_text(json.dumps(freeze, indent=2) + "\n")
    # Import data only after the final selection guard has passed.
    from scripts import evaluate_causal_candidates as evaluator
    x, y, adj = evaluator.gap.screen.load_cora(evaluator.gap.DATA)
    sp = evaluator.gap.splits(y)
    rows = []
    for candidate in candidates:
        for seed in seeds:
            path = output / "jobs" / f"{_job_id(candidate, seed)}.json"
            if path.exists():
                old = json.loads(path.read_text())
                if (old.get("id") != candidate.get("id") or old.get("seed") != seed or
                        old.get("call_id") != candidate.get("call_id") or old.get("arm") != candidate.get("arm")):
                    raise ValueError(f"cached job metadata mismatch: {path}")
                rows.append(old)
                continue
            if candidate.get("status") != "valid": row = _failure(candidate, seed, candidate.get("error", "candidate marked invalid"))
            else: row = evaluator.run_one(x, y, adj, sp, candidate, seed=seed, epochs=120, window=WINDOW)
            path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(row, indent=2) + "\n")
            rows.append(row)
    result = {"schema": "causal-campaign-result.v1", "stage": stage, "inventory_sha256": digest,
              "dependency_hashes": dependencies,
              "selection_sha256": sha(selection_path) if selection_path else None, "seed_count": len(seeds),
              "candidate_count": len(candidates), "runs": rows, "paid_calls": 0,
              "auto_execute": False, "novelty_established": False, "train_eligible": False}
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--inventory", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--stage", choices=("dev", "final"), default="dev"); p.add_argument("--selection", type=Path); p.add_argument("--dev-result", type=Path); p.add_argument("--code-sha256"); p.add_argument("--execute", action="store_true")
    p.add_argument("--seed", type=int, help="single smoke seed; only for local smoke tests")
    args = p.parse_args()
    if args.seed is not None and (args.stage != "dev" or args.seed != 98):
        raise ValueError("--seed is smoke-only and cannot select measurement seeds")
    if args.execute: result = execute(args.inventory, args.output, args.stage, args.selection, dev_result_path=args.dev_result, code_sha256=args.code_sha256, seed_override=args.seed)
    else: result = plan(args.inventory, args.output, args.stage, args.selection, dev_result_path=args.dev_result, code_sha256=args.code_sha256)
    print(json.dumps({"schema": result["schema"], "stage": result["stage"], "output": str(args.output), "paid_calls": 0}))


if __name__ == "__main__": main()
