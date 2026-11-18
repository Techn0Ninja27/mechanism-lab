"""Execute the frozen seven-arm development control stage (local only).

Default mode is plan-only. ``--execute`` runs 7 × 16 local CPU jobs and writes
immutable per-job checkpoints. This stage never opens final seeds or calls an
API; it only estimates the reference variance needed to plan final replication.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "operator-studies/gnn-closedloop-gap")]
from generation.campaign_selection import canonical_hash
from scripts.evaluate_causal_candidates import run_one
from scripts.run_causal_campaign import _hold_candidate, _dependency_hashes, WINDOW
from scripts.generate_causal_campaign import linear_controls, aa_calibration
import gap

SEEDS = tuple(range(100, 116))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _controls() -> list[dict[str, Any]]:
    rows = [_hold_candidate()]
    gains = {"gain_m05": -0.5, "gain_m025": -0.25, "gain_p01": 0.1}
    linear = linear_controls()
    for name, gain in gains.items():
        row = next(x for x in linear if x.get("gain") == gain)
        rows.append({**row, "id": name, "call_id": name, "arm": name})
    mean_row = next(x for x in linear if x.get("control_kind") == "mean_sync")
    rows.append({**mean_row, "id": "mean_sync", "call_id": "mean_sync", "arm": "mean_sync"})
    aa = aa_calibration()[0]
    rows.append({**aa, "id": "aa_control", "call_id": "aa_control", "arm": "aa_control"})
    rows.append({"id": "g1_exact_detached", "call_id": "g1_exact_detached", "arm": "g1_exact_detached", "status": "reference"})
    return rows


def _freeze(output: Path) -> dict[str, Any]:
    controls = _controls()
    confirm = ROOT / "operator-studies/gnn-closedloop-confirm/confirm.py"
    protocol = ROOT / "reports/generation-comparison-v2/protocol.md"
    return {"schema": "causal-stage0-freeze.v1", "seeds": list(SEEDS), "controls": [x["id"] for x in controls],
            "window": list(WINDOW), "task": {"parts": 4, "layers": 3, "sync": 16, "lr_drop": False, "epochs": 120},
            "dependency_hashes": {**_dependency_hashes(), "stage0_runner": _sha(Path(__file__)),
                                  "confirmation_trainer": _sha(confirm), "campaign_protocol": _sha(protocol)},
            "control_inventory_sha256": canonical_hash([{k: v for k, v in x.items() if k != "error"} for x in controls]),
            "final_execution_enabled": False, "paid_calls": 0}


def _g1_run(x, y, adj, sp, seed: int) -> dict[str, Any]:
    import importlib.util
    path = ROOT / "operator-studies/gnn-closedloop-confirm/confirm.py"
    spec = importlib.util.spec_from_file_location("stage0_confirm", path)
    if spec is None or spec.loader is None: raise RuntimeError("confirmation trainer unavailable")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    raw = mod.train(x, y, adj, sp, seed=seed, layers=3, parts=4, lr_drop=False, arm="G1", sync=16,
                    coef=[0.0, 0.0], window=WINDOW)
    return {"id": "g1_exact_detached", "call_id": "g1_exact_detached", "arm": "g1_exact_detached", "seed": seed,
            "status": "complete", "window": [{"t": r["t"], "exact_eval": r["exact_eval"], "own_eval": r["stale_eval"]} for r in raw["window"]],
            "post_update_exact_eval": raw["post_update_exact_eval_120"], "final_train_loss": raw["final_train_loss"]}


def analyse(rows: list[dict[str, Any]], controls: list[dict[str, Any]]) -> dict[str, Any]:
    expected = {(c["id"], seed) for c in controls for seed in SEEDS}
    if len(rows) != len(expected) or {(r.get("id"), r.get("seed")) for r in rows} != expected:
        raise ValueError("stage0 job keys missing, duplicated or foreign")
    by = {(r["id"], r["seed"]): r for r in rows}
    scores = {}; failures = {}
    for control in controls:
        cid = control['id']; failures[cid] = []
        for seed in SEEDS:
            row = by[(cid,seed)]
            try:
                if row.get('arm') != control['arm'] or row.get('call_id') != control['call_id']:
                    raise ValueError('metadata mismatch')
                if row.get('status') != 'complete': raise ValueError('failed job')
                window = row.get('window')
                if not isinstance(window,list) or any(not isinstance(x,dict) for x in window) or [x.get('t') for x in window] != list(WINDOW):
                    raise ValueError('invalid primary window')
                vals = [x.get('exact_eval') for x in window]
                if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<=v<=1 for v in vals):
                    raise ValueError('invalid primary accuracy')
                scores[(cid,seed)] = mean(vals)
            except ValueError as exc: failures[cid].append({'seed':seed,'reason':str(exc)})
    reference_ids = ('__hold__','aa_control','g1_exact_detached')
    reference_failed = any(failures[cid] for cid in reference_ids)
    aa_equal = not reference_failed and all(
        by[('aa_control',seed)]['window'] == by[('__hold__',seed)]['window'] and
        all(by[('aa_control',seed)].get(k)==by[('__hold__',seed)].get(k) for k in ('final_train_loss','post_update_exact_eval'))
        for seed in SEEDS)
    arms = {}
    for control in controls:
        cid = control['id']
        complete = not failures[cid] and not failures['__hold__']
        diff = [scores[(cid,s)]-scores[('__hold__',s)] for s in SEEDS] if complete else []
        arms[cid] = {'n':len(diff),'failed_seeds':failures[cid],
                     'paired_mean_minus_hold':mean(diff) if diff else None,
                     'paired_sd_minus_hold':stdev(diff) if diff else None}
    sd = arms['g1_exact_detached']['paired_sd_minus_hold']
    proposed_n = int(math.ceil(max(32, math.ceil((sd/.005)**2))/16)*16) if sd is not None else None
    ready = not reference_failed and aa_equal and proposed_n is not None and proposed_n<=128
    return {'schema':'causal-stage0-result.v1','status':'reference_ready' if ready else 'stop_before_paid',
            'arms':arms,'aa_exact_identity':aa_equal,
            'reference':{'g1_paired_sd':sd,'proposed_final_seed_count':proposed_n,'cap':128},
            'training_opened':True,'paid_calls':0}


def execute(output: Path) -> dict[str, Any]:
    if output.exists() and (output / "freeze.json").exists():
        freeze = json.loads((output / "freeze.json").read_text())
        if freeze != _freeze(output): raise ValueError("stage0 freeze changed")
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError("output is nonempty")
    output.mkdir(parents=True, exist_ok=True)
    freeze = _freeze(output)
    (output / "freeze.json").write_text(json.dumps(freeze, indent=2) + "\n")
    x, y, adj = gap.screen.load_cora(gap.DATA); sp = gap.splits(y)
    rows = []
    for control in _controls():
        for seed in SEEDS:
            path = output / "jobs" / f"{control['id']}-{seed}.json"; path.parent.mkdir(exist_ok=True)
            if path.exists():
                row = json.loads(path.read_text())
                if row.get("id") != control["id"] or row.get("seed") != seed or row.get("arm") != control.get("arm"):
                    raise ValueError(f"cached stage0 metadata mismatch: {path}")
            else:
                try:
                    row = (_g1_run(x, y, adj, sp, seed) if control["id"] == "g1_exact_detached" else
                           run_one(x, y, adj, sp, control, seed=seed, epochs=120, parts=4, layers=3, sync=16, window=WINDOW))
                except Exception as exc:
                    row = {"id": control["id"], "call_id": control["call_id"], "arm": control["arm"], "seed":seed,
                           "status":"failed", "error":f"{type(exc).__name__}: {str(exc)[:500]}"}
                with path.open('x') as f: f.write(json.dumps(row, indent=2) + "\n")
            rows.append(row)
    result = {**analyse(rows, _controls()), "freeze": freeze, "runs": rows}
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def plan(output: Path) -> dict[str, Any]:
    if output.exists(): raise FileExistsError("output exists")
    output.mkdir(parents=True)
    result = _freeze(output); result.update({"schema": "causal-stage0-plan.v1", "status": "plan_only", "training_opened": False})
    (output / "plan.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    result = execute(args.output) if args.execute else plan(args.output)
    print(json.dumps({"schema": result["schema"], "output": str(args.output), "paid_calls": 0}))


if __name__ == "__main__": main()
