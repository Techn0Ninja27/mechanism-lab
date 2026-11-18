"""Second, independent matched-representation campaign harness.

This module deliberately does not read or resume the closed v5 campaign.  It
only plans and validates a new campaign directory.  Preparation and generation
are separate stages; a source-chunk fidelity gate is required before generation.
"""
from __future__ import annotations

import argparse, fcntl, hashlib, importlib, json, random
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from world_search import Client
from generation.causal_program import compile_program
from scripts.generate_causal_campaign import (ARMS, GRAMMAR, SYSTEM, TASK,
    canonical, digest, sha, strict_json, write_new, reserve, random_inventory,
    linear_controls, aa_calibration, parse_candidates, expected_request,
    request_hash, config as old_config)
from scripts.generate_causal_campaign import _plain_representation

MODEL_ARMS = ("task-only", "raw", "atoms", "distill")
CALLS_PER_ARM = 6
K = 4
CAP = 12.0
TOKENS = 12288
DEV_SEEDS = tuple(range(100, 116))
FINAL_SEEDS = tuple(range(200, 232))

def require_calibration_ready():
    folder=ROOT/'reports/chunk-field-calibration-v1'
    score=json.loads((folder/'score.json').read_text())
    freeze=json.loads((folder/'calibration-freeze.json').read_text())
    if score.get('freeze')!=freeze: raise ValueError('calibration freeze mismatch')
    if sha(folder/'reviewer-inputs.json')!=freeze.get('input_sha256') or sha(ROOT/'reports/generation-comparison-campaign2/reviewer-instructions.md')!=freeze.get('instructions_sha256'):
        raise ValueError('calibration input or instructions changed')
    reviews=score.get('reviews')
    if not isinstance(reviews,list) or len(reviews)!=2 or any(r.get('criterion_pass') is not True for r in reviews):
        raise ValueError('reviewer calibration failed; paid campaign is disabled')
    for item,name in zip(reviews,('reviewer-a.json','reviewer-b.json')):
        if sha(folder/name)!=item.get('file_sha256'):raise ValueError('calibration reviewer artifact changed')

def config(budget: float = CAP) -> dict[str, Any]:
    c = old_config(budget)
    c["budget_usd"] = budget
    c["shared_budget_usd"] = 30.0
    return c

def _prep():
    try:
        return importlib.import_module("scripts.prepare_chunk_representations")
    except ModuleNotFoundError as exc:
        raise ValueError("campaign2 requires scripts.prepare_chunk_representations") from exc

def cases() -> list[dict[str, Any]]:
    rng = random.Random(20261120)
    rows = []
    for i in range(CALLS_PER_ARM):
        block = list(MODEL_ARMS)
        rng.shuffle(block)
        rows.extend({"id": f"{arm}-{i:02d}", "arm": arm, "replicate": i} for arm in block)
    return rows

def _deps() -> dict[str, str]:
    paths = ["scripts/generate_causal_campaign2.py", "scripts/generate_causal_campaign.py",
             "scripts/prepare_matched_representations.py", "generation/api_budget.py", "scripts/world_search.py",
             "generation/causal_program.py", "scripts/evaluate_causal_candidates.py",
             "generation/campaign_selection.py", "reports/generation-comparison-campaign2/protocol.md",
             "reports/generation-comparison-v2/comparator-amendment.md",
             "reports/generation-comparison-v2/control-selection.json", "scripts/prepare_chunk_representations.py", "scripts/run_campaign2_preparation.py",
             "reports/generation-comparison-campaign2/reviewer-instructions.md",
             "generation/campaign2_decisions.py", "scripts/run_causal_campaign.py"]
    if any(not (ROOT / x).is_file() for x in paths):
        missing = [x for x in paths if not (ROOT / x).is_file()]
        raise ValueError(f"campaign2 dependency missing: {missing}")
    return {x: sha(ROOT / x) for x in paths}

def _source_view(pack: dict[str, Any]) -> dict[str, Any]:
    prep = _prep()
    if not hasattr(prep, "render_source_context"):
        raise ValueError("chunk preparation module lacks render_source_context")
    return prep.render_source_context(pack)

def _prep_prompt(pack: dict[str, Any], kind: str) -> str:
    prep = _prep()
    if hasattr(prep, "prompt"):
        return prep.prompt(pack, kind)
    if hasattr(prep, "_prompt"):
        return prep._prompt(pack, kind)
    raise ValueError("chunk preparation module lacks prompt builder")

def plan(pack_path: Path, out: Path) -> dict[str, Any]:
    if out.exists(): raise FileExistsError("choose a fresh campaign2 directory")
    pack = json.loads(pack_path.read_text())
    out.mkdir(parents=True)
    (out / "sourcepack.json").write_bytes(pack_path.read_bytes())
    prep = _prep()
    manifest = prep.prepare(pack_path, out / "preparation")
    random_rows = random_inventory(replicates=6)
    controls = linear_controls()
    doc = {"schema": "causal-generation-campaign2-plan.v1", "campaign": "generation-comparison-campaign2",
           "config": config(), "pack_sha256": sha(out / "sourcepack.json"),
           "preparation_manifest_sha256": sha(out / "preparation/manifest.json"),
           "dependency_hashes": _deps(), "task": TASK, "grammar": GRAMMAR,
           "cases": cases(), "candidate_capacity": K, "calls_per_arm": CALLS_PER_ARM,
           "dev_seeds": list(DEV_SEEDS), "final_seeds": list(FINAL_SEEDS),
           "preparation_cap_usd": 3.0, "campaign_cap_usd": CAP,
           "prep_draws_per_representation": 3,
           "source_support_rule": "claims and ports must be entailed by cited source chunks; quotes are pointers",
           "strong_control": "mean_sync selected from stage-0 and evaluated on final seeds",
           "random_inventory_sha256": digest({"candidates": random_rows}),
           "scope": "One new campaign; six independent calls per model arm; no novelty claim."}
    write_new(out / "random-inventory.json", {"candidates": random_rows})
    write_new(out / "controls.json", {"linear_controls": controls})
    write_new(out / "plan.json", doc)
    return doc

def verify(out: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    doc = json.loads((out / "plan.json").read_text())
    if doc.get("schema") != "causal-generation-campaign2-plan.v1": raise ValueError("campaign2 plan schema")
    if doc.get("config") != config(): raise ValueError("campaign2 config changed")
    if doc.get("dependency_hashes") != _deps(): raise ValueError("campaign2 dependency changed")
    if doc.get("pack_sha256") != sha(out / "sourcepack.json"): raise ValueError("source pack changed")
    if doc.get("cases") != cases() or doc.get("dev_seeds") != list(DEV_SEEDS) or doc.get("final_seeds") != list(FINAL_SEEDS): raise ValueError("campaign2 sampling changed")
    if doc.get("candidate_capacity") != K or doc.get("calls_per_arm") != CALLS_PER_ARM or doc.get("preparation_cap_usd") != 3.0: raise ValueError("campaign2 capacity changed")
    inv = json.loads((out / "random-inventory.json").read_text())
    if doc.get("random_inventory_sha256") != digest(inv): raise ValueError("random inventory changed")
    if len(inv.get("candidates", [])) != 24: raise ValueError("campaign2 random inventory must contain 24 controls")
    controls = json.loads((out / "controls.json").read_text())
    if len(controls.get("linear_controls", [])) != 4: raise ValueError("controls changed")
    pack = json.loads((out / "sourcepack.json").read_text())
    prompts = {"atoms": _prep_prompt(pack, "atoms"), "distillations": _prep_prompt(pack, "distillations")}
    manifest = json.loads((out / "preparation/manifest.json").read_text())
    for kind, text in prompts.items():
        if manifest.get("prompts", {}).get(kind, {}).get("sha256") != hashlib.sha256(text.encode()).hexdigest(): raise ValueError("preparation prompt changed")
    return doc, pack, prompts

def prepare(out: Path, execute: bool = False) -> dict[str, Any]:
    if execute: raise ValueError("use the sequential draw lifecycle; direct paid preparation is disabled")
    doc, pack, prompts = verify(out)
    target = out / "preparation-result.json"
    if target.exists(): raise FileExistsError("campaign2 preparation already exists")
    if not execute: return {"status": "plan_only", "maximum_reservation_usd": sum(reserve(x) for x in prompts.values()), "paid_calls": 0}
    if sum(reserve(x) for x in prompts.values()) > doc["preparation_cap_usd"]: raise ValueError("preparation exceeds campaign2 cap")
    prep = _prep(); client = Client(out, config(doc["preparation_cap_usd"])); result = {}
    for kind, validator_name in (("atoms", "validate_atoms"), ("distillations", "validate_distillations")):
        try:
            content = client.call({"id": f"prep-{kind}"}, 0, prompts[kind])
            val = getattr(prep, validator_name)(strict_json(content), pack)
            path = out / "representations" / f"{kind}.json"; write_new(path, val)
            result[kind] = {"status": "validated", "path": str(path.relative_to(out)), "sha256": sha(path)}
        except Exception as exc:
            result[kind] = {"status": "failed", "error": f"{type(exc).__name__}: {str(exc)[:400]}"}; break
    report = {"schema": "causal-generation-campaign2-preparation.v1", "status": "ready" if len(result) == 2 and all(x["status"] == "validated" for x in result.values()) else "failed", "representations": result, "plan_sha256": sha(out / "plan.json"), "semantic_gate_required": True}
    write_new(target, report); return report

def _verify_prepared(out: Path, pack: dict[str, Any], prompts: dict[str, str]) -> None:
    report = json.loads((out / "preparation-result.json").read_text())
    if report.get("status") != "ready" or report.get("plan_sha256") != sha(out / "plan.json"): raise ValueError("preparation not ready")
    prep = _prep()
    for kind in ("atoms", "distillations"):
        info = report.get("representations", {}).get(kind, {}); rel = info.get("path")
        if not isinstance(rel, str): raise ValueError("missing prepared representation")
        path = (out / rel).resolve()
        try: path.relative_to(out.resolve())
        except ValueError as exc: raise ValueError("prepared path escapes campaign") from exc
        if not path.is_file() or sha(path) != info.get("sha256"): raise ValueError("prepared representation hash mismatch")
        value = json.loads(path.read_text())
        call = (out / info.get("call_path", "")).resolve()
        try: call.relative_to(out.resolve())
        except ValueError as exc: raise ValueError("call path escapes campaign") from exc
        if not call.is_file() or sha(call) != info.get("call_sha256"): raise ValueError("prepared call receipt changed")
        if not call.is_file(): raise ValueError("missing preparation call record")
        rec = json.loads(call.read_text())
        if rec.get("status") != "complete" or not isinstance(rec.get("content"), str): raise ValueError("preparation call incomplete")
        validated = getattr(prep, f"validate_{kind}")(strict_json(rec["content"]), pack)
        if validated != value: raise ValueError("prepared content differs from call")
        req = expected_request(prompts[kind])
        if rec.get("request") != req or rec.get("request_hash") != request_hash(req): raise ValueError("preparation request hash mismatch")

def _require_chunk_gate(out: Path) -> None:
    gate = out / "chunk-fidelity-gate.json"
    if not gate.exists(): raise ValueError("campaign2 generation blocked: missing chunk-fidelity-gate")
    d = json.loads(gate.read_text())
    if d.get("schema") != "campaign2-dual-draw-gate.v1" or d.get("status") != "passed": raise ValueError("campaign2 generation blocked: selected dual-draw gate not passed")
    if d.get("criterion") != "cited_chunk_support.v1": raise ValueError("campaign2 generation blocked: wrong support criterion")
    if d.get("counts") != {"total": 24, "supported": 24, "unsupported": 0, "contradicted": 0, "indeterminate": 0}: raise ValueError("campaign2 generation blocked: incomplete chunk support")
    from scripts import run_campaign2_preparation as lifecycle
    if d.get("selection_sha256") != sha(out/"selected/selection.json"): raise ValueError("selection lock changed")
    lock=json.loads((out/"selected/selection.json").read_text())
    checked_pack=_prep()._pack(out/"sourcepack.json")
    for kind in ("atoms","distillations"):
        if lifecycle._selected(out,kind,checked_pack) != lock.get(kind): raise ValueError("selected draw no longer matches lock")
    reviews = d.get("reviews")
    if not isinstance(reviews, list) or len(reviews) != 4: raise ValueError("campaign2 generation blocked: exactly four selected review files required")
    prep = _prep()
    pack = json.loads((out / "sourcepack.json").read_text())
    atoms = json.loads((out / "representations/atoms.json").read_text())
    distill = json.loads((out / "representations/distillations.json").read_text())
    review_values = {"atoms": [], "distillations": []}
    for item in reviews:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str) or not isinstance(item.get("sha256"), str):
            raise ValueError("campaign2 generation blocked: malformed review binding")
        path = (out / item["path"]).resolve()
        try: path.relative_to(out.resolve())
        except ValueError as exc: raise ValueError("campaign2 review path escapes campaign") from exc
        if not path.is_file() or sha(path) != item["sha256"]: raise ValueError("campaign2 generation blocked: review hash")
        review = json.loads(path.read_text())
        if review.get("reviewer_id") != item["reviewer_id"]:
            raise ValueError("campaign2 generation blocked: reviewer identity mismatch")
        if item.get("reviewer_id") != review.get("reviewer_id") or item.get("artifact_sha256") != hashlib.sha256(json.dumps(review, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest():
            raise ValueError("campaign2 generation blocked: review identity/hash mismatch")
        review_values[item.get("kind")].append(review)
    if set(review_values) != {"atoms", "distillations"} or any(len(v) != 2 for v in review_values.values()):
        raise ValueError("campaign2 generation blocked: review kinds incomplete")
    for kind, vals in review_values.items():
        if vals[0].get("reviewer_id") == vals[1].get("reviewer_id"): raise ValueError("campaign2 generation blocked: reviewers are not distinct")
        for review in vals:
            checked = prep.validate_draw_review(kind, review, pack, atoms if kind == "atoms" else distill)
            if checked.get("status") != "passed": raise ValueError("campaign2 generation blocked: unsupported selected record")
    receipts = d.get("draw_receipts")
    if not isinstance(receipts, list) or len(receipts) != 2 or any(not isinstance(x, dict) or not isinstance(x.get("call_sha256"), str) for x in receipts):
        raise ValueError("campaign2 generation blocked: selected draw receipts missing")
    for receipt in receipts:
        call_path = out / str(receipt.get("call_path", ""))
        if not call_path.is_file() or sha(call_path) != receipt.get("call_sha256"):
            raise ValueError("campaign2 generation blocked: selected call receipt hash")

def generation_plan(out: Path) -> dict[str, Any]:
    doc, pack, prep_prompts = verify(out)
    _verify_prepared(out, pack, prep_prompts)
    prompts = {"task-only": "", "raw": "", "atoms": "", "distill": ""}
    prep = _prep(); view = _source_view(pack)
    raw = "\n\n".join(f"CHUNK {c['id']} (source={c['source_id']})\n{c['text']}" for c in view["chunks"])
    atoms = json.loads((out / "representations/atoms.json").read_text())
    distill = json.loads((out / "representations/distillations.json").read_text())
    def render(records, key):
        return _plain_representation("atoms" if key == "atoms" else "distillations", records, pack)
    common = "<<<TASK>>>\n" + TASK + "\n" + GRAMMAR + "\n<<<END TASK>>>\n\n"
    prompts["task-only"] = common + "REFERENCE MATERIAL: none"
    prompts["raw"] = common + "REFERENCE MATERIAL (source chunks)\n" + raw
    prompts["atoms"] = common + "REFERENCE MATERIAL (typed records)\n" + render(atoms, "atoms")
    prompts["distill"] = common + "REFERENCE MATERIAL (source-linked summaries)\n" + render(distill, "distillations")
    outdir = out / "prompts"; outdir.mkdir(exist_ok=True); files={}
    for arm, text in prompts.items():
        path=outdir/f"{arm}.txt"
        if path.exists() and path.read_text()!=text: raise ValueError("campaign2 prompt drift")
        if not path.exists(): path.write_text(text)
        files[arm]={"path":str(path.relative_to(out)),"sha256":sha(path),"chars":len(text)}
    requests=[]
    for case in doc["cases"]: requests.append({**case,"prompt_sha256":files[case['arm']]['sha256'],"reserved_usd":reserve(prompts[case['arm']])})
    report={"schema":"causal-generation-campaign2-requests.v1","plan_sha256":sha(out/'plan.json'),"requests":requests,"prompt_files":files,"reservation_usd":sum(x['reserved_usd'] for x in requests),"campaign_cap_usd":CAP}
    target = out/'generation-plan.json'
    if target.exists():
        if json.loads(target.read_text()) != report: raise ValueError("campaign2 generation plan drift")
    else: write_new(target, report)
    return report

def generate(out: Path, execute: bool = False) -> dict[str, Any]:
    if execute:
        from scripts.run_campaign2_preparation import _locked
        with _locked(out): return _generate(out, True)
    return _generate(out,False)

def _generate(out: Path, execute: bool = False) -> dict[str, Any]:
    if execute:
        # The semantic gate is checked before rebuilding prompts or constructing
        # a client, so a stopped attempt cannot spend or accidentally resume.
        _require_chunk_gate(out)
    req = generation_plan(out)
    if not execute: return req
    if req["reservation_usd"] + 3.0 > CAP: raise ValueError("campaign2 reservation exceeds cap")
    if (out/"inventory.json").exists(): raise FileExistsError("inventory exists; no regeneration")
    client=Client(out, config()); candidates=[]; prompts={k:(out/v['path']).read_text() for k,v in req['prompt_files'].items()}; stopped=False
    for case in req['requests']:
        try: candidates.extend(parse_candidates(client.call(case,0,prompts[case['arm']]),case))
        except Exception as exc:
            candidates.extend([{"id":f"{case['id']}-cand-{i:02d}","call_id":case['id'],"arm":case['arm'],"status":"invalid","program":None,"program_sha256":digest(None),"error":f"{type(exc).__name__}: {str(exc)[:400]}"} for i in range(K)])
            stopped=True
            break
    candidates.extend(json.loads((out/'random-inventory.json').read_text())["candidates"]); candidates.extend(json.loads((out/'controls.json').read_text())["linear_controls"])
    inv={"schema":"causal-candidate-inventory.v1","campaign":"generation-comparison-campaign2","status":"incomplete" if stopped else "generated","plan_sha256":sha(out/'plan.json'),"requests_sha256":sha(out/'generation-plan.json'),"candidates":candidates,"train_eligible":False,"novelty_established":False}
    write_new(out/'inventory.json',inv); return {"status":"generated","candidates":len(candidates)}

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('plan','prepare','generation-plan','generate'));p.add_argument('--pack',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
    if a.stage=='plan': result=plan(a.pack.resolve(),a.output.resolve())
    elif a.stage=='prepare': result=prepare(a.output.resolve(),a.execute)
    elif a.stage=='generation-plan': result=generation_plan(a.output.resolve())
    else: result=generate(a.output.resolve(),a.execute)
    print(json.dumps({k:v for k,v in result.items() if k not in ('requests','config')},indent=2))
if __name__=='__main__': main()
