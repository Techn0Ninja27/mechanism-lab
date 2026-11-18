"""Offline planner for a fixed-artifact, intention-to-treat comparison.

This module deliberately has no paid execution path.  It freezes inputs and
budgets for a possible follow-up while leaving the stopped Campaign 2 tree
untouched.
"""
from __future__ import annotations

import hashlib, json, shutil, random, math
from pathlib import Path
from typing import Any

from scripts import generate_causal_campaign2 as c2
from scripts.generate_causal_campaign import GRAMMAR, TASK, SYSTEM, _plain_representation, _source_header
from scripts.prepare_chunk_representations import render_source_context

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("task-only", "raw", "atoms", "distill")
DECODERS = {
    "opus-5.5": {"model": "anthropic/claude-opus-5.5", "prompt": 4e-6, "completion": 2e-5},
    "fable-5.2": {"model": "anthropic/claude-fable-5.2", "prompt": 8e-6, "completion": 4e-5},
}
CALLS_PER_ARM = 6
K = 4
CAP = 24.50
RAW_RENDER_UPPER_BYTES = 32406
PHASE_CAP = 30.0
STABLE_PLAN_SEED = 20261120
UNAVAILABLE_DISTILL_PROMPT = "<prose unavailable; generation slots remain planned failures>"

def sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def canon(value: Any) -> str: return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
def digest(value: Any) -> str: return hashlib.sha256(canon(value).encode()).hexdigest()
def text_sha(value: str) -> str: return hashlib.sha256(value.encode("utf-8")).hexdigest()

def source_pack() -> Path: return ROOT / "results/gnn-useful-proposal-v1/pack.json"
def stopped_atoms() -> Path: return ROOT / "results/generation-comparison-campaign2-run/prep-draws/atoms/00/atoms.json"
def stopped_atom_draw() -> Path: return ROOT / "results/generation-comparison-campaign2-run/prep-draws/atoms/00/draw.json"
def stopped_outcome() -> Path: return ROOT / "reports/generation-comparison-campaign2/outcome.json"
def frozen_prose_prompt() -> Path: return ROOT / "results/generation-comparison-campaign2-run/preparation/prompts.json"
def _code_hashes() -> dict[str, str]:
    names=("scripts/run_fixed_artifact_campaign.py","scripts/generate_causal_campaign2.py","scripts/generate_causal_campaign.py","scripts/evaluate_causal_candidates.py","scripts/run_causal_campaign.py","scripts/prepare_chunk_representations.py","scripts/prepare_matched_representations.py","generation/api_budget.py","generation/causal_program.py","generation/fixed_artifact_decisions.py","generation/fixed_artifact_fidelity.py","reports/fixed-artifact-comparison-v1/protocol.md","reports/fixed-artifact-comparison-v1/atoms-fidelity.json")
    return {n:sha(ROOT/n) for n in names}
def _phase_charged() -> float:
    ledger=json.loads((ROOT/"results/improvement-phase-2026-11-16/budget.json").read_text())
    total=0.0
    for rec in ledger.get("reservations",{}).values():
        value=rec.get("actual_usd",rec.get("reserved_usd",0.0)) if rec.get("status") in {"reconciled","over_budget"} else rec.get("reserved_usd",0.0)
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(float(value)) or float(value)<0: raise ValueError("phase ledger contains invalid charge")
        total += float(value)
    if not math.isfinite(total) or total < 0 or total > PHASE_CAP: raise ValueError("phase ledger charge exceeds cap")
    return float(total)
def _reserve_bytes(n: int, pricing: dict[str,float]) -> float:
    return (len(SYSTEM.encode("utf-8"))+n+512)*pricing["prompt"] + 12288*pricing["completion"]

def _raw_body(pack: dict[str, Any]) -> str:
    return "\n\n".join(f"CHUNK {x['id']} (source={x['source_id']})\n{x['text']}" for x in render_source_context(pack)["chunks"])

def _cases() -> list[dict[str, Any]]:
    """Freeze the protocol's seeded call order, including decoder order."""
    rng = random.Random(STABLE_PLAN_SEED)
    rows = []
    for case in c2.cases():
        decoders = list(DECODERS)
        rng.shuffle(decoders)
        for decoder in decoders:
            rows.append({**case, "decoder": decoder,
                         "id": f"{decoder}-{case['id']}",
                         "call_id": f"{decoder}-{case['id']}",
                         "candidate_capacity": K})
    return rows

def _prompt_bodies(out: Path, prose: dict[str, Any] | None) -> dict[str, str]:
    pack=json.loads((out/"sourcepack.json").read_text())
    common="<<<TASK>>>\n"+TASK+"\n"+GRAMMAR+"\n<<<END TASK>>>\n\n"
    header=_source_header(pack)
    bodies={"task-only":common+"REFERENCE MATERIAL: none", "raw":common+"REFERENCE MATERIAL\n"+header+"\n\nSOURCE EXCERPTS\n"+_raw_body(pack)}
    atoms=json.loads((out/"representations/atoms.json").read_text())
    bodies["atoms"]=common+"REFERENCE MATERIAL\n"+_plain_representation("atoms",atoms,pack)
    bodies["distill"] = (common+"REFERENCE MATERIAL\n"+
                          _plain_representation("distillations",prose,pack)
                          if prose is not None else UNAVAILABLE_DISTILL_PROMPT)
    return bodies

def _load_prose(path: Path, pack: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    if not path.exists(): return None, "unavailable"
    try:
        value=json.loads(path.read_text()); expected=c2._prep().validate_distillations(value,pack)
    except (ValueError, TypeError, KeyError, UnicodeError):
        return None, "unavailable"
    return expected, "validated"

def build_plan(out: Path, *, prose_path: Path | None = None) -> dict[str, Any]:
    if out.exists(): raise FileExistsError("fixed-artifact output already exists")
    stopped=stopped_atoms(); pack_path=source_pack(); outcome=stopped_outcome()
    if not all(x.is_file() for x in (stopped,pack_path,outcome)): raise ValueError("stopped artifacts missing")
    out.mkdir(parents=True); (out/"sourcepack.json").write_bytes(pack_path.read_bytes())
    rep=out/"representations";rep.mkdir(); shutil.copyfile(stopped,rep/"atoms.json")
    pack=json.loads((out/"sourcepack.json").read_text()); prose,prose_status=_load_prose(prose_path or Path("__missing_prose__"),pack)
    if prose_path is not None and prose_path.is_file(): shutil.copyfile(prose_path,rep/"distillations-raw.json")
    if prose is not None:(rep/"distillations.json").write_text(json.dumps(prose,ensure_ascii=False,indent=2)+"\n")
    prompt_source=frozen_prose_prompt(); prompt_doc=json.loads(prompt_source.read_text()); prose_prompt=prompt_doc["distillations"]
    fidelity={"atoms":{"status":"gate_failed_reused_unselected","representation_sha256":sha(rep/"atoms.json"),"draw_sha256":sha(stopped_atom_draw()),"outcome_sha256":sha(outcome)},"distill":{"status":prose_status,"path":str(prose_path) if prose_path else None}}
    fidelity_source=ROOT/"reports/fixed-artifact-comparison-v1/atoms-fidelity.json"
    fidelity["atoms"]["measurement_table_sha256"]=sha(fidelity_source)
    (out/"fidelity.json").write_text(json.dumps(fidelity,indent=2)+"\n")
    shutil.copyfile(fidelity_source,out/"atoms-fidelity.json")
    bodies=_prompt_bodies(out,prose)
    # One fixed case order with seeded decoder order within each case.
    cases=_cases()
    random=json.loads((ROOT/"results/generation-comparison-campaign2-run/random-inventory.json").read_text())
    controls=json.loads((ROOT/"results/generation-comparison-campaign2-run/controls.json").read_text())
    prose_prompt_input=len(prose_prompt.encode("utf-8")); prompt_bytes={k:len(v.encode("utf-8")) for k,v in bodies.items()}
    prep_reserve=_reserve_bytes(prose_prompt_input,DECODERS["opus-5.5"])
    generation_reserve=sum(6*_reserve_bytes(RAW_RENDER_UPPER_BYTES if arm == "distill" and prose is None else prompt_bytes[arm], pricing) for pricing in DECODERS.values() for arm in ARMS)
    prompt_dir=out/"prompts"; prompt_dir.mkdir()
    for arm, body in bodies.items(): (prompt_dir/f"{arm}.txt").write_text(body)
    (prompt_dir/"prose-preparation.txt").write_text(prose_prompt)
    plan={"schema":"fixed-artifact-comparison-plan.v1","status":"offline_plan_only","canonical_output":"results/fixed-artifact-comparison-v1","status_requires_prose":True,"source_pack_sha256":sha(out/"sourcepack.json"),"stopped_atom_sha256":sha(rep/"atoms.json"),"stopped_atom_draw_sha256":sha(stopped_atom_draw()),"stopped_outcome_sha256":sha(outcome),"prose_prompt_file_sha256":sha(prompt_source),"prose_prompt_body_sha256":text_sha(prose_prompt),"prose_prompt_input_bytes":prose_prompt_input,"raw_render_upper_bytes":RAW_RENDER_UPPER_BYTES,"prose_status":prose_status,"prose_raw_sha256":sha(rep/"distillations-raw.json") if (rep/"distillations-raw.json").exists() else None,"prose_artifact_sha256":sha(rep/"distillations.json") if prose is not None else None,"fidelity_sha256":sha(out/"fidelity.json"),"task_sha256":text_sha(TASK),"grammar_sha256":text_sha(GRAMMAR),"system_sha256":text_sha(SYSTEM),"code_hashes":_code_hashes(),"decoders":DECODERS,"provider_max_output_tokens":12288,"reasoning":"low","arms":ARMS,"calls_per_arm":CALLS_PER_ARM,"candidate_capacity":K,"cases":cases,"case_seed":STABLE_PLAN_SEED,"random_inventory_sha256":digest(random),"controls_sha256":digest(controls),"random_inventory_file_sha256":sha(ROOT/"results/generation-comparison-campaign2-run/random-inventory.json"),"controls_file_sha256":sha(ROOT/"results/generation-comparison-campaign2-run/controls.json"),"prompt_hashes":{k:text_sha(v) for k,v in bodies.items()},"prompt_bytes":prompt_bytes,"dev_seeds":list(range(100,116)),"final_seeds":list(range(200,232)),"phase_charged_before_usd":_phase_charged(),"phase_cap_usd":PHASE_CAP,"campaign_cap_usd":CAP,"prose_preparation_reserve_usd":prep_reserve,"generation_reserve_upper_usd":generation_reserve,"worst_case_campaign_reserve_usd":prep_reserve+generation_reserve,"phase_worst_case_charged_usd":_phase_charged()+prep_reserve+generation_reserve,"paid_calls":0,"generation_enabled":False,"scope":"Fixed-artifact intention-to-treat pilot; descriptive, no novelty or causal representation claim."}
    if plan["worst_case_campaign_reserve_usd"] > CAP: raise ValueError("fixed-artifact cap exceeded")
    if plan["phase_worst_case_charged_usd"] > PHASE_CAP: raise ValueError("phase cap exceeded")
    (out/"plan.json").write_text(json.dumps(plan,indent=2)+"\n");(out/"prompt-hashes.json").write_text(json.dumps({k:text_sha(v) for k,v in bodies.items()},indent=2)+"\n")
    (out/"random-inventory.json").write_bytes((ROOT/"results/generation-comparison-campaign2-run/random-inventory.json").read_bytes());(out/"controls.json").write_bytes((ROOT/"results/generation-comparison-campaign2-run/controls.json").read_bytes())
    return plan

def verify_plan(out: Path) -> dict[str, Any]:
    d=json.loads((out/"plan.json").read_text());
    if d.get("generation_enabled") is not False or d.get("paid_calls")!=0 or d.get("canonical_output") != "results/fixed-artifact-comparison-v1": raise ValueError("offline plan metadata changed")
    if len(d.get("cases",[])) != 48: raise ValueError("expected 48 decoder-arm call slots")
    if d.get("decoders") != DECODERS or d.get("provider_max_output_tokens") != 12288 or d.get("reasoning") != "low" or d.get("arms") != list(ARMS) or d.get("calls_per_arm") != CALLS_PER_ARM or d.get("candidate_capacity") != K: raise ValueError("decoder configuration changed")
    if d.get("system_sha256") != text_sha(SYSTEM): raise ValueError("system prompt changed")
    if d.get("raw_render_upper_bytes") != RAW_RENDER_UPPER_BYTES or d.get("dev_seeds") != list(range(100,116)) or d.get("final_seeds") != list(range(200,232)): raise ValueError("sampling metadata changed")
    for arm in ARMS:
        if sum(x["arm"]==arm for x in d["cases"]) != 12: raise ValueError("arm denominator changed")
    if d["source_pack_sha256"] != sha(out/"sourcepack.json") or d["stopped_atom_sha256"] != sha(out/"representations/atoms.json"): raise ValueError("immutable input changed")
    if d.get("stopped_atom_draw_sha256") != sha(stopped_atom_draw()) or d.get("stopped_outcome_sha256") != sha(stopped_outcome()): raise ValueError("stopped provenance changed")
    if d.get("prose_prompt_file_sha256") != sha(frozen_prose_prompt()): raise ValueError("prose prompt changed")
    if d.get("phase_cap_usd") != PHASE_CAP or d.get("campaign_cap_usd") != CAP: raise ValueError("budget caps changed")
    if d.get("code_hashes") != _code_hashes(): raise ValueError("code dependency changed")
    random_path=out/"random-inventory.json"; controls_path=out/"controls.json"
    if not random_path.is_file() or not controls_path.is_file(): raise ValueError("copied controls missing")
    random=json.loads(random_path.read_text()); controls=json.loads(controls_path.read_text())
    random_source=ROOT/"results/generation-comparison-campaign2-run/random-inventory.json"; controls_source=ROOT/"results/generation-comparison-campaign2-run/controls.json"
    if (d.get("random_inventory_file_sha256") != sha(random_source) or d.get("controls_file_sha256") != sha(controls_source) or sha(random_path) != sha(random_source) or sha(controls_path) != sha(controls_source) or d.get("random_inventory_sha256") != digest(random) or d.get("controls_sha256") != digest(controls)): raise ValueError("control inputs changed")
    if d.get("task_sha256") != text_sha(TASK) or d.get("grammar_sha256") != text_sha(GRAMMAR): raise ValueError("task or grammar changed")
    if d.get("case_seed") != STABLE_PLAN_SEED or d.get("cases") != _cases(): raise ValueError("case order changed")
    if d.get("prose_status") not in {"unavailable", "validated"}: raise ValueError("invalid prose status")
    for decoder in DECODERS:
        for arm in ARMS:
            rows=[x for x in d["cases"] if x["decoder"]==decoder and x["arm"]==arm]
            if len(rows) != CALLS_PER_ARM or len({x["id"] for x in rows}) != CALLS_PER_ARM: raise ValueError("case uniqueness changed")
    raw_path=out/"representations/distillations-raw.json"
    if d.get("prose_raw_sha256") != (sha(raw_path) if raw_path.exists() else None): raise ValueError("raw prose changed")
    prose = None
    if d.get("prose_status") == "validated":
        path=out/"representations/distillations.json"
        if not path.is_file(): raise ValueError("validated prose artifact missing")
        pack=json.loads((out/"sourcepack.json").read_text())
        prose=json.loads(path.read_text())
        if d.get("prose_artifact_sha256") != sha(path) or prose.get("schema") != "chunk-grounded-distillations.v1" or len(prose.get("distillations",[])) != 12: raise ValueError("validated prose artifact changed")
    bodies=_prompt_bodies(out,prose)
    prompt_dir=out/"prompts"
    if not prompt_dir.is_dir(): raise ValueError("prompt files missing")
    for arm, body in bodies.items():
        path=prompt_dir/f"{arm}.txt"
        if not path.is_file() or path.read_text()!=body or d.get("prompt_hashes",{}).get(arm) != text_sha(body): raise ValueError("prompt changed")
    prose_prompt=json.loads(frozen_prose_prompt().read_text())["distillations"]
    prep_path=prompt_dir/"prose-preparation.txt"
    if not prep_path.is_file() or prep_path.read_text()!=prose_prompt: raise ValueError("preparation prompt changed")
    if d.get("prose_prompt_body_sha256") != text_sha(prose_prompt) or d.get("prose_prompt_input_bytes") != len(prose_prompt.encode("utf-8")): raise ValueError("preparation prompt binding changed")
    prep_reserve=_reserve_bytes(len(prose_prompt.encode()),DECODERS["opus-5.5"])
    generation_reserve=sum(6*_reserve_bytes(RAW_RENDER_UPPER_BYTES if arm == "distill" and prose is None else len(bodies[arm].encode()), pricing) for pricing in DECODERS.values() for arm in ARMS)
    if d.get("prompt_bytes") != {k:len(v.encode("utf-8")) for k,v in bodies.items()}: raise ValueError("prompt byte metadata changed")
    if abs(d.get("prose_preparation_reserve_usd",-1)-prep_reserve)>1e-12 or abs(d.get("generation_reserve_upper_usd",-1)-generation_reserve)>1e-12 or d.get("worst_case_campaign_reserve_usd") != prep_reserve+generation_reserve or d.get("phase_worst_case_charged_usd") != d.get("phase_charged_before_usd")+prep_reserve+generation_reserve or d.get("phase_charged_before_usd") != _phase_charged() or d["worst_case_campaign_reserve_usd"] > CAP or d["phase_worst_case_charged_usd"] > PHASE_CAP: raise ValueError("budget reservation changed or exceeds cap")
    if d["fidelity_sha256"] != sha(out/"fidelity.json"): raise ValueError("fidelity snapshot changed")
    fidelity_source=ROOT/"reports/fixed-artifact-comparison-v1/atoms-fidelity.json"
    if not (out/"atoms-fidelity.json").is_file() or sha(out/"atoms-fidelity.json") != sha(fidelity_source): raise ValueError("fidelity table changed")
    return d

if __name__ == "__main__": raise SystemExit("offline module: no execution command")
