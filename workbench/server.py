#!/usr/bin/env python3
"""Local-only HTTP API and static server for the Alien Science workbench.

The API exposes local research state and allowlisted engine operations. Grounded
generation uses selected source packs and a shared capped API ledger; arbitrary
shell commands and caller-supplied filesystem paths are never accepted.
Run with ``python server.py`` from the experiment directory.
"""
from __future__ import annotations

import http.server
import hashlib
import json
import mimetypes
import os
import argparse
import functools
import math
import sys
import shutil
import zipfile
import pathlib
import signal
import subprocess
import threading
import time
import uuid
from urllib.parse import parse_qs, urlparse, unquote

ROOT = pathlib.Path(os.environ.get("ALIEN_WORKSPACE", pathlib.Path(__file__).resolve().parents[1])).resolve()
ENGINE = pathlib.Path(os.environ.get("ALIEN_REPO", ROOT / "upstream" if (ROOT / "upstream/alien.py").exists() else ROOT)).resolve()
WEB = pathlib.Path(__file__).resolve().parent
RESULTS = ROOT / "results"
JOBS_DIR = ROOT / "workspace" / ".app" / "jobs"
ADJUDICATIONS = ROOT / "workspace" / ".app" / "adjudications.json"
JOBS_DIR.mkdir(parents=True, exist_ok=True)
HOST = os.environ.get("ALIEN_SCIENCE_HOST", "127.0.0.1")
PORT = int(os.environ.get("ALIEN_SCIENCE_PORT", "8765"))

LOCK = threading.RLock()
JOBS: dict[str, dict] = {}
PROCESSES: dict[str, subprocess.Popen] = {}
legacy_jobs = WEB / "jobs"
if legacy_jobs.exists():
    for legacy in legacy_jobs.glob("*.json"):
        target = JOBS_DIR / legacy.name
        if not target.exists(): shutil.copyfile(legacy, target)
if not ADJUDICATIONS.exists() and (WEB / "adjudications.json").exists():
    shutil.copyfile(WEB / "adjudications.json", ADJUDICATIONS)
for saved in JOBS_DIR.glob("*.json"):
    try:
        record = json.loads(saved.read_text())
        if record.get("status") in {"running", "queued", "cancel_requested"}: record.update(status="interrupted", reason="Server restarted; inspect outputs before rerunning")
        JOBS[record["id"]] = record
    except (ValueError, KeyError, OSError): pass
ALLOWED_ACTIONS = {"audit_provenance", "sample_offline", "rerank_offline", "prompt_export", "inventory", "grounded_execute", "decode_paid"}


def read_json(path: pathlib.Path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def _safe_engine_relative(value):
    """Return an engine-relative artifact path, or None for unsafe values.

    The operator catalog is data-only. In particular, report paths are never
    imported, opened for execution, or allowed to escape the engine checkout.
    """
    if not isinstance(value, str) or not value or pathlib.PurePosixPath(value).is_absolute():
        return None
    parts = pathlib.PurePosixPath(value).parts
    if any(part in {"", ".", ".."} for part in parts):
        return None
    target = (ENGINE / pathlib.Path(*parts)).resolve()
    engine_root = ENGINE.resolve()
    if target != engine_root and engine_root not in target.parents:
        return None
    return "/".join(parts)


def operator_library():
    """Read the hand-curated operator catalog as inert JSON data.

    Missing or malformed catalogs intentionally produce an empty, clearly
    bounded response. No catalog field is treated as Python/module input.
    """
    catalog_path = ENGINE / "reports" / "operator-composition" / "catalog.json"
    raw = read_json(catalog_path, {})
    if not isinstance(raw, dict):
        raw = {}
    operators = []
    for item in raw.get("operators", []):
        if not isinstance(item, dict):
            continue
        operators.append({
            "id": item.get("id"),
            "content_id": item.get("content_id"),
            "input_type": item.get("input_type"),
            "output_type": item.get("output_type"),
            "source_url": item.get("source_url"),
            "assumptions": item.get("assumptions", []),
            "contract_status": item.get("contract_status"),
        })
    chains = []
    for item in raw.get("chains", []):
        if not isinstance(item, dict):
            continue
        chains.append({
            "id": item.get("id"),
            "operator_ids": item.get("operator_ids", []),
            "input_type": item.get("input_type"),
            "output_type": item.get("output_type"),
            "molecule_id": item.get("molecule_id"),
        })
    study = raw.get("study") if isinstance(raw.get("study"), dict) else {}
    report_path = _safe_engine_relative(study.get("report_path"))
    return {
        "schema_version": raw.get("schema_version"),
        "scope": raw.get("scope"),
        "operators": operators,
        "chains": chains,
        "study": {"report_path": report_path, "status": study.get("status")},
    }


def run_operator_check(body):
    """Validate and run one bounded, local operator chain."""
    if not isinstance(body, dict):
        raise ValueError("JSON object required")
    operator_ids = body.get("operator_ids")
    values = body.get("values")
    if not isinstance(operator_ids, list) or not 1 <= len(operator_ids) <= 4 or any(not isinstance(x, str) or not x for x in operator_ids):
        raise ValueError("operator_ids must contain 1-4 known operator IDs")
    if not isinstance(values, list) or not 1 <= len(values) <= 4096:
        raise ValueError("values must contain 1-4096 signed 64-bit integers")
    if any(isinstance(x, bool) or not isinstance(x, int) or x < -(2**63) or x > 2**63 - 1 for x in values):
        raise ValueError("values must contain only signed 64-bit integers")
    catalog_ids = {item.get("id") for item in operator_library().get("operators", [])}
    if any(operator_id not in catalog_ids for operator_id in operator_ids):
        raise ValueError("unknown operator ID")
    # Fixed trusted engine import; caller input is data, never a module/path.
    if str(ENGINE) not in sys.path:
        sys.path.insert(0, str(ENGINE))
    from generation.operator_checks import check_chain
    digest = hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()
    with LOCK:
        result = check_chain(operator_ids, values)
        if not isinstance(result, dict):
            raise ValueError("operator check returned invalid JSON")
        artifact_id = uuid.uuid4().hex
        artifact_dir = ROOT / "results" / "operator-checks"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        record = {"id": artifact_id, "created_at": time.time(), "input_sha256": digest, "result": result}
        (artifact_dir / f"{artifact_id}.json").write_text(json.dumps(record, indent=2, ensure_ascii=False))
    return record


def public_path(path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return path.name



def engine_python():
    choices = [ROOT / ".venv/bin/python", ENGINE / ".venv/bin/python", ENGINE.parent / ".venv/bin/python"]
    return str(next((p for p in choices if p.exists()), pathlib.Path(sys.executable)))


@functools.lru_cache(maxsize=1)
def get_workspace():
    from workspace import Workspace
    return Workspace(ROOT / "workspace")


def setup_info():
    checks = [{"name": "Engine source", "path": str(ENGINE / "alien.py"), "ready": (ENGINE / "alien.py").exists()},
              {"name": "Original source catalogue", "path": str(ENGINE / "papers/clusters.json"), "ready": (ENGINE / "papers/clusters.json").exists()},
              {"name": "Coherence checkpoint", "path": str(ENGINE / "models/coherence_model/model.pt"), "ready": (ENGINE / "models/coherence_model/model.pt").is_file()},
              {"name": "Availability checkpoint", "path": str(ENGINE / "models/availability_model/model.pt"), "ready": (ENGINE / "models/availability_model/model.pt").is_file()}]
    return {"engine_ready": all(c["ready"] for c in checks), "python": engine_python(), "checks": checks,
            "paths": {"workspace": str(ROOT / "workspace"), "engine": str(ENGINE), "results": str(RESULTS), "jobs": str(JOBS_DIR)},
            "folder_guide": [{"folder": "sources/", "purpose": "Imported source documents, preserved as supplied. Import does not train the engine."}, {"folder": "notes/", "purpose": "Your reading notes and working hypotheses."}, {"folder": "reviews/", "purpose": "Your evidence comparisons and decisions."}, {"folder": "runs/", "purpose": "References to engine jobs, whose immutable outputs live in results/webapp-jobs/."}, {"folder": "exports/", "purpose": "Portable ZIP exports of a project."}],
            "operations": {"documents": True, "sample": all(c["ready"] for c in checks), "paid_decode": False, "training": False}}

def provenance():
    audit = read_json(RESULTS / "provenance-repair-2026-11-16" / "audit.json", {}) or {}
    return {
        "status": "quarantined",
        "ranking_claim_allowed": False,
        "decode_allowed": False,
        "message": "Released scorer rankings and old drafts have unresolved source lineage. Repaired links are for inspection only.",
        "audit": audit,
    }



def run_path(run):
    base = RESULTS
    if run.startswith("upstream:"):
        base = ENGINE / "results"
        run = run.split(":", 1)[1]
    target = (base / run).resolve()
    if base.resolve() not in target.parents or any(x.startswith(".") for x in pathlib.Path(run).parts):
        raise ValueError("invalid run")
    return target


@functools.lru_cache(maxsize=1)
def canonical_atoms():
    return (read_json(ENGINE / "papers/clusters.json", {}) or {}).get("clusters", {})

def candidate_rows(run="proposals-v2"):
    run_root = run_path(run)
    selected = read_json(run_root / "selection.json", {}) or {}
    if not selected:
        scored = read_json(run_root / "k3" / "scored_candidates.json", {}) or read_json(run_root / "scored_candidates.json", {}) or {}
        selected = {"candidates": scored.get("candidates", [])}
        if not selected["candidates"]:
            ranked = read_json(run_root / "rerank.json", {}) or {}
            selected["candidates"] = ranked.get("top_k_sequences", [])
    rows = selected.get("candidates", [])
    repaired = read_json(RESULTS / "provenance-repair-2026-11-16" / "selected-evidence.repaired.json", {}) or {}
    atoms = repaired.get("atoms", {})
    # Keep the selected run useful even when its old selection artifact is partial.
    out = []
    for i, row in enumerate(rows):
        item = dict(row)
        item["index"] = i
        item.setdefault("id", "-".join(map(str, item.get("cluster_ids", []))))
        item["run"] = run
        ids = [str(x) for x in item.get("cluster_ids", [])]
        item["atoms"] = [{"id": cid, "description": (atoms.get(cid) or {}).get("canonical_description") or canonical_atoms().get(cid, {}).get("name", "Unknown atom"),
                          "assignment_status": (atoms.get(cid) or {}).get("assignment_status", "unverified")} for cid in ids]
        item["status"] = "draft available" if (run_root / item.get("id", "") / "proposal.md").exists() else "not decoded"
        item["validity"] = "untrusted semantics"
        out.append(item)
    return out


def runs():
    result = []
    if not RESULTS.exists():
        return result
    for d in sorted(RESULTS.iterdir(), key=lambda x: x.stat().st_mtime if x.exists() else 0, reverse=True):
        if not d.is_dir() or d.name == "webapp-jobs":
            continue
        config = read_json(d / "run.json") or read_json(d / "beta_sweep_summary.json") or {}
        result.append({"id": d.name, "path": public_path(d), "config": config, "has_report": (d / "report.html").exists(),
                       "has_selection": (d / "selection.json").exists(), "modified": d.stat().st_mtime})
    jobs_root = RESULTS / "webapp-jobs"
    if jobs_root.exists():
        for d in sorted(jobs_root.iterdir(), reverse=True):
            if d.is_dir():
                result.insert(0, {"id": "webapp-jobs/" + d.name, "path": public_path(d), "config": read_json(d / "run.json", {}), "has_report": (d / "report.html").exists(), "has_selection": False, "modified": d.stat().st_mtime})
    upstream_results = ENGINE / "results"
    if upstream_results.exists():
        for d in sorted(upstream_results.iterdir(), key=lambda x: x.stat().st_mtime if x.exists() else 0, reverse=True):
            if not d.is_dir(): continue
            if not ((d / "scored_candidates.json").exists() or (d / "k3" / "scored_candidates.json").exists()): continue
            result.append({"id": "upstream:" + d.name, "path": public_path(d), "config": read_json(d / "run.json") or read_json(d / "k3" / "results.json") or {}, "has_report": False, "has_selection": False, "modified": d.stat().st_mtime})
    return result


def adjudications():
    return read_json(ADJUDICATIONS, []) or []


def inventory():
    props = list((RESULTS / "proposals-v2").glob("*/proposal.md")) if (RESULTS / "proposals-v2").exists() else []
    evidence = list((RESULTS / "proposals-v2" / "retrieval").glob("*-evidence.json")) if (RESULTS / "proposals-v2" / "retrieval").exists() else []
    return {"runs": runs(), "counts": {"proposals": len(props), "evidence_sets": len(evidence), "candidates": len(candidate_rows()),
            "source_units_repaired": len((read_json(RESULTS / "provenance-repair-2026-11-16" / "selected-evidence.repaired.json", {}) or {}).get("atoms", {}))},
            "provenance": provenance()}


def _study_root():
    """Return the checked-out study directory, if the fork is available."""
    root = ENGINE / "reconstruction-studies"
    return root if root.is_dir() else ENGINE / "candidate-studies"


def study_rows():
    """Build a read-only registry from the committed study artifacts.

    This deliberately reports the authors' raw verdict separately from the app's
    audit qualification. A score or ``supported_pilot`` label never becomes a
    novelty claim automatically.
    """
    if str(ENGINE / "scripts") not in sys.path:
        sys.path.insert(0, str(ENGINE / "scripts"))
    from export_research_ledger import build_ledger
    ledger = build_ledger(ENGINE)
    rows = []
    for item in ledger.get("studies", []):
        cohort = item.get("cohort", "unknown")
        rows.append({
            "id": item.get("id"), "kind": cohort,
            "path": ("upstream:" + ("candidate-studies/" if cohort == "candidate" else "reconstruction-studies/") + str(item.get("id"))),
            "raw_verdict": item.get("raw_verdict", "unknown"),
            "audit_qualification": item.get("audit_qualification", "unknown"),
            "audit_note": item.get("audit_note"),
            "novelty_established": bool(item.get("novelty_established", False)),
            "human_adjudicated": bool(item.get("human_adjudicated", False)),
            "primary": item.get("primary_metrics", {}),
            "secondary": item.get("secondary_metrics", {}),
            "limitations": item.get("limitations", [])[:8],
            "prior_art_status": item.get("prior_art_status") or "",
            "proposal_title": item.get("title") or "",
            "artifacts": [pathlib.PurePosixPath(a.get("path", "")).name for a in item.get("artifact_refs", []) if isinstance(a, dict) and a.get("path")],
        })
    return rows


def study_by_id(study_id):
    if not isinstance(study_id, str) or study_id in {"", ".", ".."} or any(part in {".", ".."} for part in pathlib.PurePosixPath(study_id).parts):
        raise ValueError("invalid study")
    return next((row for row in study_rows() if row["id"] == study_id), None)


def grounded_rows():
    """Expose completed grounded runs without source payloads or call receipts."""
    rows = []
    links = read_json(ENGINE / "reports" / "improvement-2026-11-16" / "evaluation-links.json", {}) or {}

    def operator_check(run_id):
        record = links.get(run_id) if isinstance(links, dict) else None
        if not isinstance(record, dict):
            return {"status": "unavailable", "verified": False, "summary": "No reviewed operator evaluation has been linked."}
        result_ref = record.get("result_path")
        report_ref = record.get("report_path")
        refs = (result_ref, report_ref)
        if any(not isinstance(x, str) or pathlib.PurePosixPath(x).is_absolute() or not x.startswith("operator-studies/") for x in refs):
            return {"status": "unverified", "verified": False, "summary": "Evaluation links failed the operator-studies path policy."}
        result_path = (ENGINE / result_ref).resolve(); report_path = (ENGINE / report_ref).resolve()
        operator_root = (ENGINE / "operator-studies").resolve()
        if any(operator_root not in p.parents or not p.is_file() for p in (result_path, report_path)):
            return {"status": "unverified", "verified": False, "summary": "Linked operator artifacts are missing or outside the allowed root."}
        recorded_hash = record.get("result_sha256")
        actual_hash = hashlib.sha256(result_path.read_bytes()).hexdigest()
        verified = isinstance(recorded_hash, str) and recorded_hash == actual_hash
        return {"status": record.get("status", "unverified") if verified else "unverified", "verified": verified,
                "summary": str(record.get("summary", ""))[:1000] if verified else "Result hash does not match the reviewed operator artifact.",
                "result_path": result_ref, "report_path": report_ref,
                "result_sha256": actual_hash}
    candidates = [(p, p.parent.parent.name, None) for p in (ENGINE / "results").glob("improvement-phase-*/grounded-*/result.json")]
    workspace_root = (ROOT / "workspace" / "projects").resolve()
    if workspace_root.is_dir():
        for p in workspace_root.glob("*/runs/sourcepack-*/execution/result.json"):
            resolved = p.resolve()
            if workspace_root in resolved.parents:
                candidates.append((resolved, "project", p.parts[-5] if len(p.parts) >= 5 else None))
    for result_path, phase, project_id in sorted(candidates, key=lambda x: str(x[0])):
        data = read_json(result_path, {}) or {}
        proposal = data.get("proposal") if isinstance(data.get("proposal"), dict) else {}
        review = data.get("review") if isinstance(data.get("review"), dict) else {}
        evidence = [{"chunk_id": e.get("chunk_id"), "quote": str(e.get("quote", ""))[:500], "supports": str(e.get("supports", ""))[:500]}
                    for e in proposal.get("evidence", [])[:12] if isinstance(e, dict)]
        budget_path = (ENGINE / "results" / "improvement-phase-2026-11-16" / "budget.json") if phase == "project" else result_path.parent.parent / "budget.json"
        budget = read_json(budget_path, {}) or {}
        reservations = budget.get("reservations", {}) if isinstance(budget, dict) else {}
        def money(value):
            try:
                number = float(value or 0)
                return number if math.isfinite(number) and number >= 0 else 0.0
            except (TypeError, ValueError):
                return 0.0
        spent = sum(money(x.get("actual_usd")) for x in reservations.values() if isinstance(x, dict))
        reserved = sum(money(x.get("reserved_usd")) for x in reservations.values() if isinstance(x, dict) and x.get("status") not in {"released", "reconciled", "over_budget"})
        rows.append({"id": result_path.parent.name if phase != "project" else result_path.parent.parent.name, "phase": phase,
                     "project_id": project_id,
                     "status": data.get("status", "unknown"), "machine_review_only": True,
                     "empirical_result": False, "novelty_established": bool(data.get("novelty_established", False)),
                     "pack_hash": data.get("pack_hash"), "proposal": {
                         "title": proposal.get("title"), "hypothesis": proposal.get("hypothesis"),
                         "baseline": proposal.get("baseline"), "operator_delta": proposal.get("operator_delta"),
                         "matched_control": proposal.get("matched_control"), "evidence": evidence,
                         "experiment": proposal.get("experiment")},
                     "retrieval_coverage": (data.get("retrieval") or {}).get("coverage"),
                     "retrieved_source_count": data.get("retrieved_source_count"),
                     "retrieval_failures": (data.get("retrieval") or {}).get("failures",[]),
                     "lineage_coverage": (data.get("lineage") or {}).get("coverage"),
                     "lineage_warnings": (data.get("lineage") or {}).get("warnings",[]),
                     "review": {k: review.get(k) for k in ("decision", "failure_type", "reason", "strongest_counterargument", "required_test")},
                     "budget": {"cap_usd": budget.get("cap_usd"), "actual_usd": spent, "active_reserved_usd": reserved},
                     "artifact": public_path(result_path),
                     "operator_evaluation": operator_check(result_path.parent.name) if phase == "improvement-phase-2026-11-16" else {"status":"unavailable","verified":False,"summary":"No operator check linked for this run."}})
    return rows


def build_project_source_pack(project_id, body):
    """Create an offline, explicitly selected source pack and research plan."""
    if not isinstance(body, dict) or not isinstance(body.get("document_ids"), list) or not body["document_ids"]:
        raise ValueError("document_ids must be a non-empty list")
    question = str(body.get("question", "")).strip()
    if not question or len(question) > 2000:
        raise ValueError("question must be 1-2000 characters")
    # Prior-art retrieval is opt-in and bounded.  An absent query list must
    # never accidentally enable a network search through a truthy flag.
    raw_queries = body.get("prior_art_queries", [])
    if raw_queries is None:
        raw_queries = []
    if not isinstance(raw_queries, list):
        raise ValueError("prior_art_queries must be a list")
    if len(raw_queries) > 3:
        raise ValueError("at most 3 prior-art queries are allowed")
    queries = []
    for query in raw_queries:
        if not isinstance(query, str):
            raise ValueError("each prior-art query must be a string")
        query = query.strip()
        if not query or len(query) > 200:
            raise ValueError("each prior-art query must be 1-200 characters")
        queries.append(query)
    allow_arxiv = body.get("allow_arxiv", False)
    if not isinstance(allow_arxiv, bool):
        raise ValueError("allow_arxiv must be a boolean")
    if allow_arxiv and not queries:
        raise ValueError("allow_arxiv requires at least one prior-art query")
    search_config = {"queries": queries, "allow_arxiv": bool(allow_arxiv and queries)}
    workspace = get_workspace()
    project = workspace.get_project(project_id)
    docs = {d["id"]: d for d in workspace.list_documents(project_id)}
    selected = []
    for doc_id in body["document_ids"]:
        if not isinstance(doc_id, str) or doc_id not in docs:
            raise ValueError("unknown selected document")
        doc = docs[doc_id]
        if doc["kind"] != "text" or pathlib.Path(doc["relative_path"]).suffix.lower() not in {".md", ".txt", ".json", ".py"}:
            raise ValueError("source packs accept selected UTF-8 text, Markdown, JSON, or Python documents only")
        selected.append(doc)
    try:
        if str(ENGINE) not in sys.path:
            sys.path.insert(0, str(ENGINE))
        from generation.sourcepack import build_source_pack
    except ImportError as exc:
        raise ValueError("source-pack engine is unavailable") from exc
    manifest = {"sources": [{"id": "doc-" + d["id"], "title": d["name"],
                              "url": "local://project/" + project_id + "/" + d["id"],
                              "path": d["relative_path"], "evidence_level": "user_document"}
                             for d in selected]}
    project_root = pathlib.Path(project["path"]).resolve()
    pack = build_source_pack(manifest, project_root, max_total_chars=100000, chunk_chars=4000)
    run_id = "sourcepack-" + uuid.uuid4().hex[:12]
    output = project_root / "runs" / run_id
    output.mkdir(parents=True, exist_ok=False)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    (output / "source-pack.json").write_text(json.dumps(pack, indent=2, ensure_ascii=False) + "\n")
    question_path = output / "question.txt"
    question_path.write_text(question + "\n", encoding="utf-8")
    search_config_path = output / "search-config.json"
    search_config_path.write_text(json.dumps(search_config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    grounded_script = ENGINE / "scripts" / "grounded_research.py"
    planned_output = output / "execution"
    plan = {"schema_version": "grounded-research-plan.v1", "question": question,
            "source_pack_digest": pack["digest"], "selected_document_ids": [d["id"] for d in selected],
            "execution": {"enabled": False, "reason": "Plan-only app operation; paid execution requires reviewed CLI workflow.",
                          "command": [engine_python(), str(grounded_script), "--pack", str(output / "source-pack.json"),
                                      "--question", str(question_path), "--output", str(planned_output)],
                          "script_available": grounded_script.is_file()}}
    plan["execution"]["command"].extend(["--search-config", str(search_config_path)])
    plan["search_config"] = search_config
    (output / "grounded-plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n")
    return {"run_id": run_id, "project_id": project_id, "source_count": len(selected),
            "source_pack": os.path.relpath(output / "source-pack.json", ROOT.resolve()),
            "plan": os.path.relpath(output / "grounded-plan.json", ROOT.resolve()), "execution_enabled": False,
            "digest": pack["digest"], "search_config": search_config,
            "search_config_path": os.path.relpath(search_config_path, ROOT.resolve())}


def job_write(job):
    dest = JOBS_DIR / f"{job['id']}.json"
    temp = dest.with_suffix("." + uuid.uuid4().hex + ".tmp"); temp.write_text(json.dumps(job, indent=2)); temp.replace(dest)


def launch_job(job_id, action, options):
    out_dir = RESULTS / "webapp-jobs" / job_id
    if options.get("project_id"):
        project = get_workspace().get_project(str(options["project_id"]))
        job = JOBS[job_id]; job["project_id"] = project["id"]
    if action == "audit_provenance":
        audit_script = ROOT / "scripts/repair_provenance.py"
        if audit_script.exists(): cmd = [sys.executable, str(audit_script)]
        else: cmd = [engine_python(), str(ENGINE / "scripts/audit_release_provenance.py"), "--output", str(out_dir / "provenance-audit.json")]
    elif action == "sample_offline":
        n_gen = int(options.get("n_gen", 1000))
        if not 100 <= n_gen <= 50000: raise ValueError("Samples must be between 100 and 50000")
        seed = max(0, min(int(options.get("seed", 42)), 2**31 - 1))
        device = options.get("device", "cpu") if options.get("device", "cpu") in {"cpu", "mps", "cuda", "auto"} else "cpu"
        source = options.get("candidate_source", "hybrid") if options.get("candidate_source", "hybrid") in {"autoregressive", "uniform", "hybrid"} else "hybrid"
        top_k = max(1, min(int(options.get("top_k", 20)), 100))
        betas = options.get("betas", [0.0, 0.4, 0.7])
        if not isinstance(betas, list) or not betas or len(betas) > 8: betas = [0.0, 0.4, 0.7]
        if any(not math.isfinite(float(b)) or not 0 <= float(b) <= 1 for b in betas): raise ValueError("Beta values must be finite and between 0 and 1")
        betas = [str(float(b)) for b in betas]
        cmd = [engine_python(), str(ENGINE / "local_runner/run.py" if (ENGINE / "local_runner/run.py").exists() else ROOT / "run.py"), "--n-gen", str(n_gen), "--seed", str(seed), "--top-k", str(top_k), "--betas", *betas, "--decode", "0", "--device", device, "--candidate-source", source, "--output", str(out_dir)]
    elif action == "rerank_offline":
        beta = float(options.get("beta", 0.6)); beta = max(0.0, min(beta, 1.0))
        if not math.isfinite(beta): raise ValueError("Beta must be finite")
        limit = max(1, min(int(options.get("limit", 20)), 100))
        source_run = str(options.get("run", "upstream:hybrid-50000-seed45"))
        source_root = run_path(source_run)
        pool = source_root / "k3/scored_candidates.json"
        if not pool.exists(): pool = source_root / "scored_candidates.json"
        if not pool.exists(): raise ValueError("This run has no scored candidate pool; choose a completed sampling run")
        cmd = [engine_python(), str(ENGINE / "alien.py"), "rerank", "--pool", str(pool), "--output", str(out_dir / "rerank.json"), "--beta", str(beta), "--top-k", str(limit)]
    elif action == "prompt_export":
        cid = str(options.get("candidate_id", ""))
        rows = candidate_rows(str(options.get("run", "proposals-v2")))
        row = next((x for x in rows if x.get("id") == cid), None)
        if not row: raise ValueError("unknown candidate_id")
        atoms = "\n".join(f"{i+1}. {a.get('description','')}" for i, a in enumerate(row.get("atoms", [])))
        prompt = f"Design one narrowly specified, falsifiable research proposal from these three seed concepts. Treat all rankings as untrusted and verify supplied evidence.\n\n{atoms}\n"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "prompt.txt").write_text(prompt)
        with LOCK:
            job = JOBS[job_id]; job.update({"status": "completed", "output": str(out_dir.relative_to(RESULTS)), "result": {"candidate_id": cid, "prompt": public_path(out_dir / "prompt.txt")}, "finished_at": time.time()}); job_write(job)
        return
    elif action == "grounded_execute":
        if not options.get("project_id") or not isinstance(options.get("sourcepack_run"), str):
            raise ValueError("grounded execution requires a project and sourcepack_run")
        run_id = options["sourcepack_run"]
        if not run_id.startswith("sourcepack-") or len(run_id) != len("sourcepack-") + 12 or not all(c in "0123456789abcdef" for c in run_id[len("sourcepack-"):]):
            raise ValueError("invalid sourcepack_run")
        project = get_workspace().get_project(str(options["project_id"]))
        source_dir = (pathlib.Path(project["path"]) / "runs" / run_id).resolve()
        project_root = pathlib.Path(project["path"]).resolve()
        if project_root not in source_dir.parents or not source_dir.is_dir():
            raise ValueError("source pack is not in the selected project")
        pack_path = (source_dir / "source-pack.json").resolve(); question_path = (source_dir / "question.txt").resolve()
        plan_path = (source_dir / "grounded-plan.json").resolve()
        if any(source_dir not in p.parents for p in (pack_path, question_path, plan_path)):
            raise ValueError("source pack files escape the selected run")
        search_config_path = (source_dir / "search-config.json").resolve()
        if any(source_dir not in p.parents for p in (pack_path, question_path, plan_path, search_config_path)):
            raise ValueError("source pack files escape the selected run")
        if not pack_path.is_file() or not question_path.is_file() or not plan_path.is_file() or not search_config_path.is_file():
            raise ValueError("source pack plan is incomplete")
        if (source_dir / "execution" / "result.json").exists():
            raise ValueError("this source pack already has a completed result; create a new pack")
        if any(j.get("id") != job_id and j.get("action") == "grounded_execute" and j.get("options", {}).get("project_id") == str(options["project_id"]) and j.get("options", {}).get("sourcepack_run") == run_id and j.get("status") in {"queued", "running", "cancel_requested", "completed"} for j in JOBS.values()):
            raise ValueError("this source pack already has a grounded job")
        if str(ENGINE) not in sys.path: sys.path.insert(0, str(ENGINE))
        from generation.sourcepack import validate_source_pack
        try: pack = json.loads(pack_path.read_text()); validate_source_pack(pack)
        except Exception as exc: raise ValueError("source pack failed integrity validation") from exc
        plan = read_json(plan_path, {}) or {}
        if plan.get("source_pack_digest") != pack.get("digest"):
            raise ValueError("source pack digest does not match plan")
        execution = source_dir / "execution"
        if execution.exists() and execution.is_symlink():
            raise ValueError("execution path cannot be a symlink")
        if execution.resolve().parent != source_dir:
            raise ValueError("execution path escapes the selected run")
        config = read_json(search_config_path, None)
        if not isinstance(config, dict) or not isinstance(config.get("queries"), list) or len(config["queries"]) > 3 or not isinstance(config.get("allow_arxiv"), bool):
            raise ValueError("search config is invalid")
        if any(not isinstance(q, str) or not q.strip() or len(q.strip()) > 200 for q in config["queries"]):
            raise ValueError("search config contains an invalid query")
        if config["allow_arxiv"] and not config["queries"]:
            raise ValueError("search config cannot enable arXiv without queries")
        cmd = [engine_python(), str(ENGINE / "scripts/grounded_research.py"), "--pack", str(pack_path), "--question", str(question_path), "--output", str(execution), "--execute", "--search-config", str(search_config_path)]
    else:
        cmd = ["python3", "-c", "import json; from pathlib import Path; print(json.dumps({\"inventory\": True}))"]
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / "job.log"
    with log.open("w") as fp:
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=fp, stderr=subprocess.STDOUT, start_new_session=True,
                             env={**os.environ, "PYTHONUNBUFFERED": "1", "ALIEN_WORKSPACE": str(ROOT), "ALIEN_REPO": str(ENGINE)})
    with LOCK:
        job = JOBS[job_id]
        if job.get("project_id"):
            get_workspace().register_run(job["project_id"], job_id, {"action": action, "options": options, "output": public_path(out_dir)})
        PROCESSES[job_id] = p
        job.update({"status": "running", "pid": p.pid, "command": cmd, "log": str(log.relative_to(RESULTS)), "output": str(out_dir.relative_to(RESULTS)), "started_at": time.time()})
        job_write(job)

    def wait_for_process():
        code = p.wait()
        with LOCK:
            job["status"] = "completed" if code == 0 else ("cancelled" if code < 0 else "failed")
            job["returncode"] = code
            job["finished_at"] = time.time()
            PROCESSES.pop(job_id, None)
            job_write(job)
    threading.Thread(target=wait_for_process, daemon=True).start()


class Handler(http.server.BaseHTTPRequestHandler):
    server_version = "AlienScience/1.0"

    def log_message(self, fmt, *args):
        return

    def send_json(self, payload, status=200):
        data = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status); self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store"); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)

    def do_OPTIONS(self):
        self.send_response(204); self.send_header("Access-Control-Allow-Origin", "http://127.0.0.1:%d" % PORT); self.send_header("Access-Control-Allow-Methods", "GET,POST,DELETE,OPTIONS"); self.send_header("Access-Control-Allow-Headers", "Content-Type"); self.end_headers()

    def mutation_ok(self):
        origin = self.headers.get("Origin")
        host = self.headers.get("Host", "").split(":", 1)[0]
        return host in {"127.0.0.1", "localhost", "[::1]"} and (not origin or origin in {f"http://127.0.0.1:{PORT}", f"http://localhost:{PORT}"})

    def body(self):
        n = int(self.headers.get("Content-Length", "0"))
        if n < 0 or n > 60_000_000: raise ValueError("Request exceeds 60MB limit")
        try: return json.loads(self.rfile.read(n) or b"{}")
        except ValueError: return None

    def do_GET(self):
        try: return self.get_request()
        except (ValueError, TypeError, KeyError) as exc: return self.send_json({"error": str(exc)}, 400)
        except FileNotFoundError as exc: return self.send_json({"error": str(exc)}, 404)

    def get_request(self):
        parsed = urlparse(self.path); path = parsed.path.rstrip("/") or "/"; qs = parse_qs(parsed.query)
        project_response = self.project_get(path, qs)
        if project_response: return
        if path.startswith("/api/guide/"):
            name = path.split("/")[-1]
            if name not in {"FOLDERS.md", "ENGINE_SETUP.md", "ARCHITECTURE.md"}: raise ValueError("Unknown guide")
            return self.send_json({"name": name, "content": (WEB / "docs" / name).read_text()})
        if path == "/api/setup": return self.send_json(setup_info())
        if path == "/api/health": return self.send_json({"ok": True, "service": "alien-science", "root": public_path(ROOT)})
        if path == "/api/summary": return self.send_json(inventory())
        if path == "/api/operator-library":
            return self.send_json(operator_library())
        if path == "/api/studies":
            rows = study_rows(); return self.send_json({"items": rows, "total": len(rows), "warning": "Study verdicts are evidence records, not automatic novelty claims."})
        if path == "/api/generated-research":
            rows = grounded_rows(); return self.send_json({"items": rows, "total": len(rows), "warning": "Machine review only; these records are not empirical results or novelty findings."})
        if path.startswith("/api/studies/"):
            pieces = path[len("/api/studies/"):].split("/", 1)
            row = study_by_id(unquote(pieces[0]))
            if not row: return self.send_json({"error": "study not found"}, 404)
            if len(pieces) == 1: return self.send_json({"study": row})
            artifact = unquote(pieces[1])
            if pathlib.PurePosixPath(artifact).is_absolute() or any(part in {"", ".", ".."} for part in pathlib.PurePosixPath(artifact).parts):
                return self.send_json({"error": "invalid artifact"}, 400)
            root = (ENGINE / "reconstruction-studies" if row["kind"] == "reconstruction" else ENGINE / "candidate-studies") / row["id"]
            target = (root / artifact).resolve()
            if root.resolve() not in target.parents or target.suffix.lower() not in {".json", ".md", ".py"} or not target.is_file():
                return self.send_json({"error": "artifact not found"}, 404)
            return self.send_json({"name": artifact, "content": target.read_text(errors="replace")})
        if path == "/api/runs": return self.send_json({"runs": runs(), "provenance": provenance()})
        if path == "/api/candidates":
            run_id = qs.get("run", ["proposals-v2"])[0]
            try: run_path(run_id)
            except ValueError: return self.send_json({"error": "invalid run"}, 400)
            rows = candidate_rows(run_id); q = qs.get("q", [""])[0].lower()
            if q: rows = [r for r in rows if q in json.dumps(r, ensure_ascii=False).lower()]
            try: offset = max(0, int(qs.get("offset", [0])[0])); limit = min(100, max(1, int(qs.get("limit", [50])[0])))
            except ValueError: return self.send_json({"error": "invalid pagination"}, 400)
            return self.send_json({"items": rows[offset:offset+limit], "total": len(rows), "offset": offset, "provenance": provenance()})
        if path.startswith("/api/candidates/"):
            cid = path.split("/")[-1]; row = next((r for r in candidate_rows(qs.get("run", ["proposals-v2"])[0]) if r.get("id") == cid), None)
            if not row: return self.send_json({"error": "candidate not found"}, 404)
            return self.send_json({"candidate": row, "evidence": self.evidence(cid, qs.get("run", ["proposals-v2"])[0]), "provenance": provenance()})
        if path.startswith("/api/evidence/"):
            data = self.evidence(path.split("/")[-1], qs.get("run", ["proposals-v2"])[0]); return self.send_json(data, 200 if data.get("found") else 404)
        if path == "/api/jobs":
            with LOCK: return self.send_json({"jobs": list(JOBS.values())})
        if path == "/api/adjudications":
            return self.send_json({"items": adjudications(), "storage": public_path(ADJUDICATIONS)})
        if path == "/api/artifacts":
            run_id = qs.get("run", ["proposals-v2"])[0]
            try: run_path(run_id)
            except ValueError: return self.send_json({"error": "invalid run"}, 400)
            root = run_path(run_id)
            files = [{"name": str(p.relative_to(root)), "size": p.stat().st_size, "kind": p.suffix[1:] or "text"} for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".json", ".md", ".txt", ".html"}][:500] if root.exists() else []
            return self.send_json({"run": run_id, "files": files})
        if path.startswith("/api/artifacts/"):
            rel = path[len("/api/artifacts/"):]
            pieces = rel.split("/", 1)
            if rel.startswith("webapp-jobs/"):
                parts = rel.split("/", 2)
                pieces = ["/".join(parts[:2]), parts[2]] if len(parts) == 3 else []
            if len(pieces) != 2: return self.send_json({"error": "invalid artifact"}, 400)
            try: root = run_path(unquote(pieces[0]))
            except ValueError: return self.send_json({"error": "invalid artifact"}, 400)
            target = (root / unquote(pieces[1])).resolve()
            if pieces == ["provenance-repair-2026-11-16", "REPAIR-REPORT.md"] and not target.exists() and (ENGINE / "PROVENANCE.md").is_file():
                return self.send_json({"name": "PROVENANCE.md", "content": (ENGINE / "PROVENANCE.md").read_text()})
            if root not in target.parents or target.suffix.lower() not in {".json", ".md", ".txt", ".html"} or not target.is_file(): return self.send_json({"error": "artifact not found"}, 404)
            return self.send_json({"name": str(target.relative_to(root)), "content": target.read_text(errors="replace")})
        if path.startswith("/api/jobs/"):
            jid = path.split("/")[-1]; job = JOBS.get(jid) or read_json(JOBS_DIR / f"{jid}.json")
            if not job: return self.send_json({"error": "job not found"}, 404)
            log = RESULTS / job.get("output", "") / "job.log" if job.get("output") else None
            if log and log.exists(): job = {**job, "log_tail": log.read_text(errors="replace")[-12000:]}
            return self.send_json(job)
        return self.static(path)

    def evidence(self, cid, run="proposals-v2"):
        data = read_json(RESULTS / "provenance-repair-2026-11-16" / "selected-evidence.repaired.json", {}) or {}
        c = next((x for x in data.get("candidates", []) if x.get("id") == cid), None)
        atoms = {str(k): v for k, v in data.get("atoms", {}).items()}
        if not c:
            c = next((r for r in candidate_rows(run) if r["id"] == cid), None)
            if not c: return {"found": False, "error": "evidence not found"}
            original = read_json(ENGINE / "papers/clusters.json", {}) or {}
            for aid in c.get("cluster_ids", []):
                if str(aid) not in atoms:
                    units = [{"assignment_index": i, "paper_id": x.get("paper_id"), "text": x.get("text")} for i, x in enumerate(original.get("atom_assignments", [])) if x.get("cluster") == aid]
                    atoms[str(aid)] = {"canonical_description": original.get("clusters", {}).get(str(aid), {}).get("name"), "source_units": units, "status": "original assignments; not a validation of summary claims"}
        return {"found": True, "candidate": c, "atoms": {k: atoms.get(str(k)) for k in c.get("cluster_ids", [])}, "source": "original assignments (clusters.json); repaired sidecars where available", "status": data.get("status")}

    def do_POST(self):
        try: return self.post_request()
        except (ValueError, TypeError, KeyError) as exc: return self.send_json({"error": str(exc)}, 400)
        except FileNotFoundError as exc: return self.send_json({"error": str(exc)}, 404)

    def post_request(self):
        if not self.mutation_ok(): return self.send_json({"error": "origin rejected"}, 403)
        path = urlparse(self.path).path.rstrip("/")
        if self.project_post(path): return
        if path == "/api/operator-check":
            try:
                record = run_operator_check(self.body())
            except (ValueError, TypeError, KeyError, ImportError) as exc:
                return self.send_json({"error": str(exc)}, 400)
            return self.send_json(record, 200)
        if path == "/api/adjudications":
            body = self.body()
            if not isinstance(body, dict) or not isinstance(body.get("candidate_id"), str) or not isinstance(body.get("decision"), str):
                return self.send_json({"error": "candidate_id and decision are required"}, 400)
            decision = body["decision"].strip().lower()
            if decision not in {"investigate", "revise", "reject", "hold"}:
                return self.send_json({"error": "decision must be investigate, revise, reject, or hold"}, 400)
            note = str(body.get("notes", "")).strip()
            if len(note) > 5000: return self.send_json({"error": "notes too long"}, 400)
            if not any(r.get("id") == body["candidate_id"] for r in candidate_rows(str(body.get("run", "proposals-v2")))):
                return self.send_json({"error": "unknown candidate"}, 404)
            record = {"id": uuid.uuid4().hex[:12], "candidate_id": body["candidate_id"], "run": body.get("run", "proposals-v2"), "decision": decision, "notes": note, "created_at": time.time()}
            if body.get("project_id"):
                project = get_workspace().get_project(str(body["project_id"]))
                record["project_id"] = project["id"]
                get_workspace().save_note(project["id"], f"decision-{record['id']}", "# Candidate decision\n\n```json\n" + json.dumps(record, indent=2, ensure_ascii=False) + "\n```\n", "reviews")
            with LOCK:
                items = adjudications(); items.append(record); ADJUDICATIONS.write_text(json.dumps(items, indent=2, ensure_ascii=False))
            return self.send_json(record, 201)
        if path != "/api/jobs": return self.send_json({"error": "not found"}, 404)
        body = self.body()
        if not isinstance(body, dict) or body.get("action") not in ALLOWED_ACTIONS: return self.send_json({"error": "unknown or disabled action"}, 400)
        jid = uuid.uuid4().hex[:12]; job = {"id": jid, "action": body["action"], "status": "queued", "created_at": time.time(), "options": body.get("options") or {}}
        with LOCK:
            if body["action"] in {"sample_offline", "rerank_offline", "audit_provenance", "grounded_execute"} and any(p.poll() is None for p in PROCESSES.values()):
                return self.send_json({"error": "Another engine job is running. Wait for it or cancel it first."}, 409)
            JOBS[jid] = job; job_write(job)
        if body["action"] == "inventory":
            job.update({"status": "completed", "result": inventory(), "finished_at": time.time()}); job_write(job)
        elif body["action"] == "decode_paid":
            job.update({"status": "blocked", "reason": "Paid decoding is disabled in the local app until a fresh provenance-reviewed manifest and explicit user approval exist.", "finished_at": time.time()}); job_write(job)
            return self.send_json(job, 409)
        else:
            try: launch_job(jid, body["action"], body.get("options") or {})
            except (ValueError, TypeError, OSError) as exc:
                job.update(status="failed", reason=str(exc), finished_at=time.time()); job_write(job)
                return self.send_json(job, 400)
        return self.send_json(job, 202)

    def do_DELETE(self):
        if not self.mutation_ok(): return self.send_json({"error": "origin rejected"}, 403)
        path = urlparse(self.path).path.rstrip("/"); jid = path.split("/")[-1]
        with LOCK: p = PROCESSES.get(jid); job = JOBS.get(jid) or read_json(JOBS_DIR / f"{jid}.json")
        if not job: return self.send_json({"error": "job not found"}, 404)
        if p and p.poll() is None:
            try: os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError: pass
            job["status"] = "cancel_requested"; job_write(job)
        return self.send_json(job)

    def project_get(self, path, qs):
        if not path.startswith('/api/projects'): return False
        parts = path.strip('/').split('/')
        workspace = get_workspace()
        if len(parts) == 2:
            self.send_json({'items': workspace.list_projects()}); return True
        project_id = parts[2]
        if len(parts) == 3:
            self.send_json(workspace.get_project(project_id)); return True
        section = parts[3]
        if section == 'folders':
            self.send_json({'items': workspace.folder_map(project_id), 'project': workspace.get_project(project_id)}); return True
        if section == 'documents':
            if len(parts) == 4:
                self.send_json({'items': workspace.list_documents(project_id)}); return True
            doc = workspace.read_document(project_id, parts[4])
            doc['download_url'] = f'/api/projects/{project_id}/documents/{parts[4]}/download'
            if len(parts) == 6 and parts[5] == 'download':
                content = workspace.document_path(project_id, parts[4]).read_bytes()
                self.send_bytes(content, doc['name'], 'application/pdf' if doc['binary'] else 'text/plain; charset=utf-8')
            else: self.send_json(doc)
            return True
        if section == 'exports' and len(parts) == 5:
            name = unquote(parts[4])
            if pathlib.Path(name).name != name or not name.endswith('.zip'): raise ValueError('Invalid export name')
            project_root = pathlib.Path(workspace.get_project(project_id)['path']).resolve()
            export_path = (project_root / 'exports' / name).resolve()
            if project_root / 'exports' != export_path.parent or not export_path.is_file(): raise FileNotFoundError('Export not found')
            self.send_bytes(export_path.read_bytes(), name, 'application/zip'); return True
        self.send_json({'error': 'Unknown project endpoint'}, 404); return True

    def project_post(self, path):
        if not path.startswith('/api/projects'): return False
        body = self.body()
        if not isinstance(body, dict): raise ValueError('JSON object required')
        parts = path.strip('/').split('/')
        workspace = get_workspace()
        with LOCK:
            if len(parts) == 2:
                self.send_json(workspace.create_project(body.get('name'), body.get('description', '')), 201); return True
            if len(parts) == 4 and parts[3] == 'source-pack':
                self.send_json(build_project_source_pack(parts[2], body), 201); return True
            if len(parts) != 4: raise ValueError('Unknown project endpoint')
            project_id, section = parts[2], parts[3]
            if section == 'documents':
                self.send_json({'items': workspace.import_documents(project_id, body.get('files'))}, 201); return True
            if section == 'notes':
                self.send_json(workspace.save_note(project_id, body.get('title', ''), body.get('content', ''), body.get('category', 'notes')), 201); return True
            if section == 'export':
                result = workspace.export_project(project_id)
                result['download_url'] = f'/api/projects/{project_id}/exports/{result["name"]}'
                self.send_json(result, 201); return True
        raise ValueError('Unknown project endpoint')

    def send_bytes(self, data, name, content_type):
        from urllib.parse import quote
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Disposition', "attachment; filename*=UTF-8''" + quote(name))
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers(); self.wfile.write(data)

    def static(self, path):
        rel = "index.html" if path == "/" else (path[len("/static/"):] if path.startswith("/static/") else path.lstrip("/"))
        target = (WEB / "static" / rel).resolve(); root = (WEB / "static").resolve()
        if root not in target.parents and target != root: return self.send_json({"error": "not found"}, 404)
        if not target.is_file(): return self.send_json({"error": "not found"}, 404)
        data = target.read_bytes(); self.send_response(200); self.send_header("Content-Type", mimetypes.guess_type(str(target))[0] or "application/octet-stream"); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alien Science local workbench")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--open", action="store_true", help="Open the app in your default browser")
    args = parser.parse_args()
    PORT = args.port
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        raise SystemExit("Refusing non-loopback host; this app is local-only")
    try: httpd = http.server.ThreadingHTTPServer((args.host, args.port), Handler)
    except OSError as exc:
        if args.open:
            import urllib.request
            import webbrowser
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{args.port}/api/health', timeout=2) as response:
                    if json.load(response).get('service') == 'alien-science':
                        webbrowser.open(f'http://127.0.0.1:{args.port}'); raise SystemExit(0)
            except (OSError, ValueError): pass
        raise SystemExit(f'Cannot start on port {args.port}: {exc}. Try --port {args.port + 1}.')
    print(f"Alien Science: http://{args.host}:{args.port}", flush=True)
    if args.open:
        import webbrowser
        threading.Timer(0.3, lambda: webbrowser.open(f"http://127.0.0.1:{args.port}")).start()
    try: httpd.serve_forever()
    except KeyboardInterrupt: pass
    finally:
        for process in list(PROCESSES.values()):
            try: os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError: pass
        httpd.server_close()
