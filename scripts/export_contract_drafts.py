"""Export a bounded contract-draft run without publishing source excerpts.

The extractor keeps exact quotes locally so they can be validated.  This
exporter is deliberately lossy: evidence quotes become hashes, and the public
result contains no prompts, model responses, or call receipts.  The sibling
library summary is bound to the redacted result by SHA-256, as required by the
workbench reader.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from generation.contract_drafts import validate_drafts, validate_review
from generation.sourcepack import validate_source_pack

MAX_RESULT = 1_000_000
MAX_PACK = 2_000_000
MAX_SOURCES = 5
SENSITIVE_KEYS = {"prompt", "prompts", "content", "raw_output", "response", "request"}


def _read_json(path: Path, limit: int) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing or symlinked JSON: {path}")
    if path.stat().st_size > limit:
        raise ValueError(f"oversized JSON: {path.name}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path.name}")
    return value


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _redact(value: Any, key: str | None = None) -> Any:
    if key in SENSITIVE_KEYS:
        raise ValueError(f"raw model/request field is not exportable: {key}")
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if k == "quote":
                if not isinstance(v, str):
                    raise ValueError("evidence quote must be text")
                out["quote_sha256"] = _hash_text(v)
                out["quote_chars"] = len(v)
            else:
                out[k] = _redact(v, k)
        return out
    if isinstance(value, list):
        return [_redact(v, key) for v in value]
    return value


def _source_map(pack: dict[str, Any]) -> dict[str, str]:
    sources = pack.get("sources")
    if not isinstance(sources, list) or not 1 <= len(sources) <= MAX_SOURCES:
        raise ValueError("pack must contain 1-5 sources")
    result = {}
    for source in sources:
        if not isinstance(source, dict) or not isinstance(source.get("id"), str):
            raise ValueError("invalid pack source")
        sid = source["id"]
        url = source.get("url", "")
        if not isinstance(url, str) or len(url) > 2_000:
            raise ValueError("invalid source URL")
        result[sid] = url
    if len(result) != len(sources):
        raise ValueError("duplicate pack source IDs")
    return result


def _library(result: dict[str, Any], urls: dict[str, str]) -> dict[str, Any]:
    drafts = result.get("drafts", [])
    if not isinstance(drafts, list) or len(drafts) > MAX_SOURCES:
        raise ValueError("invalid draft list")
    reviews = result.get("review")
    assessments = reviews.get("assessments", []) if isinstance(reviews, dict) else []
    by_source = {a.get("source_id"): a for a in assessments if isinstance(a, dict)}
    source_ids = result.get("source_ids", [])
    draft_map = {d.get("source_id"): d for d in drafts if isinstance(d, dict)}
    rows = []
    # Include every selected source, including null/failed records. This keeps
    # a failed run visible in the workbench instead of silently shrinking it.
    for sid in source_ids:
        draft = draft_map.get(sid, {"source_id": sid, "contract": None, "unknowns": ["No draft was produced."]})
        card = draft.get("contract")
        if card is None:
            assessment = by_source.get(sid, {})
            rows.append({"source_id": draft["source_id"], "source_url": urls.get(draft["source_id"], ""),
                         "contract_id": None, "inputs": [], "outputs": [],
                         "unknowns": draft.get("unknowns", []), "review": {"decision": "not_recorded", "reason": "", "issues": []}})
            if assessment:
                rows[-1]["review"] = {"decision": assessment.get("decision", "not_recorded"),
                                       "reason": assessment.get("reason", ""), "issues": [{k:i.get(k) for k in ("port_id","field","reason")} for i in assessment.get("issues",[])]}
            continue
        if not isinstance(card, dict):
            raise ValueError("invalid contract")
        assessment = by_source.get(draft["source_id"], {})
        # Only project issue text. Evidence quotes are intentionally omitted;
        # the public result contains their hashes instead.
        safe_issues = []
        for issue in assessment.get("issues", []):
            if isinstance(issue, dict):
                safe_issues.append({"port_id": issue.get("port_id"), "field": issue.get("field", ""),
                                    "reason": issue.get("reason", "")})
        rows.append({"source_id": draft["source_id"], "source_url": urls.get(draft["source_id"], ""),
                     "contract_id": card.get("id"), "inputs": card.get("inputs", []), "outputs": card.get("outputs", []),
                     "unknowns": draft.get("unknowns", []),
                     "review": {"decision": assessment.get("decision", "not_recorded"),
                                "reason": assessment.get("reason", ""), "issues": assessment.get("issues", [])}})
        rows[-1]["review"]["issues"] = safe_issues
    return {"schema": "contract-draft-library.v1", "status": result.get("status", ""),
            "result_sha256": "", "records": rows, "curated": False,
            "novelty_established": False}


def export_run(run: Path, output: Path, pack_path: Path) -> dict[str, Any]:
    if output.exists() and output.is_symlink():
        raise ValueError("symlinked output is not allowed")
    result_path = run / "result.json"
    result = _read_json(result_path, MAX_RESULT)
    pack = _read_json(pack_path, MAX_PACK)
    validate_source_pack(pack)
    urls = _source_map(pack)
    source_ids = result.get("source_ids")
    if (not isinstance(source_ids, list) or not 1 <= len(source_ids) <= MAX_SOURCES
            or any(not isinstance(s,str) for s in source_ids)
            or len(set(source_ids)) != len(source_ids) or set(source_ids) != set(urls)):
        raise ValueError("result source_ids must match pack")
    status = result.get("status")
    if status not in {"failed", "reviewed_drafts_not_curated"}:
        raise ValueError("only failed or reviewed extractor runs can be exported")
    drafts = result.get("drafts", [])
    if not isinstance(drafts, list):
        raise ValueError("invalid drafts")
    if len(drafts) > MAX_SOURCES or any(not isinstance(d,dict) for d in drafts):
        raise ValueError("invalid draft records")
    draft_ids = [d.get("source_id") for d in drafts]
    if (any(not isinstance(s,str) or s not in source_ids for s in draft_ids)
            or len(set(draft_ids)) != len(draft_ids)):
        raise ValueError("invalid or duplicate draft source IDs")
    if drafts:
        envelope = {"records": [{k:d.get(k) for k in ("source_id","contract","unknowns")} for d in drafts]}
        validate_drafts(envelope, pack, draft_ids)
    if result.get("review") is not None:
        if not drafts or not isinstance(result["review"],dict):
            raise ValueError("review requires valid drafts")
        validate_review({"assessments": result["review"].get("assessments")},drafts,pack)
    if status == "reviewed_drafts_not_curated" and (set(draft_ids)!=set(source_ids) or result.get("review") is None):
        raise ValueError("reviewed result must contain every draft and review")
    for flag in ("train_eligible","novelty_established","executable_compatibility_proven","curated_registry_updated"):
        if result.get(flag,False) is not False:
            raise ValueError("draft export cannot assert promotion flags")
    plan_path = run / "plan.json"
    pack_sha = hashlib.sha256(pack_path.read_bytes()).hexdigest()
    if plan_path.is_file():
        plan = _read_json(plan_path, MAX_RESULT)
        if plan.get("pack_sha256") != pack_sha:
            raise ValueError("plan source-pack hash mismatch")
    redacted = _redact(result)
    redacted["export_scope"] = "Evidence quotes replaced by hashes; prompts, raw model responses and call receipts are excluded."
    redacted["source_urls"] = {sid: urls[sid] for sid in source_ids}
    redacted["original_result_sha256"] = hashlib.sha256(result_path.read_bytes()).hexdigest()
    redacted["source_pack_sha256"] = pack_sha
    result_bytes = (json.dumps(redacted, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
    summary = _library(result, urls)
    summary["result_sha256"] = hashlib.sha256(result_bytes).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    for name, data in (("result.json", result_bytes), ("library-summary.json", (json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"))):
        target = output / name
        if target.exists() or target.is_symlink():
            if target.is_symlink() or target.read_bytes() != data:
                raise FileExistsError(f"refusing to overwrite existing export: {target.name}")
        else:
            target.write_bytes(data)
    return {"output": str(output), "status": result.get("status"), "result_sha256": summary["result_sha256"], "records": len(summary["records"])}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_run(args.run, args.output, args.pack), sort_keys=True))
