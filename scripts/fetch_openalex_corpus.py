#!/usr/bin/env python3
"""Resumable, bounded OpenAlex metadata/abstract collector.

No API key is read. The command is deliberately a plan-and-run tool: a parent
process must explicitly invoke it to make network requests. Every response is
recorded in receipts.jsonl and the cursor is advanced only after records and
state are durably written. A 429 or request cap stops cleanly.
"""
from __future__ import annotations
import argparse, gzip, hashlib, json, os, time, urllib.parse, urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = ROOT / "docs/corpus-expansion/OPENALEX_BULK_PLAN.json"
API = "https://api.openalex.org/works"
FIELDS = "id,doi,title,publication_date,abstract_inverted_index,cited_by_count,fwci,citation_normalized_percentile,topics,type,locations,ids"

def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def reconstruct(index):
    if not isinstance(index, dict): return ""
    words = {}
    for word, positions in index.items():
        for pos in positions if isinstance(positions, list) else []: words[int(pos)] = str(word)
    return " ".join(words[i] for i in sorted(words))

def load_plan(path):
    data = json.loads(path.read_text())
    if data.get("schema") != "mechanism-lab.openalex-plan.v1": raise ValueError("unsupported plan schema")
    if data.get("max_requests", 0) > 700 or data.get("per_page") != 200: raise ValueError("unsafe plan bounds")
    if data.get("pagination_mode", "cursor") not in {"cursor", "page"}: raise ValueError("unsupported pagination mode")
    if data.get("pagination_mode", "cursor") == "page" and not 1 <= int(data.get("max_pages_per_slot", 0)) <= 50: raise ValueError("unsafe page bound")
    for category, topics in data["topics"].items():
        if not topics or data["targets"].get(category, 0) <= 0: raise ValueError("empty category plan")
        for topic in topics:
            if not topic["id"].startswith("T"): raise ValueError("invalid topic ID")
    return data

def query_plan(plan):
    slots = []
    windows = plan["year_windows"]
    for category in ("machine_learning", "systems", "mathematics"):
        topics = plan["topics"][category]
        target = plan["targets"][category]
        base, rem = divmod(target, len(topics) * len(windows))
        for ti, topic in enumerate(topics):
            for wi, (start, end) in enumerate(windows):
                slots.append({"id": f"{category}:{topic['id']}:{wi}", "category": category,
                              "topic": topic, "from": start, "to": end,
                              "target": base + int(ti * len(windows) + wi < rem)})
    return slots

def normalize(row, slot, retrieved_at):
    ids = row.get("ids") or {}
    if not row.get("id") or not row.get("title") or not row.get("abstract_inverted_index"):
        return None
    return {"source_id": "openalex:" + str(row.get("id", "")).rsplit("/", 1)[-1],
            "canonical_id": str(row.get("id", "")), "doi": row.get("doi"),
            "arxiv_id": ids.get("arxiv"), "title": row.get("title") or "",
            "abstract": reconstruct(row.get("abstract_inverted_index")),
            "published_date": row.get("publication_date") or "",
            "cited_by_count": row.get("cited_by_count"), "fwci": row.get("fwci"),
            "citation_normalized_percentile": row.get("citation_normalized_percentile"),
            "topics": row.get("topics") or [], "type": row.get("type"),
            "locations": row.get("locations") or [], "query_slot": slot["id"],
            "retrieved_at": retrieved_at, "retrieval_method": "openalex_api",
            "evidence_level": "metadata_abstract", "source_hash": hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()}

def _write_state(path, state):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True))
    os.replace(temporary, path)

def _rebuild_state(state, records_path, slots):
    """Reconcile append-only records after a crash before state replacement."""
    seen, counts = {}, {slot["id"]: 0 for slot in slots}
    if records_path.exists():
        with records_path.open() as stream:
            for line in stream:
                try: row = json.loads(line)
                except json.JSONDecodeError: continue
                key = row.get("doi") or row.get("arxiv_id") or row.get("canonical_id")
                if key and key not in seen:
                    seen[key] = row.get("canonical_id")
                    if row.get("query_slot") in counts: counts[row["query_slot"]] += 1
    state["seen"], state["records"] = seen, len(seen)
    if records_path.parent.joinpath("receipts.jsonl").exists():
        requests = 0
        with records_path.parent.joinpath("receipts.jsonl").open() as stream:
            for line in stream:
                try: requests = max(requests, int(json.loads(line).get("request", 0)))
                except (ValueError, TypeError, json.JSONDecodeError): continue
        state["requests"] = max(int(state.get("requests", 0)), requests)
    for slot_id, count in counts.items():
        state.setdefault("slots", {}).setdefault(slot_id, {"cursor":"*"})["count"] = count
    return state

def run(plan_path: Path, out: Path, *, max_requests=None, interval=.25, timeout=30, save_responses=False):
    plan = load_plan(plan_path); out.mkdir(parents=True, exist_ok=True)
    plan_sha = digest(plan_path); state_path, records_path, receipts_path = out/"state.json", out/"records.jsonl", out/"receipts.jsonl"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"schema":"openalex-run.v1","plan_sha256":plan_sha,"requests":0,"records":0,"seen":{},"slots":{}}
    if state.get("plan_sha256") != plan_sha: raise ValueError("output belongs to another plan")
    cap = int(max_requests or plan["max_requests"])
    if not 1 <= cap <= plan["max_requests"] or interval < .2: raise ValueError("unsafe runtime bounds")
    slots = query_plan(plan); state = _rebuild_state(state, records_path, slots); stop_reason = "targets_reached"
    seen = state.setdefault("seen", {})
    last_request = 0.0
    for slot in slots:
        progress = state["slots"].setdefault(slot["id"], {"cursor":"*","count":0,"done":False})
        if progress.get("done"): continue
        while (not progress.get("done") and state["requests"] < cap and
               state["records"] < plan["max_records"] and progress["count"] < slot["target"]):
            params = {"filter":f"topics.id:{slot['topic']['id']},has_abstract:true,from_publication_date:{slot['from']},to_publication_date:{slot['to']}","select":FIELDS,"per-page":plan["per_page"]}
            if plan.get("pagination_mode", "cursor") == "page":
                params.update(page=int(progress.get("page", 1)), sort=plan.get("sort", "publication_date:desc"))
            else:
                params["cursor"] = progress.get("cursor", "*")
            url = API + "?" + urllib.parse.urlencode(params)
            delay = interval - (time.monotonic() - last_request)
            if delay > 0: time.sleep(delay)
            last_request = time.monotonic()
            started = time.time(); status = 0; body = b""; error = None
            try:
                req = urllib.request.Request(url, headers={"User-Agent":"MechanismLab/OpenAlexCollector/1.0"})
                with urllib.request.urlopen(req, timeout=timeout) as response: status = int(response.status); body = response.read(15_000_001)
                if len(body) > 15_000_000: raise ValueError("response exceeds 15MB cap")
                payload = json.loads(body); rows = payload.get("results", []); meta = payload.get("meta", {})
            except urllib.error.HTTPError as exc:
                status = int(exc.code)
                error = {"type":type(exc).__name__,"message":str(exc)}; payload = {}; rows = []; meta = {}
            except Exception as exc:
                error = {"type":type(exc).__name__,"message":str(exc)}; payload = {}; rows = []; meta = {}
            receipt = {"timestamp":datetime.now(timezone.utc).isoformat(),"slot":slot["id"],"request":state["requests"]+1,"url":url,"cursor":progress["cursor"],"status":status,"bytes":len(body),"response_sha256":hashlib.sha256(body).hexdigest() if body else None,"elapsed_seconds":time.time()-started,"error":error,"returned":len(rows)}
            if save_responses and body:
                response_dir = out / "responses"; response_dir.mkdir(exist_ok=True)
                with gzip.open(response_dir / f"{state['requests'] + 1:06d}.json.gz", "wb") as raw: raw.write(body)
            with receipts_path.open("a") as f: f.write(json.dumps(receipt, sort_keys=True)+"\n")
            state["requests"] += 1
            if error:
                state["stop_reason"] = "http_429" if status == 429 else "request_error"; state_path.write_text(json.dumps(state,indent=2)); return state
            normalized = [x for x in (normalize(r, slot, receipt["timestamp"][:10]) for r in rows) if x is not None]
            with records_path.open("a") as f:
                for row in normalized:
                    if state["records"] >= plan["max_records"] or progress["count"] >= slot["target"]: break
                    key = row["doi"] or row["arxiv_id"] or row["canonical_id"]
                    if key in seen: continue
                    seen[key] = row["canonical_id"]; f.write(json.dumps(row,sort_keys=True)+"\n"); state["records"] += 1; progress["count"] += 1
            next_cursor = meta.get("next_cursor")
            if plan.get("pagination_mode", "cursor") == "page":
                progress["page"] = int(progress.get("page", 1)) + 1
                progress["done"] = len(rows) < plan["per_page"] or progress["count"] >= slot["target"] or progress["page"] > plan["max_pages_per_slot"]
            else:
                progress["cursor"] = next_cursor or ""; progress["done"] = not next_cursor or len(rows) == 0 or progress["count"] >= slot["target"]
            _write_state(state_path, state)
        if state["requests"] >= cap: stop_reason = "request_cap"; break
    state["stop_reason"] = stop_reason; _write_state(state_path, state); return state

def main():
    p=argparse.ArgumentParser(); p.add_argument("--plan",type=Path,default=DEFAULT_PLAN); p.add_argument("--output",type=Path,required=True); p.add_argument("--max-requests",type=int); p.add_argument("--interval",type=float,default=.25); p.add_argument("--timeout",type=float,default=30); p.add_argument("--save-responses",action="store_true"); p.add_argument("--dry-run",action="store_true"); a=p.parse_args()
    plan=load_plan(a.plan)
    if a.dry_run: print(json.dumps({"plan_sha256":digest(a.plan),"slots":query_plan(plan),"max_requests":plan["max_requests"]},indent=2)); return
    result=run(a.plan,a.output,max_requests=a.max_requests,interval=a.interval,timeout=a.timeout,save_responses=a.save_responses)
    print(json.dumps({k:result[k] for k in ("schema","plan_sha256","requests","records","stop_reason")},indent=2))
if __name__ == "__main__": main()
