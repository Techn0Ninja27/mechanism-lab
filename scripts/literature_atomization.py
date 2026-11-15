"""Extract provenance-bound literature atoms with a bounded OpenRouter run.

The model proposes source-grounded conceptual units.  It is not asked to judge
novelty and its output is never training-eligible automatically.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
import pathlib
import re
import sys
import time
import urllib.request
from typing import Any, Iterable

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "local_runner"))
from run import key  # noqa: E402

DEFAULT_MODEL = "anthropic/claude-haiku-5.5"
DEFAULT_MAX_TOKENS = 4096
MAX_CALLS = 8
PAPERS_PER_BATCH = 4
KINDS = {"method", "finding", "limitation"}


def _norm(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def load_papers(path: pathlib.Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("input must be a non-empty JSON list")
    result = []
    seen = set()
    for i, paper in enumerate(raw):
        if not isinstance(paper, dict):
            raise ValueError(f"paper {i} must be an object")
        pid = str(paper.get("source_id", paper.get("id", ""))).strip()
        for field in ("title", "url", "abstract"):
            if not isinstance(paper.get(field), str) or not paper[field].strip():
                raise ValueError(f"paper {i} missing non-empty {field}")
        if not pid or pid in seen:
            raise ValueError(f"missing or duplicate source_id: {pid!r}")
        seen.add(pid)
        result.append({
            "source_id": pid,
            "title": paper["title"].strip(),
            "url": paper["url"].strip(),
            "abstract": paper["abstract"].strip(),
            "authors": paper.get("authors", []),
        })
    return result


def source_hash(paper: dict[str, Any]) -> str:
    return _sha({k: paper[k] for k in ("source_id", "title", "url", "abstract")})


def batches(papers: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    return [papers[i:i + PAPERS_PER_BATCH] for i in range(0, len(papers), PAPERS_PER_BATCH)]


def build_prompt(papers: Iterable[dict[str, Any]]) -> str:
    payload = []
    for paper in papers:
        payload.append({"source_id": paper["source_id"], "title": paper["title"],
                        "url": paper["url"], "abstract": paper["abstract"]})
    return """You are extracting reusable literature atoms for a research corpus.
This is source indexing, not a novelty judgment. Use ONLY the supplied papers.
Extract 2 to 4 concise atoms per paper when the abstract supports them; do not
pad a paper with unsupported claims. Every atom must have a source_id and a
source_quote that is an exact contiguous substring of that paper's abstract
(whitespace may differ). The statement must be a faithful paraphrase of that
quote. Mark assumptions as explicit when the abstract states them and inferred
when you infer a necessary condition; never present an inferred assumption as
quoted evidence. Do not say that an atom is novel, unavailable, or a new
primitive. Return JSON only, with an object containing an `atoms` array.

Each atom object MUST have exactly these fields:
source_id (string), source_quote (string), statement (string),
kind (one of method, finding, limitation), inputs (array of strings),
outputs (array of strings), assumptions (array of objects with `text` and
`basis`, where basis is explicit or inferred).

PAPERS:
""" + json.dumps(payload, ensure_ascii=False, indent=2)


def _parse_json(text: str) -> Any:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        cleaned = "\n".join(lines[1:-1])
    return json.loads(cleaned)


def validate_atoms(value: Any, papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(value, dict) or not isinstance(value.get("atoms"), list):
        raise ValueError("response must be an object with an atoms array")
    by_id = {p["source_id"]: p for p in papers}
    output = []
    for i, atom in enumerate(value["atoms"]):
        if not isinstance(atom, dict):
            raise ValueError(f"atom {i} must be an object")
        required = {"source_id", "source_quote", "statement", "kind", "inputs", "outputs", "assumptions"}
        if set(atom) != required:
            raise ValueError(f"atom {i} fields must be exactly {sorted(required)}")
        sid = atom["source_id"]
        if sid not in by_id:
            raise ValueError(f"atom {i} has unknown source_id {sid!r}")
        quote = atom["source_quote"]
        if not isinstance(quote, str) or not quote.strip():
            raise ValueError(f"atom {i} source_quote must be non-empty")
        if _norm(quote).casefold() not in _norm(by_id[sid]["abstract"]).casefold():
            raise ValueError(f"atom {i} source_quote is not a contiguous abstract span")
        if not isinstance(atom["statement"], str) or not atom["statement"].strip():
            raise ValueError(f"atom {i} statement must be non-empty")
        if atom["kind"] not in KINDS:
            raise ValueError(f"atom {i} has invalid kind")
        for field in ("inputs", "outputs"):
            if not isinstance(atom[field], list) or not all(isinstance(x, str) and x.strip() for x in atom[field]):
                raise ValueError(f"atom {i} {field} must be a list of strings")
        if not isinstance(atom["assumptions"], list):
            raise ValueError(f"atom {i} assumptions must be a list")
        assumptions = []
        for j, assumption in enumerate(atom["assumptions"]):
            if not isinstance(assumption, dict) or set(assumption) != {"text", "basis"}:
                raise ValueError(f"atom {i} assumption {j} malformed")
            if not isinstance(assumption["text"], str) or not assumption["text"].strip():
                raise ValueError(f"atom {i} assumption {j} text missing")
            if assumption["basis"] not in {"explicit", "inferred"}:
                raise ValueError(f"atom {i} assumption {j} basis invalid")
            assumptions.append({"text": assumption["text"].strip(), "basis": assumption["basis"]})
        output.append({
            "source_id": sid, "source_quote": _norm(quote), "statement": atom["statement"].strip(),
            "kind": atom["kind"], "inputs": [x.strip() for x in atom["inputs"]],
            "outputs": [x.strip() for x in atom["outputs"]], "assumptions": assumptions,
            "source_hash": source_hash(by_id[sid]), "human_adjudicated": False,
            "train_eligible": False,
        })
    return output


def plan(input_path: pathlib.Path, output: pathlib.Path, model: str, max_tokens: int) -> dict[str, Any]:
    papers = load_papers(input_path)
    bs = batches(papers)
    if len(bs) > MAX_CALLS:
        raise ValueError(f"input requires {len(bs)} calls; maximum is {MAX_CALLS}")
    config = {"model": model, "max_tokens": max_tokens, "reasoning_enabled": False,
              "papers_per_batch": PAPERS_PER_BATCH, "max_calls": MAX_CALLS,
              "input_sha256": _sha(papers), "paper_count": len(papers),
              "batch_count": len(bs), "protocol": "quote-first extraction v2; 2-4 atoms per paper; no novelty claims"}
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != config:
        raise ValueError("Plan changed; use a fresh output directory")
    plan_path.write_text(json.dumps(config, indent=2) + "\n")
    return config


def run(input_path: pathlib.Path, output: pathlib.Path, model: str, max_tokens: int) -> None:
    papers = load_papers(input_path)
    config = plan(input_path, output, model, max_tokens)
    if config["batch_count"] > MAX_CALLS:
        raise ValueError("call bound exceeded")
    calls = output / "calls"; calls.mkdir(exist_ok=True)
    atoms_path = output / "atoms.jsonl"
    lock_path = output / "run.lock"
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for index, batch in enumerate(batches(papers)):
            prompt = build_prompt(batch)
            request = {"model": model, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0, "reasoning": {"enabled": False},
                       "provider": {"allow_fallbacks": False}}
            fingerprint = _sha({"model": model, "input_sha256": _sha(batch), "request": request})
            path = calls / f"batch-{index:03d}.json"
            if path.exists():
                previous = json.loads(path.read_text())
                if previous.get("fingerprint") != fingerprint:
                    raise ValueError(f"cached request mismatch: batch {index}")
                if previous.get("status") != "complete":
                    raise ValueError(f"cached batch {index} is not complete; no automatic retry")
                continue
            record = {"status": "pending", "fingerprint": fingerprint, "request": request,
                      "prompt": prompt, "batch_index": index, "timestamp": time.time()}
            path.write_text(json.dumps(record, indent=2))
            try:
                req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions",
                    data=json.dumps(request).encode(), headers={"Authorization": "Bearer " + key(), "Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=180) as response:
                    data = json.load(response)
                choice = data["choices"][0]
                record.update(status="complete", response_id=data.get("id"), resolved_model=data.get("model"),
                              usage=data.get("usage"), finish_reason=choice.get("finish_reason"),
                              raw_response=choice.get("message", {}).get("content"))
                if choice.get("finish_reason") != "stop" or not record["raw_response"]:
                    raise ValueError(f"incomplete/provider refusal: {choice.get('finish_reason')}")
                parsed = _parse_json(record["raw_response"])
                extracted = validate_atoms(parsed, batch)
                with atoms_path.open("a", encoding="utf-8") as stream:
                    for atom in extracted:
                        stream.write(json.dumps(atom, ensure_ascii=False) + "\n")
                record["atom_count"] = len(extracted)
            except Exception as exc:
                record.update(status="failed", error=type(exc).__name__ + ": " + str(exc)[:300])
                raise
            finally:
                path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    for mode in ("plan", "run"):
        p = sub.add_parser(mode); p.add_argument("--input", type=pathlib.Path, required=True); p.add_argument("--output", type=pathlib.Path, required=True)
        p.add_argument("--model", default=DEFAULT_MODEL); p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    args = parser.parse_args()
    if not 512 <= args.max_tokens <= 4096:
        parser.error("--max-tokens must be between 512 and 4096")
    if args.mode == "plan":
        print(json.dumps(plan(args.input, args.output, args.model, args.max_tokens), indent=2))
    else:
        run(args.input, args.output, args.model, args.max_tokens)


if __name__ == "__main__":
    main()
