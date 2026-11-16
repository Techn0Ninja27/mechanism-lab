"""Bounded, evidence-preserving prior-art retrieval.

This module is deliberately retrieval-only: it never calls a generation model,
downloads arbitrary URLs, or emits a novelty judgment.  Local corpus search is
always performed first; arXiv is an explicitly opt-in, tightly bounded
fallback.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

SCHEMA_VERSION = "prior-art-retrieval-v3"
ARXIV_ENDPOINT = "https://export.arxiv.org/api/query"
ARXIV_NS = {"a": "http://www.w3.org/2005/Atom"}


class CacheMismatchError(RuntimeError):
    """A cache entry exists but was produced with different parameters."""


def _queries(values: Iterable[str]) -> tuple[str, ...]:
    if isinstance(values, str):
        raise ValueError("queries must be an explicit iterable of 1-3 strings, not one document")
    q = tuple(x.strip() if isinstance(x, str) else "" for x in values)
    if not 1 <= len(q) <= 3 or any(not x for x in q):
        raise ValueError("provide 1-3 non-empty query strings")
    if any(len(x) > 200 for x in q):
        raise ValueError("each query must be at most 200 characters")
    if any(not re.findall(r"[\w]+", x, flags=re.UNICODE) for x in q):
        raise ValueError("each query must contain at least one searchable token")
    return q


def _fts_query(query: str) -> str:
    # Never pass user punctuation/operators to FTS5.  Quoted tokens also make
    # terms such as OR, NEAR and asterisks inert rather than executable syntax.
    tokens = re.findall(r"[\w]+", query, flags=re.UNICODE)
    if not tokens:
        raise ValueError("query has no searchable tokens")
    return " AND ".join('"' + t.replace('"', '""') + '"' for t in tokens)

def _arxiv_query(query: str) -> str:
    tokens = re.findall(r"[\w]+", query, flags=re.UNICODE)
    if not tokens:
        raise ValueError("query has no searchable tokens")
    return " AND ".join('all:"' + t + '"' for t in tokens)


def _sha(title: str, abstract: str) -> str:
    return hashlib.sha256((title.strip().casefold() + "\n" + abstract.strip().casefold()).encode()).hexdigest()


def _arxiv_id(identifier: str) -> str:
    value = identifier.strip()
    for marker in ("/abs/", "/pdf/"):
        if marker in value:
            value = value.split(marker, 1)[1]
            break
    return re.sub(r"v\d+$", "", value.removesuffix(".pdf"))


def _local_search(db_path: Path, query: str, limit: int, cutoff: str | None) -> list[dict[str, Any]]:
    uri = f"file:{urllib.parse.quote(str(db_path.resolve()))}?mode=ro"
    with sqlite3.connect(uri, uri=True) as db:
        date_clause = " AND (w.published_date IS NULL OR w.published_date = '' OR substr(w.published_date,1,10) <= ?) AND (w.updated_date IS NULL OR w.updated_date = '' OR substr(w.updated_date,1,10) <= ?)" if cutoff else ""
        params = [_fts_query(query)] + ([cutoff, cutoff] if cutoff else []) + [int(limit)]
        try:
            rows = db.execute("""SELECT w.source_id,w.canonical_id,w.title,w.abstract,w.published_date,
                w.updated_date,w.landing_url,w.pdf_url,w.content_sha256,w.evidence_level
                FROM works_fts f JOIN works w ON w.source_id=f.source_id
                WHERE works_fts MATCH ?""" + date_clause + " ORDER BY bm25(works_fts), w.source_id LIMIT ?", params).fetchall()
        except sqlite3.OperationalError as exc:
            raise RuntimeError(f"corpus lacks expected works/works_fts schema: {exc}") from exc
    out = []
    for row in rows:
        sid, cid, title, abstract, published, updated, landing, pdf, digest, evidence = row
        if cutoff and ((published and str(published)[:10] > cutoff) or (updated and str(updated)[:10] > cutoff)):
            continue
        out.append({"source_id": sid, "canonical_id": cid, "title": title or "", "abstract": abstract or "",
                    "url": landing or pdf, "published_date": (published or "")[:10], "updated_date": (updated or "")[:10],
                    "evidence_level": evidence or "abstract_only", "source_hash": digest or _sha(title or "", abstract or ""),
                    "source": "local_corpus", "matched_queries": [query]})
    return out


def _parse_arxiv(payload: bytes, query: str) -> list[dict[str, Any]]:
    root = ET.fromstring(payload)
    out = []
    for e in root.findall("a:entry", ARXIV_NS):
        raw = (e.findtext("a:id", namespaces=ARXIV_NS) or "").strip()
        if not raw: continue
        aid = _arxiv_id(raw)
        title = re.sub(r"\s+", " ", e.findtext("a:title", namespaces=ARXIV_NS) or "").strip()
        abstract = re.sub(r"\s+", " ", e.findtext("a:summary", namespaces=ARXIV_NS) or "").strip()
        published = (e.findtext("a:published", namespaces=ARXIV_NS) or "")[:10]
        updated = (e.findtext("a:updated", namespaces=ARXIV_NS) or "")[:10]
        out.append({"source_id": f"arxiv:{aid}", "canonical_id": f"arxiv:{aid}", "title": title,
                    "abstract": abstract, "url": f"https://arxiv.org/abs/{aid}", "published_date": published,
                    "updated_date": updated, "evidence_level": "abstract_only", "source_hash": _sha(title, abstract),
                    "source": "arxiv_api", "matched_queries": [query]})
    return out


def _cache_key(query: str, limit: int, provider: str, schema: str, cutoff: str | None) -> str:
    value = json.dumps({"query": query, "limit": limit, "provider": provider, "schema": schema, "cutoff": cutoff}, sort_keys=True)
    return hashlib.sha256(value.encode()).hexdigest()


def _arxiv(query: str, limit: int, cutoff: str | None, cache_dir: Path, *, timeout: float, max_bytes: int,
           wait_before: Callable[[], None] | None, opener: Callable[..., Any] | None, failures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    provider_limit = min(int(limit), 5)
    compiled = _arxiv_query(query)
    key = _cache_key(compiled, provider_limit, "arxiv_api", SCHEMA_VERSION, cutoff)
    cache_dir.mkdir(parents=True, exist_ok=True)
    body_path, meta_path = cache_dir / f"{key}.xml", cache_dir / f"{key}.json"
    params = {"query": query, "compiled_query": compiled, "limit": provider_limit, "provider": "arxiv_api", "schema": SCHEMA_VERSION, "cutoff": cutoff}
    try:
        if body_path.exists() or meta_path.exists():
            if not (body_path.exists() and meta_path.exists()):
                raise CacheMismatchError(f"incomplete cache for {query!r}")
            meta = json.loads(meta_path.read_text())
            if meta.get("params") != params:
                raise CacheMismatchError(f"cached parameters differ for {query!r}")
            payload = body_path.read_bytes()
            if meta.get("response_sha256") != hashlib.sha256(payload).hexdigest() or meta.get("response_bytes") != len(payload):
                raise CacheMismatchError(f"cached response integrity check failed for {query!r}")
        else:
            if wait_before: wait_before()
            url = ARXIV_ENDPOINT + "?" + urllib.parse.urlencode({"search_query": compiled, "start": 0, "max_results": provider_limit})
            req = urllib.request.Request(url, headers={"User-Agent": "alien-science-prior-art/1.0"})
            open_fn = opener or urllib.request.urlopen
            with open_fn(req, timeout=timeout) as response:
                payload = response.read(max_bytes + 1)
            if len(payload) > max_bytes:
                raise ValueError(f"arXiv response exceeded {max_bytes} bytes")
            body_path.write_bytes(payload)
            meta_path.write_text(json.dumps({"params": params, "url": url, "response_sha256": hashlib.sha256(payload).hexdigest(), "response_bytes": len(payload)}, sort_keys=True))
        rows = _parse_arxiv(payload, query)
        if cutoff:
            rows = [r for r in rows if not ((r["published_date"] and r["published_date"] > cutoff) or (r["updated_date"] and r["updated_date"] > cutoff))]
        return rows[:provider_limit]
    except CacheMismatchError:
        raise
    except Exception as exc:
        failures.append({"provider": "arxiv_api", "query": query, "error_type": type(exc).__name__, "error": str(exc)})
        return []


def retrieve_prior_art(queries: Iterable[str], corpus_db: str | Path, *, limit: int = 20, cutoff: str | None = None,
                       allow_arxiv: bool = False, cache_dir: str | Path | None = None, artifact_path: str | Path | None = None,
                       timeout: float = 20.0, max_response_bytes: int = 1_000_000, min_interval: float = 3.0,
                       opener: Callable[..., Any] | None = None) -> dict[str, Any]:
    qs = _queries(queries)
    if artifact_path is not None and Path(artifact_path).exists():
        raise FileExistsError(f"immutable retrieval artifact already exists: {artifact_path}")
    if not isinstance(limit, int) or not 1 <= limit <= 20: raise ValueError("limit must be an integer from 1 to 20")
    if min_interval < 3: raise ValueError("min_interval must be at least 3 seconds")
    if not 0 < timeout <= 120: raise ValueError("timeout must be in (0, 120]")
    if not 0 < max_response_bytes <= 5_000_000: raise ValueError("max_response_bytes must be in (0, 5000000]")
    cutoff = cutoff or date.today().isoformat()
    failures: list[dict[str, Any]] = []
    records: dict[str, dict[str, Any]] = {}
    coverage: list[dict[str, Any]] = []
    for q in qs:
        try:
            local_rows = _local_search(Path(corpus_db), q, limit, cutoff)
            coverage.append({"query": q, "compiled_query": _fts_query(q), "provider": "local_corpus", "records": len(local_rows), "status": "ok"})
        except Exception as exc:
            # A missing/invalid local corpus must be visible to the caller; an
            # empty result would falsely suggest that the corpus was searched.
            local_rows = []
            failures.append({"provider": "local_corpus", "query": q, "error_type": type(exc).__name__, "error": str(exc)})
            coverage.append({"query": q, "compiled_query": _fts_query(q), "provider": "local_corpus", "records": 0, "status": "error"})
        for row in local_rows:
            existing = records.get(row["canonical_id"] or row["source_id"])
            if not existing:
                existing = next((x for x in records.values() if _sha(x["title"], x["abstract"]) == _sha(row["title"], row["abstract"])), None)
            if existing: existing["matched_queries"] = sorted(set(existing["matched_queries"] + [q]))
            else: records[row["canonical_id"] or row["source_id"]] = row
    last_uncached = [0.0]
    def wait() -> None:
        now = time.monotonic(); delay = min_interval - (now - last_uncached[0])
        if delay > 0: time.sleep(delay)
        last_uncached[0] = time.monotonic()
    if allow_arxiv:
        for q in qs:
            before = len(records)
            for row in _arxiv(q, limit, cutoff, Path(cache_dir or ".prior-art-cache"), timeout=timeout, max_bytes=max_response_bytes, wait_before=wait, opener=opener, failures=failures):
                key = row["canonical_id"]
                existing = records.get(key) or next((x for x in records.values() if _sha(x["title"], x["abstract"]) == _sha(row["title"], row["abstract"])), None)
                if existing: existing["matched_queries"] = sorted(set(existing["matched_queries"] + [q]))
                else: records[key] = row
            coverage.append({"query": q, "compiled_query": _arxiv_query(q), "provider": "arxiv_api", "records": len(records) - before,
                             "status": "error" if any(x["provider"] == "arxiv_api" and x["query"] == q for x in failures) else "ok"})
    all_rows = list(records.values())
    # Interleave query/provider buckets so broad early queries cannot consume
    # the entire retained evidence budget. Preserve rank within each bucket.
    providers = ["local_corpus"] + (["arxiv_api"] if allow_arxiv else [])
    buckets = [[r for r in all_rows if q in r.get("matched_queries", []) and r.get("source") == provider]
               for q in qs for provider in providers]
    ordered, seen = [], set()
    while any(buckets):
        for bucket in buckets:
            while bucket:
                row = bucket.pop(0)
                identity = row.get("canonical_id") or row["source_id"]
                if identity not in seen:
                    seen.add(identity); ordered.append(row)
                    break
    all_rows = ordered
    retained = all_rows[:limit]
    for item in coverage:
        item["retained_records"] = sum(1 for r in retained if item["query"] in r.get("matched_queries", []) and r.get("source") == item["provider"])
    result = {"schema": SCHEMA_VERSION, "query_hash": hashlib.sha256(json.dumps(qs, ensure_ascii=False).encode()).hexdigest(),
              "queries": list(qs), "cutoff": cutoff, "limit": limit, "providers": ["local_corpus"] + (["arxiv_api"] if allow_arxiv else []),
              "records": retained, "failures": failures,
              "errors": failures, "coverage": coverage,
              "truncation": {"truncated": len(records) > limit, "limit": limit, "arxiv_max_per_query": 5},
              "novelty_label": None}
    result["digest"] = hashlib.sha256(json.dumps(result, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if artifact_path:
        path = Path(artifact_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as fh: json.dump(result, fh, indent=2, ensure_ascii=False); fh.write("\n")
    return result


def retrieve(queries: Iterable[str], output: str | Path | None = None, *,
             corpus_path: str | Path = "results/corpus-expanded/corpus.sqlite",
             allow_arxiv: bool = False, cutoff: str = "2026-11-16", limit: int = 20,
             cache_dir: str | Path | None = None, **kwargs: Any) -> dict[str, Any]:
    """Parent-process API; ``output`` is an immutable artifact path."""
    return retrieve_prior_art(queries, corpus_path, limit=limit, cutoff=cutoff,
                              allow_arxiv=allow_arxiv, cache_dir=cache_dir,
                              artifact_path=output, **kwargs)


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--db", type=Path, required=True); p.add_argument("--query", action="append", required=True); p.add_argument("--limit", type=int, default=20); p.add_argument("--cutoff"); p.add_argument("--allow-arxiv", action="store_true"); p.add_argument("--cache-dir", type=Path); p.add_argument("--artifact", type=Path, required=True)
    a = p.parse_args(); print(json.dumps(retrieve_prior_art(a.query, a.db, limit=a.limit, cutoff=a.cutoff, allow_arxiv=a.allow_arxiv, cache_dir=a.cache_dir, artifact_path=a.artifact), indent=2))


if __name__ == "__main__": main()
