#!/usr/bin/env python3
"""Build a reproducible public research-metadata corpus.

The ingester combines the local paper metadata with a small, cached arXiv API
harvest. It stores metadata and abstracts only; it does not scrape Lacuna,
mirror a proprietary service, download PDFs, or infer novelty labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import ssl
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path
from typing import Iterable, Mapping


ARXIV_NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
DEFAULT_QUERIES = (
    'all:"continual learning" AND all:memory',
    'all:"model editing" OR all:"knowledge editing"',
    'all:"mechanistic interpretability" AND all:representation',
    'all:"adaptive computation" OR all:"early exit"',
    'all:"library learning" OR all:"program synthesis"',
    'all:"quality diversity" OR all:"novelty search"',
    'all:"world model" AND all:agent',
    'all:"scientific discovery" AND all:language model',
    'all:"research idea" AND all:language model',
    'all:dataset AND all:robustness AND all:missingness',
    'all:"large language model" AND all:China',
    'all:agent AND all:continual AND all:memory',
)


def clean(text: str | None) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def norm_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.casefold()).strip()


def content_sha(title: str, abstract: str) -> str:
    return hashlib.sha256((clean(title).casefold() + "\n" + clean(abstract).casefold()).encode()).hexdigest()


def arxiv_canonical_id(identifier: str) -> str:
    """Normalize arXiv URLs/versions to one canonical identifier."""
    value = identifier.strip()
    for marker in ("/abs/", "/pdf/"):
        if marker in value:
            value = value.split(marker, 1)[1]
            break
    value = value.strip().removesuffix(".pdf")
    value = re.sub(r"v\d+$", "", value)
    return value


def parse_arxiv(xml_bytes: bytes, *, query: str, retrieved_at: str) -> list[dict]:
    root = ET.fromstring(xml_bytes)
    rows = []
    for entry in root.findall("a:entry", ARXIV_NS):
        raw_id = clean(entry.findtext("a:id", namespaces=ARXIV_NS))
        if not raw_id:
            continue
        aid = arxiv_canonical_id(raw_id)
        title = clean(entry.findtext("a:title", namespaces=ARXIV_NS))
        abstract = clean(entry.findtext("a:summary", namespaces=ARXIV_NS))
        published = clean(entry.findtext("a:published", namespaces=ARXIV_NS))[:10]
        updated = clean(entry.findtext("a:updated", namespaces=ARXIV_NS))[:10]
        authors = [clean(x.findtext("a:name", namespaces=ARXIV_NS)) for x in entry.findall("a:author", ARXIV_NS)]
        links = [x.attrib for x in entry.findall("a:link", ARXIV_NS)]
        pdf = next((x.get("href") for x in links if x.get("title") == "pdf"), None)
        doi = clean(entry.findtext("arxiv:doi", namespaces=ARXIV_NS)) or None
        rows.append({
            "source_id": f"arxiv:{aid}", "canonical_id": f"arxiv:{aid}",
            "version": raw_id.rsplit("/", 1)[-1], "title": title, "abstract": abstract,
            "authors": authors, "published_date": published, "updated_date": updated,
            "landing_url": raw_id, "pdf_url": pdf or raw_id, "doi": doi,
            "query": query, "retrieved_at": retrieved_at,
            "content_sha256": content_sha(title, abstract), "evidence_level": "abstract_only",
            "retrieval_method": "arxiv_api", "full_text_available": False,
        })
    return rows


def future_filtered(rows: Iterable[Mapping], cutoff: str) -> tuple[list[dict], int]:
    kept, dropped = [], 0
    for row in rows:
        published = str(row.get("published_date", ""))[:10]
        updated = str(row.get("updated_date", ""))[:10]
        if (published and published > cutoff) or (updated and updated > cutoff):
            dropped += 1
        else:
            kept.append(dict(row))
    return kept, dropped


def deduplicate(rows: Iterable[Mapping]) -> tuple[list[dict], dict[str, str]]:
    """Deduplicate versions and exact title/abstract duplicates deterministically."""
    canonical: dict[str, dict] = {}
    aliases: dict[str, str] = {}
    for source in rows:
        row = dict(source)
        key = str(row.get("canonical_id") or "")
        if not key:
            key = "title:" + norm_title(str(row.get("title", "")))
        current = canonical.get(key)
        if current is None or (row.get("updated_date", "") > current.get("updated_date", "")):
            canonical[key] = row
        aliases[str(row.get("source_id", key))] = key
    # Second pass catches exact title/abstract duplicates from different IDs.
    by_content: dict[str, str] = {}
    final: dict[str, dict] = {}
    for key, row in sorted(canonical.items()):
        digest = row.get("content_sha256") or content_sha(row.get("title", ""), row.get("abstract", ""))
        target = by_content.setdefault(digest, key)
        aliases[key] = target
        final.setdefault(target, row)
    def resolve(key: str) -> str:
        seen = set()
        while aliases.get(key, key) != key:
            if key in seen:
                raise ValueError(f"alias cycle at {key}")
            seen.add(key); key = aliases[key]
        return key
    return list(final.values()), {alias: resolve(target) for alias, target in aliases.items()}


def _local_rows(db_path: Path) -> list[dict]:
    with sqlite3.connect(db_path) as db:
        rows = db.execute("SELECT paper_id,title,abstract,pdf_url,conference,venue_year,openreview_url FROM papers").fetchall()
    out = []
    for pid, title, abstract, pdf, conference, year, landing in rows:
        out.append({
            "source_id": f"local:{pid}", "canonical_id": f"local:{pid}", "version": "local",
            "title": clean(title), "abstract": clean(abstract), "authors": [],
            "published_date": f"{int(year):04d}-01-01" if year else "", "date_precision": "year" if year else "unknown",
            "updated_date": "", "landing_url": landing or pdf, "pdf_url": pdf,
            "venue": conference, "retrieval_method": "local_sqlite", "retrieved_at": date.today().isoformat(),
            "content_sha256": content_sha(title, abstract), "evidence_level": "abstract_only",
            "full_text_available": False,
        })
    return out


def _existing_corpus_rows(path: Path, cutoff: str) -> list[dict]:
    """Read a prior works.jsonl snapshot without mutating it."""
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, Mapping) or not row.get("source_id"):
            raise ValueError(f"invalid existing corpus row at {path}:{line_no}")
        published = str(row.get("published_date", ""))[:10]
        updated = str(row.get("updated_date", ""))[:10]
        if (published and published > cutoff) or (updated and updated > cutoff):
            continue
        rows.append(dict(row))
    return rows


def harvest(queries: Iterable[str], cache_dir: Path, *, max_new: int, cutoff: str, sleep_seconds: float = 3.0, prior_requests: Mapping[str, str] | None = None) -> tuple[list[dict], list[dict]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows, logs = [], []
    context = ssl.create_default_context()
    queries = tuple(queries)
    seen_ids: set[str] = set()
    for index, query in enumerate(queries):
        if len(seen_ids) >= max_new:
            break
        cache = cache_dir / f"query-{index:02d}.xml"
        request_url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode({"search_query": query, "start": 0, "max_results": min(25, max_new)})
        request_hash = hashlib.sha256(request_url.encode()).hexdigest()
        meta = cache.with_suffix(cache.suffix + ".json")
        prior_url = (prior_requests or {}).get(str(cache))
        valid_cache = cache.exists() and ((meta.exists() and json.loads(meta.read_text()).get("request_sha256") == request_hash) or prior_url == request_url)
        if valid_cache:
            payload = cache.read_bytes(); cached = True
        else:
            request = urllib.request.Request(request_url, headers={"User-Agent": "alien-science-corpus/1.0 research metadata"})
            with urllib.request.urlopen(request, timeout=60, context=context) as response:
                payload = response.read()
            cache.write_bytes(payload); meta.write_text(json.dumps({"request_url": request_url, "request_sha256": request_hash}, indent=2)); cached = False
            if index + 1 < len(tuple(queries)):
                time.sleep(sleep_seconds)
        digest = hashlib.sha256(payload).hexdigest()
        parsed = parse_arxiv(payload, query=query, retrieved_at=date.today().isoformat())
        parsed, dropped = future_filtered(parsed, cutoff)
        fresh = [row for row in parsed if row["source_id"] not in seen_ids]
        rows.extend(fresh[: max(0, max_new - len(seen_ids))])
        seen_ids.update(row["source_id"] for row in fresh[: max(0, max_new - len(seen_ids))])
        if cached and not meta.exists():
            meta.write_text(json.dumps({"request_url": request_url, "request_sha256": request_hash}, indent=2))
        logs.append({"query": query, "url": request_url, "cache": str(cache), "cached": cached, "request_sha256": request_hash, "response_sha256": digest, "records": len(parsed), "future_dropped": dropped})
    return rows, logs


def _load_openai_sources(path: Path | None, cutoff: str) -> list[dict]:
    if path is None or not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        published = str(item.get("date", ""))[:10]
        if published and published > cutoff:
            continue
        raw_aid = item.get("arxiv_id")
        aid = clean(str(raw_aid)) if raw_aid else ""
        source_id = f"openai:{aid}" if aid else "openai:url:" + hashlib.sha256(str(item.get("url", "")).encode()).hexdigest()[:16]
        rows.append({
            "source_id": source_id, "canonical_id": f"arxiv:{arxiv_canonical_id(aid)}" if aid else source_id,
            "title": clean(item.get("title")), "abstract": "", "authors": [],
            "published_date": published, "updated_date": "", "landing_url": item.get("url"),
            "pdf_url": item.get("paper_url"), "doi": None, "venue": "OpenAI official release",
            "retrieval_method": "openai_official_metadata", "retrieved_at": date.today().isoformat(),
            "content_sha256": content_sha(str(item.get("title", "")), str(item.get("mechanism_summary", ""))),
            "evidence_level": "generated_summary", "generated_summary": clean(item.get("mechanism_summary")),
            "limits": clean(item.get("limits")), "evidence_status": item.get("evidence_status"),
            "full_text_available": False,
        })
    return rows


def _packet_items(raw: object) -> list[Mapping]:
    """Return records from the supported bounded packet shapes."""
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, Mapping)]
    if isinstance(raw, Mapping):
        for key in ("papers", "sources", "records", "items"):
            value = raw.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, Mapping)]
    raise ValueError("packet must be a JSON list or an object containing papers/sources/records/items")


def _load_packet(path: Path, cutoff: str) -> tuple[list[dict], dict]:
    """Load an offline primary-abstract packet and its file-level provenance.

    Packet provenance is intentionally kept apart from generated OpenAI
    summaries.  The packet's ``abstract`` is copied only when supplied by the
    packet; no summary/mechanism field is promoted into the abstract column.
    """
    payload = path.read_bytes()
    file_sha = hashlib.sha256(payload).hexdigest()
    raw = json.loads(payload.decode("utf-8"))
    rows: list[dict] = []
    dropped = 0
    for item in _packet_items(raw):
        raw_id = clean(str(item.get("id") or item.get("arxiv_id") or ""))
        if not raw_id:
            raise ValueError(f"packet record missing id in {path}")
        canonical = arxiv_canonical_id(raw_id)
        is_arxiv = bool(re.fullmatch(r"(?:\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?/\d{7})", canonical))
        canonical_id = f"arxiv:{canonical}" if is_arxiv else (f"packet:{canonical}" if not canonical.startswith(("arxiv:", "doi:")) else canonical)
        source_id = f"arxiv:{raw_id}" if is_arxiv else f"packet:{raw_id}"
        published = clean(str(item.get("published") or item.get("published_date") or item.get("publication_date") or item.get("date") or ""))[:10]
        updated = clean(str(item.get("updated") or item.get("updated_date") or ""))[:10]
        if (published and published > cutoff) or (updated and updated > cutoff):
            dropped += 1
            continue
        title = clean(str(item.get("title") or ""))
        abstract = clean(str(item.get("abstract") or ""))
        url = clean(str(item.get("url") or item.get("primary_url") or ""))
        version = raw_id.rsplit("v", 1)[-1] if re.search(r"v\d+$", raw_id) else ""
        rows.append({
            "source_id": source_id, "canonical_id": canonical_id, "version": f"v{version}" if version else "",
            "title": title, "abstract": abstract, "authors": item.get("authors", []) if isinstance(item.get("authors", []), list) else [],
            "published_date": published, "updated_date": updated, "landing_url": url,
            "pdf_url": item.get("pdf_url") or (url.replace("/abs/", "/pdf/") if "/abs/" in url else None),
            "doi": item.get("doi"), "venue": item.get("venue") or "arXiv packet",
            "retrieval_method": "offline_packet", "retrieved_at": date.today().isoformat(),
            "content_sha256": content_sha(title, abstract), "evidence_level": item.get("evidence_level") or "primary_abstract",
            "full_text_available": False, "query": None, "date_precision": "day" if published else "unknown",
            "source_file": str(path), "source_file_sha256": file_sha,
        })
    provenance = {"path": str(path), "sha256": file_sha, "records": len(rows), "future_dropped": dropped,
                  "unknown_date_precision": sum(row.get("date_precision") == "unknown" for row in rows)}
    return rows, provenance


def write_corpus(rows: list[dict], aliases: dict[str, str], logs: list[dict], output: Path, *, cutoff: str, openai_sources: Path | None = None, packet_manifests: list[dict] | None = None) -> None:
    output.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(output / "corpus.sqlite")
    db.executescript("""
    CREATE TABLE IF NOT EXISTS works (source_id TEXT PRIMARY KEY, canonical_id TEXT, title TEXT, abstract TEXT, authors_json TEXT, published_date TEXT, updated_date TEXT, landing_url TEXT, pdf_url TEXT, doi TEXT, venue TEXT, retrieval_method TEXT, retrieved_at TEXT, content_sha256 TEXT, evidence_level TEXT, full_text_available INTEGER, query TEXT, date_precision TEXT);
    CREATE TABLE IF NOT EXISTS aliases (alias TEXT PRIMARY KEY, canonical_id TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS query_log (query TEXT, url TEXT, cache TEXT, cached INTEGER, response_sha256 TEXT, records INTEGER, future_dropped INTEGER);
    CREATE TABLE IF NOT EXISTS openai_sources (source_id TEXT PRIMARY KEY, canonical_id TEXT, title TEXT, url TEXT, paper_url TEXT, published_date TEXT, generated_summary TEXT, limits TEXT, evidence_status TEXT, content_sha256 TEXT, evidence_level TEXT);
    CREATE TABLE IF NOT EXISTS packet_sources (path TEXT PRIMARY KEY, sha256 TEXT NOT NULL, records INTEGER NOT NULL, future_dropped INTEGER NOT NULL, snapshot_manifest TEXT NOT NULL);
    CREATE VIRTUAL TABLE IF NOT EXISTS works_fts USING fts5(source_id UNINDEXED, title, abstract);
    """)
    try: db.execute("ALTER TABLE works ADD COLUMN date_precision TEXT")
    except sqlite3.OperationalError: pass
    for column in ("source_file", "source_file_sha256"):
        try: db.execute(f"ALTER TABLE works ADD COLUMN {column} TEXT")
        except sqlite3.OperationalError: pass
    db.execute("DELETE FROM works"); db.execute("DELETE FROM aliases"); db.execute("DELETE FROM query_log"); db.execute("DELETE FROM works_fts"); db.execute("DELETE FROM packet_sources")
    for row in rows:
        db.execute("INSERT OR REPLACE INTO works(source_id,canonical_id,title,abstract,authors_json,published_date,updated_date,landing_url,pdf_url,doi,venue,retrieval_method,retrieved_at,content_sha256,evidence_level,full_text_available,query,date_precision,source_file,source_file_sha256) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (row.get("source_id"), row.get("canonical_id"), row.get("title"), row.get("abstract"), json.dumps(row.get("authors", [])), row.get("published_date"), row.get("updated_date"), row.get("landing_url"), row.get("pdf_url"), row.get("doi"), row.get("venue"), row.get("retrieval_method"), row.get("retrieved_at"), row.get("content_sha256"), row.get("evidence_level"), int(bool(row.get("full_text_available"))), row.get("query"), row.get("date_precision"), row.get("source_file"), row.get("source_file_sha256")))
        db.execute("INSERT OR REPLACE INTO works_fts(source_id,title,abstract) VALUES (?,?,?)", (row.get("source_id"), row.get("title"), row.get("abstract")))
    db.executemany("INSERT OR REPLACE INTO aliases VALUES (?,?)", aliases.items())
    db.executemany("INSERT INTO query_log VALUES (?,?,?,?,?,?,?)", [(x.get("query"), x.get("url"), x.get("cache"), int(x.get("cached", False)), x.get("response_sha256"), x.get("records", 0), x.get("future_dropped", 0)) for x in logs])
    openai_rows = _load_openai_sources(openai_sources, cutoff)
    db.execute("DELETE FROM openai_sources")
    db.executemany("INSERT OR REPLACE INTO openai_sources VALUES (?,?,?,?,?,?,?,?,?,?,?)", [(x["source_id"], x["canonical_id"], x["title"], x["landing_url"], x.get("pdf_url"), x["published_date"], x.get("generated_summary", ""), x.get("limits", ""), x.get("evidence_status", ""), x["content_sha256"], x["evidence_level"]) for x in openai_rows])
    for packet in packet_manifests or []:
        db.execute("INSERT OR REPLACE INTO packet_sources VALUES (?,?,?,?,?)", (packet["path"], packet["sha256"], packet["records"], packet["future_dropped"], json.dumps(packet, sort_keys=True)))
    db.commit(); db.close()
    (output / "works.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")
    (output / "aliases.json").write_text(json.dumps(aliases, indent=2), encoding="utf-8")
    (output / "openai_sources.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in openai_rows), encoding="utf-8")
    manifest = {"cutoff": cutoff, "records": len(rows), "openai_metadata_records": len(openai_rows), "provenance_counts": {}, "query_count": len(logs), "source_hash": hashlib.sha256((output / "works.jsonl").read_bytes()).hexdigest(), "openai_sources": str(openai_sources) if openai_sources else None, "packet_sources": packet_manifests or []}
    for row in rows: manifest["provenance_counts"][row.get("retrieval_method", "unknown")] = manifest["provenance_counts"].get(row.get("retrieval_method", "unknown"), 0) + 1
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--db", type=Path, default=Path("data/llm-papers.db")); p.add_argument("--output", type=Path, default=Path("results/corpus-expanded")); p.add_argument("--cache", type=Path, default=Path("results/corpus-expanded/arxiv-cache")); p.add_argument("--cutoff", default="2026-11-16"); p.add_argument("--max-new", type=int, default=300); p.add_argument("--sleep", type=float, default=3.0); p.add_argument("--openai-sources", type=Path, default=Path("docs/corpus-expansion/OPENAI_SOURCES.json")); p.add_argument("--query", action="append", dest="queries"); p.add_argument("--packet", type=Path, action="append", default=[], help="bounded JSON packet; repeatable"); p.add_argument("--existing-corpus", type=Path, help="read-only works.jsonl snapshot to extend"); p.add_argument("--offline", action="store_true", help="do not call arXiv; use only the selected local snapshot and packets")
    a = p.parse_args(); queries = tuple(a.queries or DEFAULT_QUERIES)
    if len(queries) > 12: raise SystemExit("at most 12 arXiv queries")
    prior_requests = {}
    prior_db = a.output / "corpus.sqlite"
    if prior_db.exists():
        with sqlite3.connect(prior_db) as db:
            prior_requests = {str(cache): str(url) for cache, url in db.execute("SELECT cache,url FROM query_log") if cache and url}
    local = _existing_corpus_rows(a.existing_corpus, a.cutoff) if a.existing_corpus else _local_rows(a.db)
    harvested, logs = ([], []) if a.offline else harvest(queries, a.cache, max_new=min(a.max_new, 300), cutoff=a.cutoff, sleep_seconds=a.sleep, prior_requests=prior_requests)
    packet_rows, packet_manifests = [], []
    for packet in a.packet:
        loaded, provenance = _load_packet(packet, a.cutoff)
        packet_rows.extend(loaded); packet_manifests.append(provenance)
    all_rows, aliases = deduplicate(local + harvested + packet_rows)
    write_corpus(all_rows, aliases, logs, a.output, cutoff=a.cutoff, openai_sources=a.openai_sources if a.openai_sources.exists() else None, packet_manifests=packet_manifests)
    print(json.dumps({"base_rows": len(local), "local_rows": len(local) if not a.existing_corpus else 0, "existing_corpus_rows": len(local) if a.existing_corpus else 0, "harvest_rows": len(harvested), "packet_rows": len(packet_rows), "deduplicated_rows": len(all_rows), "queries": len(logs), "offline": a.offline, "output": str(a.output)}, indent=2))


if __name__ == "__main__": main()
