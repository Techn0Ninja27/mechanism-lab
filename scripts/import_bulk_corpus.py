#!/usr/bin/env python3
"""Import a bounded JSONL metadata snapshot into a new research corpus."""
from __future__ import annotations
import argparse, hashlib, heapq, json, re, shutil, sqlite3, sys
from collections import Counter
from datetime import date
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.ingest_research_corpus import arxiv_canonical_id, clean, content_sha, deduplicate, write_corpus

DEFAULT_GROUPS = {
    "ml": ("cs.LG", "cs.AI", "cs.CL", "cs.CV", "cs.NE", "stat.ML"),
    "systems": ("cs.DC", "cs.OS", "cs.DB", "cs.PF", "cs.NI", "cs.AR"),
    "math": ("math.", "stat.TH", "cs.IT", "cs.DS"),
}

def _date(value: object) -> str:
    text = clean(str(value or ""))
    if not text:
        return ""
    try:
        if len(text) == 4 and text.isdigit():
            return text
        if "T" in text:
            return text[:10]
        return parsedate_to_datetime(text).date().isoformat()
    except (TypeError, ValueError, OverflowError):
        return text[:10]

def _categories(raw: object) -> list[str]:
    if isinstance(raw, str):
        return sorted(set(clean(raw).split()))
    if isinstance(raw, list):
        return sorted(set(clean(str(x)) for x in raw if clean(str(x))))
    return []

def _version_date(versions: object, index: int) -> str:
    if not isinstance(versions, list) or not versions:
        return ""
    i = index if index >= 0 else len(versions) + index
    if not 0 <= i < len(versions) or not isinstance(versions[i], Mapping):
        return ""
    return _date(versions[i].get("created"))

def _abstract(raw: Mapping) -> str:
    value = raw.get("abstract")
    if isinstance(value, str):
        return clean(value)
    if isinstance(raw.get("abstract_inverted_index"), Mapping):
        positions = [(int(pos), word) for word, poss in raw["abstract_inverted_index"].items()
                     if isinstance(poss, list) for pos in poss if isinstance(pos, int)]
        return clean(" ".join(word for _, word in sorted(positions)))
    return ""

def normalize_record(raw: Mapping, *, source_file: str, source_file_sha256: str, cutoff: str) -> dict | None:
    raw_source_id = clean(str(raw.get("source_id") or ""))
    raw_canonical_id = clean(str(raw.get("canonical_id") or ""))
    raw_id = clean(str(raw.get("id") or raw.get("arxiv_id") or raw.get("openalex_id") or raw_source_id or raw_canonical_id))
    title = clean(str(raw.get("title") or raw.get("display_name") or ""))
    abstract = _abstract(raw)
    if not raw_id or not title or not abstract:
        return None
    is_openalex = raw_id.startswith(("https://openalex.org/", "W")) or raw_source_id.startswith("openalex:") or raw_canonical_id.startswith("https://openalex.org/") or "openalex_id" in raw
    canonical = raw_id.rsplit("/", 1)[-1] if is_openalex else arxiv_canonical_id(raw_id)
    canonical_id = raw_canonical_id if raw_canonical_id.startswith(("arxiv:", "openalex:", "doi:", "local:")) else (("openalex:" if is_openalex else "arxiv:") + canonical)
    if canonical_id.startswith("https://openalex.org/"):
        canonical_id = "openalex:" + canonical_id.rsplit("/", 1)[-1]
    elif canonical_id.startswith("openalex:openalex:"):
        canonical_id = canonical_id.removeprefix("openalex:")
    source_id = raw_source_id or (("openalex:" if is_openalex else "arxiv:") + raw_id)
    arxiv_id = clean(str(raw.get("arxiv_id") or ""))
    if not arxiv_id and isinstance(raw.get("ids"), Mapping):
        arxiv_id = clean(str(raw["ids"].get("arxiv") or ""))
    if not arxiv_id and isinstance(raw.get("locations"), list):
        for location in raw["locations"]:
            url = str(location.get("landing_page_url") or "") if isinstance(location, Mapping) else ""
            match = re.search(r"arxiv\.org/(?:abs|pdf)/([^/?#]+)", url)
            if match:
                arxiv_id = match.group(1)
                break
    versions = raw.get("versions")
    published = _version_date(versions, 0) or _date(raw.get("published") or raw.get("published_date"))
    updated = _version_date(versions, -1) or _date(raw.get("update_date") or raw.get("updated") or raw.get("updated_date"))
    if is_openalex:
        published = _date(raw.get("publication_date") or raw.get("publication_year")) or published
        updated = _date(raw.get("updated_date")) or updated
    if (published and published > cutoff) or (updated and updated > cutoff):
        return None
    authors = raw.get("authors", [])
    if isinstance(authors, list):
        normalized = []
        for author in authors:
            if isinstance(author, Mapping) and isinstance(author.get("author"), Mapping):
                value = author["author"].get("display_name", "")
            elif isinstance(author, Mapping):
                value = author.get("name", "")
            else:
                value = author
            if clean(str(value)):
                normalized.append(clean(str(value)))
        authors = normalized
    else:
        authors = []
    cats = _categories(raw.get("categories") or raw.get("category"))
    if is_openalex:
        cats = sorted(set(cats + [str(x.get("id", "")).rsplit("/", 1)[-1]
                                  for x in raw.get("topics", []) if isinstance(x, Mapping)]))
    primary = clean(str(raw.get("primary_category") or raw.get("primaryCategory") or (cats[0] if cats else "")))
    fallback_url = ("https://openalex.org/" + canonical_id.removeprefix("openalex:")) if is_openalex else f"https://arxiv.org/abs/{canonical}"
    landing = clean(str(raw.get("url") or raw.get("landing_url") or raw.get("doi") or fallback_url))
    slot = str(raw.get("query_slot") or "")
    group_name = raw.get("corpus_group") or slot.split(":")[0]
    group_name = {"machine_learning": "ml", "mathematics": "math"}.get(group_name, group_name)
    influence_fields = {"citation_count": raw.get("cited_by_count"),
                        "field_weighted_citation_impact": raw.get("fwci"),
                        "citation_normalized_percentile": raw.get("citation_normalized_percentile")}
    influence = []
    if any(value is not None for value in influence_fields.values()):
        influence.append({"provider": raw.get("retrieval_method") or ("openalex_api" if is_openalex else "unknown"),
                          "source_id": source_id, "as_of": raw.get("retrieved_at"),
                          "record_sha256": raw.get("source_hash"),
                          "source_file": source_file, "values": influence_fields})
    return {
        "source_id": source_id, "canonical_id": canonical_id,
        "version": raw_id.rsplit("v", 1)[-1] if raw_id.rsplit("v", 1)[-1].isdigit() else "",
        "title": title, "abstract": abstract, "authors": authors,
        "query_slot": slot, "corpus_group": group_name,
        "categories": cats, "primary_category": primary,
        "published_date": published, "updated_date": updated,
        "landing_url": landing, "pdf_url": raw.get("pdf_url"),
        "doi": raw.get("doi"), "venue": raw.get("journal-ref") or raw.get("venue"),
        "arxiv_id": arxiv_id,
        "retrieval_method": raw.get("retrieval_method") or "bulk_metadata_jsonl",
        "retrieved_at": raw.get("retrieved_at") or date.today().isoformat(),
        "content_sha256": content_sha(title, abstract),
        "evidence_level": "indexed_abstract" if is_openalex else "primary_abstract",
        "full_text_available": False, "date_precision": "day" if published else "unknown",
        "source_file": source_file, "source_file_sha256": source_file_sha256,
        "influence_provenance": influence,
        "citation_count": raw.get("cited_by_count"),
        "field_weighted_citation_impact": raw.get("fwci"),
        "citation_normalized_percentile": raw.get("citation_normalized_percentile"),
        "topics": raw.get("topics") or [], "locations": raw.get("locations") or [],
        "type": raw.get("type"), "metadata_source_hash": raw.get("source_hash"),
        "license": raw.get("license"),
        "authors_raw": raw.get("authors"),
        "mirror_source": raw.get("mirror_source") or raw.get("metadata_dataset") or raw.get("source"),
        "mirror_revision": raw.get("mirror_revision") or raw.get("metadata_revision") or raw.get("dataset_revision"),
    }

def _matches(categories: Iterable[str], prefixes: Iterable[str]) -> bool:
    return any(cat == prefix or cat.startswith(prefix) for cat in categories for prefix in prefixes)

def _group(row: Mapping, groups: Mapping) -> str | None:
    if row.get("corpus_group") in groups:
        return row["corpus_group"]
    return next((name for name, prefixes in groups.items() if _matches(row["categories"], prefixes)), None)


def _rank(row: Mapping, seed: int) -> int:
    value = f"{seed}\0{row['source_id']}\0{row['content_sha256']}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big")

def _push(bucket: list, item: tuple[int, dict], cap: int) -> None:
    rank, row = item
    entry = (-rank, row["source_id"], row)
    if len(bucket) < cap:
        heapq.heappush(bucket, entry)
    elif (rank, row["source_id"]) < (-bucket[0][0], bucket[0][1]):
        heapq.heapreplace(bucket, entry)

def select_stream(path: Path, *, max_new: int, cutoff: str, seed: int,
                  groups: Mapping[str, tuple[str, ...]] | None = None) -> tuple[list[dict], dict]:
    if max_new < 1:
        raise ValueError("max_new must be positive")
    groups = dict(groups or {})
    quotas = {name: max_new // len(groups) for name in groups}
    for i, name in enumerate(groups):
        quotas[name] += int(i < max_new % max(1, len(groups)))
    reservoirs = {name: [] for name in groups}
    fallback = []
    overall = []
    total = accepted = skipped = malformed = 0
    source_hash = hashlib.sha256()
    with path.open("rb") as fh:
        for raw_bytes in fh:
            source_hash.update(raw_bytes)
            if not raw_bytes.strip():
                continue
            total += 1
            try:
                raw = json.loads(raw_bytes)
                if not isinstance(raw, Mapping):
                    raise ValueError("record is not an object")
                row = normalize_record(raw, source_file=str(path), source_file_sha256="", cutoff=cutoff)
                if row is None:
                    skipped += 1
                    continue
                accepted += 1
                rank = _rank(row, seed)
                _push(overall, (rank, row), max_new)
                group = _group(row, groups)
                bucket = reservoirs[group] if group else fallback
                cap = quotas[group] if group else max_new
                _push(bucket, (rank, row), cap)
            except (ValueError, TypeError, json.JSONDecodeError):
                malformed += 1
    source_digest = source_hash.hexdigest()
    chosen = [entry[2] for bucket in reservoirs.values() for entry in bucket]
    chosen.extend(entry[2] for entry in sorted(fallback, key=lambda x: (-x[0], x[1]))
                  [:max(0, max_new - len(chosen))])
    selected_ids = {row["source_id"] for row in chosen}
    chosen.extend(entry[2] for entry in sorted(overall, key=lambda x: (-x[0], x[1])
                  ) if entry[2]["source_id"] not in selected_ids and len(chosen) < max_new)
    chosen = sorted({row["source_id"]: row for row in chosen}.values(), key=lambda row: row["source_id"])
    for row in chosen:
        row["source_file_sha256"] = source_digest
    counts = Counter(_group(row, groups) or "unassigned" for row in chosen)
    return chosen, {"path": str(path), "sha256": source_digest, "lines": total,
                    "accepted": accepted, "skipped": skipped, "malformed": malformed,
                    "selected": len(chosen), "selected_by_group": dict(counts),
                    "quotas": quotas, "seed": seed}

def import_bulk(input_path: Path, output: Path, *, existing: Path | None, max_new: int,
                cutoff: str, seed: int, groups: Mapping[str, tuple[str, ...]] | None = None) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output}")
    selected, source_manifest = select_stream(input_path, max_new=max_new, cutoff=cutoff,
                                               seed=seed, groups=groups)
    old = []
    if existing:
        from scripts.ingest_research_corpus import _existing_corpus_rows
        old = _existing_corpus_rows(existing, cutoff)
    old_hashes = {row.get("content_sha256") for row in old}
    def doi_key(value: object) -> str:
        return re.sub(r"^(https?://(dx\.)?doi\.org/|doi:)", "", clean(str(value or "")).casefold())
    def title_key(value: object) -> str:
        return re.sub(r"[^a-z0-9]+", " ", clean(str(value or "")).casefold()).strip()
    old_by_doi = {doi_key(row.get("doi")): row for row in old if doi_key(row.get("doi"))}
    old_by_arxiv = {arxiv_canonical_id(str(row.get("arxiv_id"))): row for row in old if row.get("arxiv_id")}
    old_by_title = {title_key(row.get("title")): row for row in old
                    if len(title_key(row.get("title"))) >= 24}
    aliases_before = {}
    unmatched = []
    for row in selected:
        match = old_by_doi.get(doi_key(row.get("doi"))) if doi_key(row.get("doi")) else None
        if match is None and row.get("arxiv_id"):
            match = old_by_arxiv.get(arxiv_canonical_id(str(row["arxiv_id"])))
        if match is None:
            match = old_by_title.get(title_key(row.get("title")))
        if match is not None:
            aliases_before[row["source_id"]] = match.get("canonical_id", match["source_id"])
            # Preserve the old canonical/source row, while retaining useful
            # bulk metadata and an auditable alias for the newer record.
            merged = match.setdefault("bulk_aliases", [])
            if row["source_id"] not in merged:
                merged.append(row["source_id"])
            if not match.get("doi") and row.get("doi"):
                match["doi"] = row["doi"]
            if not match.get("arxiv_id") and row.get("arxiv_id"):
                match["arxiv_id"] = row["arxiv_id"]
            for key in ("query_slot", "corpus_group", "citation_count", "field_weighted_citation_impact",
                        "citation_normalized_percentile", "topics", "locations", "type",
                        "metadata_source_hash"):
                if row.get(key) is not None and match.get(key) is None:
                    match[key] = row[key]
            observations = match.setdefault("influence_provenance", [])
            for observation in row.get("influence_provenance", []):
                if observation not in observations:
                    observations.append(observation)
            if row.get("categories"):
                match["categories"] = sorted(set(match.get("categories", []) + row["categories"]))
        elif row.get("content_sha256") not in old_hashes:
            unmatched.append(row)
    rows, aliases = deduplicate(old + unmatched)
    old_canonical_ids = {str(row.get("canonical_id")) for row in old if row.get("canonical_id")}
    if not old_canonical_ids.issubset({str(row.get("canonical_id")) for row in rows}):
        raise RuntimeError("bulk import would remove an existing canonical ID")
    existing_openai = None
    existing_openai_manifest = None
    if existing:
        candidate = existing.parent / "openai_sources.jsonl"
        if candidate.is_file():
            existing_openai = candidate
            existing_openai_manifest = {
                "path": str(candidate),
                "sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
                "records": sum(1 for line in candidate.read_text(encoding="utf-8").splitlines() if line.strip()),
            }
    aliases.update(aliases_before)
    if existing and (existing.parent / "aliases.json").is_file():
        previous_aliases = json.loads((existing.parent / "aliases.json").read_text())
        canonical_ids = {row["canonical_id"] for row in rows}
        for alias, target in previous_aliases.items():
            if target not in canonical_ids:
                raise ValueError("Existing alias target missing from merged corpus: " + target)
            aliases[alias] = target
    output.mkdir(parents=True, exist_ok=True)
    write_corpus(rows, aliases, [], output, cutoff=cutoff)
    if existing_openai:
        shutil.copyfile(existing_openai, output / "openai_sources.jsonl")
    db = sqlite3.connect(output / "corpus.sqlite")
    db.execute("ALTER TABLE works ADD COLUMN categories_json TEXT")
    db.execute("ALTER TABLE works ADD COLUMN bulk_metadata_json TEXT")
    by_source = {row["source_id"]: row for row in rows}
    db.executemany("UPDATE works SET categories_json=?, bulk_metadata_json=? WHERE source_id=?",
                   [(json.dumps(by_source[source].get("categories", [])),
                     json.dumps({key: by_source[source].get(key) for key in
                                 ("query_slot", "corpus_group", "citation_count", "field_weighted_citation_impact",
                                  "citation_normalized_percentile", "topics", "locations",
                                  "type", "metadata_source_hash", "influence_provenance", "license",
                                  "authors_raw", "mirror_source", "mirror_revision")}, sort_keys=True),
                     source)
                    for source in by_source])
    db.commit()
    db.close()
    manifest = {"schema": "bulk-corpus.v1", "cutoff": cutoff, "records": len(rows),
                "import_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "existing_records": len(old), "bulk_selected": len(selected),
                "bulk_new_records": len(rows) - len(old),
                "source": source_manifest, "existing_snapshot": str(existing) if existing else None,
                "existing_openai_sources": existing_openai_manifest,
                "corpus_sha256": hashlib.sha256((output / "works.jsonl").read_bytes()).hexdigest()}
    (output / "bulk-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest

def parse_groups(values: list[str]) -> dict[str, tuple[str, ...]]:
    if not values:
        return DEFAULT_GROUPS
    out = {}
    for item in values:
        if "=" not in item:
            raise ValueError("group must be NAME=PREFIX[,PREFIX]")
        name, raw = item.split("=", 1)
        prefixes = tuple(clean(x) for x in raw.split(",") if clean(x))
        if not name or not prefixes or name in out:
            raise ValueError("invalid or duplicate category group")
        out[name] = prefixes
    return out

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--existing", type=Path)
    parser.add_argument("--max-new", type=int, default=100_000)
    parser.add_argument("--cutoff", default="2026-11-16")
    parser.add_argument("--seed", type=int, default=73031)
    parser.add_argument("--group", action="append", default=[])
    args = parser.parse_args()
    print(json.dumps(import_bulk(args.metadata_jsonl, args.output, existing=args.existing,
                                  max_new=args.max_new, cutoff=args.cutoff, seed=args.seed,
                                  groups=parse_groups(args.group)), indent=2))

if __name__ == "__main__":
    main()
