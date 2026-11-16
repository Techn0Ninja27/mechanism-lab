"""Bounded, read-only feedback from completed local experiments.

This module only gates provenance and source-pair matching. A matching hash is
evidence that a saved artifact was not changed since the registry was written;
it is not evidence that the experiment is correct, useful, novel, or causal.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit

SCHEMA_VERSION = "experiment-feedback.v1"
REPORT_SCHEMA_VERSION = "experiment-feedback-report.v1"
MAX_RECORDS = 100
MAX_CONTEXT_RECORDS = 5
MAX_SOURCE_URLS = 8
MAX_RESULT_BYTES = 20_000_000
MAX_WARNINGS = 20
MAX_WARNING_CHARS = 300
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,119}$")
_ARXIV_ID = re.compile(r"^(?:[0-9]{4}\.[0-9]{4,5})(?:v[0-9]+)?$")


def _canonical_arxiv(value: object) -> str | None:
    """Canonicalize strict arXiv abs/pdf/html URLs; never arbitrary URLs."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = urlsplit(value.strip())
    except ValueError:
        return None
    try:
        hostname = parsed.hostname
        port = parsed.port
    except ValueError:
        return None
    if parsed.scheme.lower() != "https" or hostname not in {"arxiv.org", "www.arxiv.org"}:
        return None
    if parsed.username is not None or parsed.password is not None or port is not None:
        return None
    if parsed.query or parsed.fragment:
        return None
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) != 2 or parts[0] not in {"abs", "pdf", "html"}:
        return None
    identifier = parts[1]
    if parts[0] == "pdf" and identifier.endswith(".pdf"):
        identifier = identifier[:-4]
    if not _ARXIV_ID.fullmatch(identifier):
        return None
    return "arxiv:" + re.sub(r"v[0-9]+$", "", identifier)


def _safe_artifact(root: Path, relative: object, *, label: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError(f"{label} must be a relative POSIX path")
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"{label} escapes the engine root")
    if not relative.startswith("operator-studies/"):
        raise ValueError(f"{label} must be under operator-studies/")
    root = root.resolve()
    operator_root = root / "operator-studies"
    if operator_root.is_symlink():
        try:
            operator_root.resolve().relative_to(root)
        except ValueError as exc:
            raise ValueError("operator-studies/ escapes the engine root") from exc
    allowed = operator_root.resolve()
    try:
        allowed.relative_to(root)
    except ValueError as exc:
        raise ValueError("operator-studies/ escapes the engine root") from exc
    path = (root / candidate).resolve()
    try:
        path.relative_to(allowed)
    except ValueError as exc:
        raise ValueError(f"{label} escapes operator-studies/") from exc
    if not path.is_file():
        raise ValueError(f"{label} is missing")
    return path


def _record_error(record: object, root: Path, pack_urls: set[str]) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(record, Mapping):
        return None, "record is not an object"
    required = {
        "id", "source_urls", "tested_change", "verdict", "scope", "limitations",
        "report_path", "result_path", "result_sha256", "train_eligible", "novelty_established",
    }
    if set(record) != required:
        return None, f"record fields do not match required schema for {record.get('id', '<unknown>')!r}"
    rid = record["id"]
    if not isinstance(rid, str) or not _ID.fullmatch(rid):
        return None, "record has invalid id"
    urls = record["source_urls"]
    if not isinstance(urls, list) or not 1 <= len(urls) <= MAX_SOURCE_URLS:
        return None, f"record {rid!r} has invalid source_urls"
    normalized = [_canonical_arxiv(url) for url in urls]
    if any(value is None for value in normalized) or len(set(normalized)) != len(normalized):
        return None, f"record {rid!r} has a non-canonical or duplicate arXiv URL"
    if not set(normalized).issubset(pack_urls):
        return None, f"record {rid!r} source pair is absent from the supplied source pack"
    for field, limit in (("tested_change", 1000), ("verdict", 300), ("scope", 1000)):
        value = record[field]
        if not isinstance(value, str) or not value.strip() or len(value) > limit:
            return None, f"record {rid!r} has invalid {field}"
    limitations = record["limitations"]
    if (not isinstance(limitations, list) or not limitations or len(limitations) > 5 or
            any(not isinstance(value, str) or not value.strip() or len(value) > 500 for value in limitations)):
        return None, f"record {rid!r} has invalid limitations"
    if record["train_eligible"] is not False or record["novelty_established"] is not False:
        return None, f"record {rid!r} cannot be promoted to training or novelty evidence"
    digest = record["result_sha256"]
    if not isinstance(digest, str) or not _HEX64.fullmatch(digest):
        return None, f"record {rid!r} has invalid result_sha256"
    try:
        _safe_artifact(root, record["report_path"], label="report_path")
        result = _safe_artifact(root, record["result_path"], label="result_path")
    except (ValueError, OSError, RuntimeError) as exc:
        return None, f"record {rid!r}: {exc}"
    try:
        if result.stat().st_size > MAX_RESULT_BYTES:
            return None, f"record {rid!r} result exceeds size limit"
        with result.open('rb') as stream:
            payload=stream.read(MAX_RESULT_BYTES+1)
        if len(payload)>MAX_RESULT_BYTES:
            return None, f"record {rid!r} result exceeds size limit"
        actual = hashlib.sha256(payload).hexdigest()
    except OSError as exc:
        return None, f"record {rid!r} result cannot be read: {type(exc).__name__}"
    if actual != digest:
        return None, f"record {rid!r} result hash does not match"
    selected = dict(record)
    selected["source_urls"] = list(urls)
    selected["normalized_source_ids"] = sorted(normalized)
    selected["artifact_sha256_verified"] = True
    return selected, None


def inspect_feedback(pack: Mapping[str, object], root: str | Path, registry: Mapping[str, object]) -> dict[str, object]:
    """Return at most five valid, source-pair-matched feedback records."""
    warnings: list[str] = []
    def warn(message: str) -> None:
        if len(warnings) < MAX_WARNINGS:
            warnings.append(str(message)[:MAX_WARNING_CHARS])
    scope = "partial curated experiment feedback; artifact integrity does not establish scientific validity, usefulness, feasibility, or novelty"
    if not isinstance(pack, Mapping) or not isinstance(pack.get("sources"), list):
        return {"schema_version": REPORT_SCHEMA_VERSION, "scope": scope, "records": [], "warnings": ["source pack is malformed"]}
    pack_urls = set()
    for source in pack["sources"]:
        if isinstance(source, Mapping):
            canonical = _canonical_arxiv(source.get("url"))
            if canonical:
                pack_urls.add(canonical)
    if not isinstance(registry, Mapping) or registry.get("schema_version") != SCHEMA_VERSION or set(registry) != {"schema_version", "records"}:
        return {"schema_version": REPORT_SCHEMA_VERSION, "scope": scope, "records": [], "warnings": ["feedback registry schema is invalid"]}
    raw_records = registry["records"]
    if not isinstance(raw_records, list):
        return {"schema_version": REPORT_SCHEMA_VERSION, "scope": scope, "records": [], "warnings": ["feedback registry records is not a list"]}
    if len(raw_records) > MAX_RECORDS:
        warn(f"feedback registry exceeds {MAX_RECORDS} records; no records accepted")
        raw_records = []
    selected: list[dict[str, Any]] = []
    root_path = Path(root)
    raw_ids = [item.get("id") for item in raw_records if isinstance(item, Mapping) and isinstance(item.get("id"), str)]
    duplicate_ids = {rid for rid in raw_ids if raw_ids.count(rid) > 1}
    for rid in sorted(duplicate_ids):
        warn(f"duplicate feedback id {rid!r}; all copies rejected")
    for raw in raw_records:
        if isinstance(raw, Mapping) and isinstance(raw.get("id"),str) and raw.get("id") in duplicate_ids:
            continue
        record, error = _record_error(raw, root_path, pack_urls)
        if record is None:
            warn(error or "invalid feedback record")
            continue
        selected.append(record)
    selected.sort(key=lambda item: item["id"])
    if len(selected) > MAX_CONTEXT_RECORDS:
        warn(f"feedback context bounded to {MAX_CONTEXT_RECORDS} records")
        selected = selected[:MAX_CONTEXT_RECORDS]
    try:
        registry_sha = hashlib.sha256(json.dumps(registry, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    except (TypeError, ValueError):
        registry_sha = None
        warn("registry digest unavailable for non-JSON values")
    return {"schema_version": REPORT_SCHEMA_VERSION, "scope": scope, "records": selected,
            "warnings": warnings, "matched_count": len(selected),
            "registry_sha256": registry_sha}
