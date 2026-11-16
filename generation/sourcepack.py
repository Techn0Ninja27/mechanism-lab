"""Build bounded, provenance-preserving packs from local source files.

The packer is intentionally offline.  It reads only explicitly listed UTF-8
files beneath a supplied root, records byte and content hashes, and emits
numbered line chunks suitable for evidence-grounded prompts.  It does not
execute imported code, fetch URLs, or infer novelty.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "sourcepack.v1"
ALLOWED_EXTENSIONS = {".md", ".txt", ".py", ".json"}
EVIDENCE_LEVELS = {"full_text", "official_code", "abstract", "user_document"}


class SourcePackError(ValueError):
    pass


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _digest(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return _sha(raw)


def validate_source_pack(pack: Mapping[str, Any]) -> dict[str, Any]:
    """Validate snapshot integrity before a pack enters a generation prompt.

    Without the original root files this cannot prove source authenticity,
    completeness, semantic entailment, licensing, or novelty.
    """
    if not isinstance(pack, Mapping) or pack.get("schema_version") != SCHEMA_VERSION:
        raise SourcePackError("unsupported or missing source-pack schema_version")
    if not isinstance(pack.get("sources"), list) or not isinstance(pack.get("chunks"), list):
        raise SourcePackError("pack must contain sources and chunks lists")
    limits = pack.get("limits")
    if not isinstance(limits, Mapping):
        raise SourcePackError("pack limits are required")
    max_total, chunk_limit, included_limit = (limits.get(k) for k in ("max_total_chars", "chunk_chars", "included_chars"))
    if not isinstance(max_total, int) or max_total < 1 or not isinstance(chunk_limit, int) or chunk_limit < 1:
        raise SourcePackError("invalid pack limits")
    if not isinstance(included_limit, int) or included_limit < 0 or included_limit > max_total:
        raise SourcePackError("invalid included_chars limit")
    source_by_id: dict[str, Mapping[str, Any]] = {}
    for source in pack["sources"]:
        if not isinstance(source, Mapping):
            raise SourcePackError("source entries must be objects")
        source_id = source.get("id")
        if not isinstance(source_id, str) or not source_id.strip() or source_id in source_by_id:
            raise SourcePackError("source IDs must be unique non-empty strings")
        for field in ("title", "url", "path", "evidence_level"):
            if not isinstance(source.get(field), str) or not source[field].strip():
                raise SourcePackError(f"source {source_id!r} missing {field}")
        if source["evidence_level"] not in EVIDENCE_LEVELS:
            raise SourcePackError(f"source {source_id!r} has invalid evidence_level")
        digest = source.get("content_sha256")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise SourcePackError(f"source {source_id!r} has invalid content hash")
        for field in ("content_chars", "included_chars", "chunk_count", "total_chunk_count", "omitted_chunks"):
            if not isinstance(source.get(field), int) or source[field] < 0:
                raise SourcePackError(f"source {source_id!r} has invalid {field}")
        if source["included_chars"] > source["content_chars"] or source["chunk_count"] > source["total_chunk_count"]:
            raise SourcePackError(f"source {source_id!r} has inconsistent counts")
        if source["omitted_chunks"] != source["total_chunk_count"] - source["chunk_count"]:
            raise SourcePackError(f"source {source_id!r} has inconsistent omission count")
        source_by_id[source_id] = source
    seen: set[str] = set()
    chars: dict[str, int] = {sid: 0 for sid in source_by_id}
    counts: dict[str, int] = {sid: 0 for sid in source_by_id}
    for chunk in pack["chunks"]:
        if not isinstance(chunk, Mapping):
            raise SourcePackError("chunk entries must be objects")
        cid, sid, text = chunk.get("id"), chunk.get("source_id"), chunk.get("text")
        if not isinstance(cid, str) or not cid.strip() or cid in seen:
            raise SourcePackError("chunk IDs must be unique non-empty strings")
        if sid not in source_by_id or not isinstance(text, str) or not text:
            raise SourcePackError("chunk has unknown source or empty text")
        start, end = chunk.get("start_line"), chunk.get("end_line")
        if not isinstance(start, int) or not isinstance(end, int) or start < 1 or end < start:
            raise SourcePackError(f"invalid line span for chunk {cid!r}")
        if chunk.get("source_sha256") != source_by_id[sid]["content_sha256"]:
            raise SourcePackError(f"chunk source hash mismatch: {cid!r}")
        seen.add(cid); chars[sid] += len(text); counts[sid] += 1
    if sum(chars.values()) != included_limit or sum(chars.values()) > max_total:
        raise SourcePackError("chunk text is inconsistent with pack limits")
    for sid, source in source_by_id.items():
        if chars[sid] != source["included_chars"] or counts[sid] != source["chunk_count"]:
            raise SourcePackError(f"source {sid!r} chunk counts do not match")
    body = {key: value for key, value in pack.items() if key != "digest"}
    if pack.get("digest") != _digest(body):
        raise SourcePackError("source-pack digest mismatch")
    return dict(pack)


def _rooted_path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative.strip():
        raise SourcePackError("source path must be a non-empty relative string")
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise SourcePackError(f"source path is not safely relative: {relative!r}")
    root = root.resolve()
    path = (root / candidate).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise SourcePackError(f"source escapes root: {relative!r}") from exc
    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise SourcePackError(f"unsupported source extension: {path.suffix!r}")
    if not path.is_file():
        raise SourcePackError(f"source is not a regular file: {relative!r}")
    return path


def _chunks(source_id: str, text: str, source_sha256: str, chunk_chars: int) -> list[dict[str, Any]]:
    if chunk_chars < 1:
        raise SourcePackError("chunk_chars must be positive")
    lines = text.splitlines()
    result: list[dict[str, Any]] = []
    current: list[str] = []
    current_chars = 0
    start_line = 1
    for line_number, line in enumerate(lines, 1):
        addition = len(line) + (1 if current else 0)
        if current and current_chars + addition > chunk_chars:
            body = "\n".join(current)
            result.append({"id": f"{source_id}:chunk:{len(result)+1:04d}", "source_id": source_id,
                           "text": body, "start_line": start_line, "end_line": line_number - 1,
                           "source_sha256": source_sha256})
            current = []
            current_chars = 0
            start_line = line_number
        # Keep a single overlong line intact; splitting it would make line
        # spans ambiguous and exact quote checks less useful.
        current.append(line)
        current_chars += len(line) + (1 if len(current) > 1 else 0)
    if current:
        result.append({"id": f"{source_id}:chunk:{len(result)+1:04d}", "source_id": source_id,
                       "text": "\n".join(current), "start_line": start_line,
                       "end_line": len(lines), "source_sha256": source_sha256})
    return result


def build_source_pack(manifest: Mapping[str, Any], root: Path, *, max_total_chars: int = 100_000,
                      chunk_chars: int = 4_000, max_source_bytes: int = 5_000_000) -> dict[str, Any]:
    """Build a deterministic pack from a manifest and local root directory."""
    if not isinstance(manifest, Mapping) or not isinstance(manifest.get("sources"), list):
        raise SourcePackError("manifest must contain a sources list")
    if max_total_chars < 1:
        raise SourcePackError("max_total_chars must be positive")
    if max_source_bytes < 1:
        raise SourcePackError("max_source_bytes must be positive")
    root = Path(root).resolve()
    sources: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []
    seen: set[str] = set()
    remaining = max_total_chars
    for item in manifest["sources"]:
        if not isinstance(item, Mapping):
            raise SourcePackError("each source must be an object")
        source_id = item.get("id")
        title = item.get("title")
        url = item.get("url")
        level = item.get("evidence_level")
        if not isinstance(source_id, str) or not source_id.strip() or source_id in seen:
            raise SourcePackError(f"invalid or duplicate source id: {source_id!r}")
        if not isinstance(title, str) or not title.strip() or not isinstance(url, str) or not url.strip():
            raise SourcePackError(f"source {source_id!r} needs title and url")
        if level not in EVIDENCE_LEVELS:
            raise SourcePackError(f"source {source_id!r} has invalid evidence_level")
        path_text = item.get("path")
        path = _rooted_path(root, path_text)
        if path.stat().st_size > max_source_bytes:
            raise SourcePackError(f"source exceeds max_source_bytes: {path_text!r}")
        data = path.read_bytes()
        if len(data) > max_source_bytes:
            raise SourcePackError(f"source exceeds max_source_bytes: {path_text!r}")
        if not data:
            raise SourcePackError(f"source is empty: {path_text!r}")
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise SourcePackError(f"source is not UTF-8: {path_text!r}") from exc
        source_sha = _sha(data)
        all_chunks = _chunks(source_id, text, source_sha, chunk_chars)
        selected: list[dict[str, Any]] = []
        used = 0
        for chunk in all_chunks:
            cost = len(chunk["text"])
            if cost <= remaining:
                selected.append(chunk)
                used += cost
                remaining -= cost
            else:
                break
        omitted = len(all_chunks) - len(selected)
        record = {"id": source_id, "title": title.strip(), "url": url.strip(),
                  "path": str(Path(path_text)), "evidence_level": level,
                  "content_sha256": source_sha, "content_chars": len(text),
                  "included_chars": used, "chunk_count": len(selected),
                  "total_chunk_count": len(all_chunks), "truncated": bool(omitted),
                  "omitted_chunks": omitted}
        if item.get("license") is not None:
            if not isinstance(item["license"], str):
                raise SourcePackError(f"source {source_id!r} license must be a string")
            record["license"] = item["license"]
        sources.append(record)
        chunks.extend(selected)
        seen.add(source_id)
    body = {"schema_version": SCHEMA_VERSION, "sources": sources, "chunks": chunks,
            "limits": {"max_total_chars": max_total_chars, "chunk_chars": chunk_chars,
                        "included_chars": max_total_chars - remaining,
                        "truncated_source_count": sum(1 for s in sources if s["truncated"])}}
    body["digest"] = _digest(body)
    return body


def build_from_file(manifest_path: Path, root: Path, **kwargs: Any) -> dict[str, Any]:
    try:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourcePackError(f"cannot read manifest: {manifest_path}") from exc
    return build_source_pack(manifest, root, **kwargs)
