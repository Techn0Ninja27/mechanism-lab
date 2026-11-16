"""Offline checks for curated implementation-version lineage.

This module deliberately does not search the network or make novelty claims.  A
lineage file is a small, reviewed index of relationships that a source pack can
be checked against before a method comparison is started.
"""

from __future__ import annotations

from collections.abc import Mapping
from urllib.parse import urlsplit, urlunsplit
import re


SCHEMA_VERSION = "method-lineage.v1"


def _normal_url(value: str) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parts = urlsplit(value.strip())
    except ValueError:
        return None
    if parts.scheme.lower() not in {"http", "https"} or not parts.hostname:
        return None
    # urlsplit.hostname removes userinfo and normalises host case.  Reject
    # credentials: a repository identifier must not silently include secrets.
    if parts.username is not None or parts.password is not None:
        return None
    try:
        port = parts.port
    except ValueError:
        return None
    host = parts.hostname.lower().rstrip(".")
    if port is not None and not ((parts.scheme.lower() == "http" and port == 80) or
                                 (parts.scheme.lower() == "https" and port == 443)):
        host = f"{host}:{port}"
    path = "/" + "/".join(p for p in parts.path.split("/") if p)
    if path.endswith("/.git"):
        path = path[:-5]
    return urlunsplit((parts.scheme.lower(), host, path.rstrip("/"), "", ""))


def _repo_key(value: str) -> str | None:
    """Return a repository identity, including for pinned blob/commit URLs.

    Matching is host- and path-segment based; a URL merely containing a repo
    string (for example an arbitrary hostile host) cannot match.
    """
    normal = _normal_url(value)
    if normal is None:
        return None
    parts = urlsplit(normal)
    segments = [p for p in parts.path.split("/") if p]
    if parts.hostname in {"github.com", "gitlab.com"} and len(segments) >= 2:
        return urlunsplit((parts.scheme, parts.netloc, "/" + "/".join(segments[:2]), "", ""))
    return normal


def _source_urls(source: Mapping[str, object]) -> list[str]:
    values = []
    for key in ("url", "official_code", "repository_url", "repo_url"):
        value = source.get(key)
        if isinstance(value, str) and value.strip():
            values.append(value)
    return values


def _source_commit(source: Mapping[str, object], value: str) -> str | None:
    commit = source.get("commit")
    if isinstance(commit, str) and commit.strip():
        return commit.strip()
    match = re.search(r"/(?:blob|tree|commit)/([0-9a-fA-F]{40})(?:/|$)", value)
    return match.group(1) if match else None


def _implementation_present(source: Mapping[str, object], definition: Mapping[str, object], value: str) -> bool:
    if source.get("evidence_level") != "official_code":
        return False
    paths = definition.get("implementation_paths", [])
    if not isinstance(paths, list) or not paths:
        return True
    candidate = value.replace("\\", "/")
    local = str(source.get("path", "")).replace("\\", "/")
    return any(str(path).strip("/") in candidate or local.endswith(str(path).strip("/"))
               for path in paths if isinstance(path, str))


def _present_sources(pack: Mapping[str, object]) -> dict[str, list[tuple[str | None, str, Mapping[str, object]]]]:
    present: dict[str, list[tuple[str | None, str, Mapping[str, object]]]] = {}
    for source in pack.get("sources", []) if isinstance(pack, Mapping) else []:
        if not isinstance(source, Mapping):
            continue
        for value in _source_urls(source):
            key = _repo_key(value)
            if key:
                commit = _source_commit(source, value)
                present.setdefault(key, []).append((commit, value, source))
    return present


def _validate_lineage(lineage: Mapping[str, object]) -> None:
    if lineage.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported or missing method-lineage schema_version")
    if not isinstance(lineage.get("sources"), list) or not isinstance(lineage.get("relations"), list):
        raise ValueError("lineage must contain sources and relations lists")


def inspect_pack(pack: Mapping[str, object], lineage: Mapping[str, object]) -> dict[str, object]:
    """Check a source pack against reviewed, curated method relationships.

    Results describe evidence coverage only.  In particular, an absent known
    successor is not evidence that an implementation is latest or novel.
    """
    _validate_lineage(lineage)
    present = _present_sources(pack)
    definitions = {s.get("id"): s for s in lineage["sources"]
                   if isinstance(s, Mapping) and isinstance(s.get("id"), str)}
    warnings: list[str] = []
    relations: dict[str, object] = {}
    for relation in lineage["relations"]:
        if not isinstance(relation, Mapping):
            continue
        rid = relation.get("id")
        if not isinstance(rid, str):
            continue
        older = definitions.get(relation.get("older_source"), {})
        newer = definitions.get(relation.get("newer_source"), {})
        older_key = _repo_key(next(iter(_source_urls(older)), "")) if isinstance(older, Mapping) else None
        newer_key = _repo_key(next(iter(_source_urls(newer)), "")) if isinstance(newer, Mapping) else None
        def is_present(definition: object, key: str | None, implementation: bool = False) -> bool:
            if not key or key not in present or not isinstance(definition, Mapping):
                return False
            wanted = definition.get("commit")
            return any((wanted == commit) and
                       (not implementation or _implementation_present(source, definition, value))
                       for commit, value, source in present[key])
        older_metadata = is_present(older, older_key)
        newer_metadata = is_present(newer, newer_key)
        older_present = is_present(older, older_key, True)
        newer_present = is_present(newer, newer_key, True)
        missing = [name for name, ok in (("older_baseline", older_present), ("newer_related_version", newer_present)) if not ok]
        successors = list(_source_urls(newer)) if isinstance(newer, Mapping) else []
        status = "covered" if older_present and newer_present else ("older_only" if older_present else "absent")
        relations[rid] = {"curated_relation": True,
                          "older_present": older_present, "newer_present": newer_present,
                          "older_metadata_present": older_metadata, "newer_metadata_present": newer_metadata,
                          "mechanism": relation.get("mechanism"),
                          "paper_reported_change": relation.get("paper_reported_change"),
                          "claims": relation.get("claims", []),
                          "supports": relation.get("supports", []),
                          "review_status": relation.get("review_status", {}),
                          "known_successor_urls": successors if older_present else [],
                          "missing_baseline_evidence": missing, "status": status}
        if older_present:
            if newer_present:
                warnings.append(f"known related version covered for {rid}: {successors[0] if successors else 'source recorded'}")
            else:
                warnings.append(f"known successor absent for {rid}; do not infer latest or novelty")
    return {"schema_version": "method-lineage-report.v1",
            "scope": "curated2families not exhaustive",
            "warnings": warnings,
            "coverage": {"relations": relations, "matched_repository_urls": sorted(present)}}

