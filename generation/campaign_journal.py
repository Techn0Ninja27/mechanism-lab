"""Crash-consistent, append-only provenance journal for campaign preparation.

The journal is designed to catch ordinary interruption, accidental deletion and
post-run edits.  It is not tamper proof against an actor who can rewrite every
file in the campaign directory.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

EVENTS = {
    "draw_started", "draw_completed", "draw_failed", "review_decided",
    "representation_locked", "pair_locked",
}
JOURNAL_NAME = "campaign-events.jsonl"


def _canon(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _journal_path(out: Path) -> Path:
    return Path(out) / JOURNAL_NAME


def _event_hash(seq: int, event: str, payload: Mapping[str, Any], previous: str) -> str:
    return sha256_bytes(_canon({"seq": seq, "event": event,
                                "payload": payload, "previous_hash": previous}))


def _read_lines(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    raw = path.read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError("campaign journal has a partial final line")
    rows = []
    for line in raw.splitlines():
        try:
            row = json.loads(line)
        except Exception as exc:
            raise ValueError("campaign journal contains invalid JSON") from exc
        if not isinstance(row, dict):
            raise ValueError("campaign journal event is not an object")
        rows.append(row)
    return rows


def _validate_artifacts(out: Path, payload: Mapping[str, Any]) -> None:
    artifacts = payload.get("artifacts", {})
    if artifacts is None:
        return
    if not isinstance(artifacts, Mapping):
        raise ValueError("event artifacts must be an object")
    root = Path(out).resolve()
    for rel, expected in artifacts.items():
        if not isinstance(rel, str) or not isinstance(expected, str) or len(expected) != 64:
            raise ValueError("artifact binding is malformed")
        path = (root / rel).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError("artifact path escapes campaign") from exc
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"artifact binding missing or changed: {rel}")


def record(out: Path, event: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Append one fsynced event.  Caller must already hold the campaign lock."""
    if event not in EVENTS:
        raise ValueError(f"unknown campaign event: {event}")
    if not isinstance(payload, Mapping):
        raise ValueError("event payload must be an object")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    path = _journal_path(out)
    rows = audit(out, allow_active=True)["events"]
    seq = len(rows)
    previous = rows[-1]["event_hash"] if rows else "0" * 64
    row = {"seq": seq, "event": event, "payload": dict(payload),
           "previous_hash": previous}
    row["event_hash"] = _event_hash(seq, event, row["payload"], previous)
    encoded = _canon(row) + b"\n"
    with path.open("ab") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    return row


def atomic_write(out: Path, relative: str, data: bytes) -> str:
    """Create an immutable artifact atomically, or accept identical existing bytes."""
    root = Path(out).resolve()
    target = (root / relative).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError("artifact path escapes campaign") from exc
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = sha256_bytes(data)
    if target.exists():
        if target.is_file() and target.read_bytes() == data:
            return digest
        raise FileExistsError(f"immutable artifact differs: {relative}")
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=str(target.parent))
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(tmp, target)
        except FileExistsError:
            if target.is_file() and target.read_bytes() == data:
                return digest
            raise
        finally:
            tmp.unlink(missing_ok=True)
        dfd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return digest


def audit(out: Path, *, allow_active: bool = False) -> dict[str, Any]:
    """Validate the chain and artifact bindings, deriving the campaign stop state."""
    out = Path(out)
    rows = _read_lines(_journal_path(out))
    previous = "0" * 64
    starts: dict[str, dict[str, Any]] = {}
    terminal: set[str] = set()
    stopped = False
    for expected_seq, row in enumerate(rows):
        if row.get("seq") != expected_seq or row.get("event") not in EVENTS:
            raise ValueError("campaign journal sequence or event is invalid")
        payload = row.get("payload")
        if not isinstance(payload, dict) or row.get("previous_hash") != previous:
            raise ValueError("campaign journal chain is broken")
        if row.get("event_hash") != _event_hash(expected_seq, row["event"], payload, previous):
            raise ValueError("campaign journal event hash is invalid")
        _validate_artifacts(out, payload)
        previous = row["event_hash"]
        if row["event"] == "draw_started":
            op = payload.get("operation_id")
            if not isinstance(op, str) or op in starts or op in terminal:
                raise ValueError("draw operation identity is invalid")
            starts[op] = payload
        elif row["event"] in {"draw_completed", "draw_failed"}:
            op = payload.get("operation_id")
            if not isinstance(op, str) or op not in starts or op in terminal:
                raise ValueError("draw terminal identity is invalid")
            terminal.add(op)
            if row["event"] == "draw_failed":
                stopped = True
    active = sorted(set(starts) - terminal)
    if active and not allow_active:
        raise ValueError("campaign journal has an unfinished draw")
    return {"events": rows, "stopped": stopped, "active_operations": active,
            "terminal_operations": sorted(terminal), "last_hash": previous}


def is_stopped(out: Path) -> bool:
    result = audit(out, allow_active=True)
    return bool(result["stopped"] or result["active_operations"])
