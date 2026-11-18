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
import base64
from pathlib import Path
from typing import Any, Mapping

EVENTS = {
    "draw_started", "draw_completed", "draw_failed", "review_decided",
    "representation_locked", "pair_locked", "event_intent", "reference_packets_exported",
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


def _append_row(out: Path, row: dict[str, Any]) -> dict[str, Any]:
    """Append a pre-hashed row after checking it is exactly the next row."""
    path = _journal_path(Path(out))
    rows = audit(out, allow_active=True)["events"]
    expected = len(rows)
    if row.get("seq") != expected:
        raise ValueError("row is not the next journal sequence")
    previous = rows[-1]["event_hash"] if rows else "0" * 64
    if row.get("previous_hash") != previous or row.get("event_hash") != _event_hash(expected, row["event"], row["payload"], previous):
        raise ValueError("anchored row does not extend the journal")
    encoded = _canon(row) + b"\n"
    with path.open("ab") as handle:
        handle.write(encoded); handle.flush(); os.fsync(handle.fileno())
    return row


def _anchor_put(anchor: Any, chain_id: str, expected_head: str, event: Mapping[str, Any]) -> None:
    if anchor is None:
        return
    anchor.anchor_event(chain_id, expected_head, dict(event))


def commit_event(out: Path, event: str, payload: Mapping[str, Any],
                 artifacts: Mapping[str, bytes] | None = None, *,
                 anchor: Any = None, chain_id: str = "campaign") -> dict[str, Any]:
    """Commit an artifact-bearing event using intent-before-write ordering.

    The intent contains exact bytes and hashes, so recovery never adopts an
    unrelated file.  If an external anchor is supplied it is written before
    the local intent append.
    """
    if event not in EVENTS - {"event_intent"}:
        raise ValueError("invalid target event")
    artifacts = dict(artifacts or {})
    encoded = {rel: {"data": base64.b64encode(data).decode("ascii"),
                     "sha256": sha256_bytes(data)} for rel, data in artifacts.items()}
    rows = audit(out, allow_active=True)["events"]
    previous = rows[-1]["event_hash"] if rows else "0" * 64
    intent_id = hashlib.sha256(_canon({"event": event, "payload": payload, "artifacts": encoded, "previous": previous})).hexdigest()
    intent_payload = {"intent_id": intent_id, "target_event": event,
                      "target_payload": dict(payload), "artifacts": encoded}
    intent = {"seq": len(rows), "event": "event_intent", "payload": intent_payload,
              "previous_hash": previous}
    intent["event_hash"] = _event_hash(intent["seq"], intent["event"], intent_payload, previous)
    _anchor_put(anchor, chain_id, previous, intent)
    _append_row(out, intent)
    return recover_pending(out, anchor=anchor, chain_id=chain_id, intent_id=intent_id)


def recover_pending(out: Path, *, anchor: Any = None, chain_id: str = "campaign",
                    intent_id: str | None = None) -> dict[str, Any]:
    """Complete only hash-bound intents whose declared artifacts are present."""
    rows = audit(out, allow_active=True)["events"]
    completed = {r.get("payload", {}).get("intent_id") for r in rows
                 if r.get("event") != "event_intent"}
    for intent in rows:
        if intent.get("event") != "event_intent":
            continue
        p = intent["payload"]
        if intent_id is not None and p.get("intent_id") != intent_id:
            continue
        if p.get("intent_id") in completed:
            continue
        artifacts = p.get("artifacts", {})
        for rel, spec in artifacts.items():
            data = base64.b64decode(spec["data"], validate=True)
            if sha256_bytes(data) != spec["sha256"]:
                raise ValueError("intent artifact payload hash is invalid")
            path = Path(out) / rel
            if path.exists() and sha256_file(path) != spec["sha256"]:
                raise ValueError("existing artifact conflicts with intent")
            if not path.exists():
                atomic_write(out, rel, data)
        target = {"seq": len(rows), "event": p["target_event"],
                  "payload": {**p["target_payload"], "intent_id": p["intent_id"],
                              "artifacts": {r: s["sha256"] for r, s in artifacts.items()}},
                  "previous_hash": rows[-1]["event_hash"]}
        target["event_hash"] = _event_hash(target["seq"], target["event"], target["payload"], target["previous_hash"])
        _anchor_put(anchor, chain_id, target["previous_hash"], target)
        _append_row(out, target)
        rows.append(target)
        return target
    return rows[-1] if rows else {}


def recover_anchored_tail(out: Path, anchor: Any, *, chain_id: str = "campaign") -> dict[str, Any]:
    """Restore only the exact externally anchored last event, including a partial write."""
    out=Path(out); path=_journal_path(out); envelope=anchor.read_event_anchor(chain_id)
    row=envelope.get('last_event'); head=envelope.get('head')
    raw=path.read_bytes() if path.exists() else b''
    if raw and not raw.endswith(b'\n'):
        prefix,sep,partial=raw.rpartition(b'\n')
        complete=prefix+sep
        rows=[json.loads(line) for line in complete.splitlines()]
        prev=rows[-1]['event_hash'] if rows else '0'*64
        encoded=_canon(row)+b'\n' if row is not None else b''
        if row is None or row.get('seq')!=len(rows) or row.get('previous_hash')!=prev or not encoded.startswith(partial):
            raise ValueError('partial journal does not match externally anchored event')
        if row.get('event_hash')!=head or head!=_event_hash(row['seq'],row['event'],row['payload'],prev):
            raise ValueError('external event hash invalid')
        fd,name=tempfile.mkstemp(prefix='.recover-',dir=str(out))
        with os.fdopen(fd,'wb') as f:f.write(complete+encoded);f.flush();os.fsync(f.fileno())
        os.replace(name,path)
    rows=audit(out,allow_active=True)['events']
    local=rows[-1]['event_hash'] if rows else '0'*64
    if local==head:return rows[-1] if rows else {}
    if row is None or row.get('event_hash')!=head or row.get('previous_hash')!=local or row.get('seq')!=len(rows):
        raise ValueError('journal history differs from external phase-ledger anchor')
    return _append_row(out,dict(row))


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
        if row["event"] != "event_intent":
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
