"""A conservative, cross-process ledger for paid model-call budgets.

Reservations are charged before a request is sent.  A reservation remains
charged while a call is pending or its receipt is unknown; only a verified
receipt can replace the reservation with actual provider usage.  The ledger
contains no credentials and is intended for a run-local JSON path.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


class BudgetError(RuntimeError):
    """Raised when a reservation or reconciliation is unsafe."""


def _amount(value: Any, name: str = "amount") -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BudgetError(f"{name} must be a finite non-negative number")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise BudgetError(f"{name} must be a finite non-negative number")
    return value


def request_id(request: Any) -> str:
    """Return a stable ID for a canonical request object."""
    raw = json.dumps(request, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def receipt_request_id(path: str | Path, request_hash: str) -> str:
    """Stable reservation ID used by the model client for one receipt path."""
    if not isinstance(request_hash, str) or len(request_hash) != 64:
        raise BudgetError("request_hash must be a sha256")
    # Match scripts.world_search.digest exactly (including its default JSON
    # separators), since this ID is already part of the persisted protocol.
    raw = json.dumps({"path": str(Path(path).resolve()), "request": request_hash},
                     sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()


def receipt_cost(receipt: dict[str, Any]) -> float:
    """Extract the verified provider cost from a complete receipt."""
    usage = receipt.get("usage") or {}
    if not isinstance(usage, dict):
        raise BudgetError("complete receipt usage is malformed")
    details = usage.get("cost_details")
    if details is not None and not isinstance(details, dict):
        raise BudgetError("complete receipt cost details are malformed")
    raw = []
    if "cost" in usage:
        raw.append(usage["cost"])
    if isinstance(details, dict) and "upstream_inference_cost" in details:
        raw.append(details["upstream_inference_cost"])
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or
           not math.isfinite(float(x)) or float(x) < 0 for x in raw):
        raise BudgetError("complete receipt contains an invalid provider cost")
    values = [float(x) for x in raw]
    if (not values or not isinstance(receipt.get("response_id"), str) or
            not receipt["response_id"].strip() or receipt.get("finish_reason") != "stop" or
            not isinstance(receipt.get("content"), str) or not receipt["content"].strip()):
        raise BudgetError("complete receipt lacks verified provider usage")
    return max(values)


def validate_and_reconcile_receipt(ledger: "BudgetLedger", path: str | Path,
                                   receipt: dict[str, Any], request_hash: str,
                                   reserve_usd: float) -> dict[str, Any]:
    """Bind a complete local receipt to its existing reservation.

    This rejects fabricated or copied receipts with no reservation, while
    allowing a legitimate crash after the receipt was written but before
    reconciliation. The ledger's own idempotent reconcile operation prevents
    double charging. It cannot detect content swapped under a genuine matching
    reservation.
    """
    if receipt.get("status") != "complete" or receipt.get("request_hash") != request_hash:
        raise BudgetError("receipt status or request hash is invalid")
    reserve = _amount(reserve_usd, "reserve_usd")
    rid = receipt_request_id(path, request_hash)
    if not ledger.path.exists():
        raise BudgetError("complete receipt has no matching budget reservation")
    snap = ledger.snapshot()
    rec = snap["reservations"].get(rid)
    if not rec:
        raise BudgetError("complete receipt has no matching budget reservation")
    if abs(float(rec.get("reserved_usd", -1)) - reserve) > 1e-12:
        raise BudgetError("reservation amount does not match request")
    metadata = rec.get("metadata") or {}
    if metadata.get("request_hash") != request_hash:
        raise BudgetError("reservation request hash mismatch")
    actual = receipt_cost(receipt)
    if rec.get("status") in {"pending", "unknown"}:
        if actual > reserve + 1e-12:
            raise BudgetError("receipt cost exceeds reserved amount")
        return ledger.reconcile(rid, actual, verified=True, receipt_id=receipt["response_id"])
    if rec.get("status") in {"reconciled", "over_budget"}:
        if abs(float(rec.get("actual_usd", -1)) - actual) > 1e-12 or rec.get("receipt_id") != receipt.get("response_id"):
            raise BudgetError("reconciled receipt does not match ledger")
        return rec
    raise BudgetError("reservation is in an invalid state")


class BudgetLedger:
    """File-locked budget ledger safe for independent worker processes."""

    def __init__(self, path: str | Path, cap_usd: float = 10.0):
        self.path = Path(path)
        self.lock_path = Path(str(self.path) + ".lock")
        self.cap_usd = _amount(cap_usd, "cap_usd")
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _locked(self) -> Iterator[dict[str, Any]]:
        self.lock_path.touch(exist_ok=True)
        with self.lock_path.open("r+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = self._read()
            if state["cap_usd"] != self.cap_usd:
                raise BudgetError("ledger cap differs from requested cap")
            try:
                yield state
            except BaseException:
                # Mutations made before a safety exception (for example a
                # verified receipt that puts the ledger over cap) are audit
                # records and must survive the exception.
                self._write(state)
                raise
            else:
                self._write(state)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"version": 1, "cap_usd": self.cap_usd, "reservations": {}}
        try:
            state = json.loads(self.path.read_text())
        except (OSError, ValueError) as exc:
            raise BudgetError(f"cannot read ledger: {exc}") from exc
        if state.get("version") != 1 or state.get("cap_usd") != self.cap_usd or not isinstance(state.get("reservations"), dict):
            raise BudgetError("invalid ledger format")
        heads = state.get("audit_heads", {})
        if not isinstance(heads, dict):
            raise BudgetError("invalid audit_heads")
        for chain_id, entry in heads.items():
            if not isinstance(chain_id, str) or not chain_id.strip() or not isinstance(entry, dict):
                raise BudgetError("invalid audit head entry")
            head = entry.get("head")
            if not isinstance(head, str) or len(head) != 64 or any(c not in "0123456789abcdef" for c in head):
                raise BudgetError("invalid audit head hash")
            if "last_event" in entry and entry["last_event"] is not None and not isinstance(entry["last_event"], dict):
                raise BudgetError("invalid audit head event")
        for rid, rec in state["reservations"].items():
            if not isinstance(rid, str) or not isinstance(rec, dict):
                raise BudgetError("invalid reservation record")
            status = rec.get("status")
            if status not in {"pending", "unknown", "reconciled", "over_budget"}:
                raise BudgetError("invalid reservation status")
            _amount(rec.get("reserved_usd"), "reserved_usd")
            if status in {"reconciled", "over_budget"}:
                _amount(rec.get("actual_usd"), "actual_usd")
        return state

    def _write(self, state: dict[str, Any]) -> None:
        fd, name = tempfile.mkstemp(prefix=self.path.name + ".", dir=str(self.path.parent))
        try:
            with os.fdopen(fd, "w") as out:
                json.dump(state, out, indent=2, sort_keys=True)
                out.write("\n")
                out.flush()
                os.fsync(out.fileno())
            os.replace(name, self.path)
        finally:
            if os.path.exists(name):
                os.unlink(name)

    @staticmethod
    def _charged(state: dict[str, Any]) -> float:
        total = 0.0
        for rec in state["reservations"].values():
            total += rec["actual_usd"] if rec["status"] in {"reconciled", "over_budget"} else rec["reserved_usd"]
        return total

    def snapshot(self) -> dict[str, Any]:
        with self._locked() as state:
            return {"cap_usd": self.cap_usd, "charged_usd": self._charged(state), "available_usd": self.cap_usd - self._charged(state), "reservations": state["reservations"]}

    def reserve(self, request: str, amount_usd: float, metadata: dict[str, Any] | None = None, *, allow_existing: bool = True) -> dict[str, Any]:
        if not isinstance(request, str) or not request.strip():
            raise BudgetError("request must be a non-empty ID")
        amount = _amount(amount_usd)
        with self._locked() as state:
            if any(rec["status"] == "over_budget" for rec in state["reservations"].values()):
                raise BudgetError("ledger is over budget; no further reservations permitted")
            old = state["reservations"].get(request)
            if old:
                if old["reserved_usd"] != amount:
                    raise BudgetError("request already exists with a different reservation")
                if not allow_existing:
                    raise BudgetError("request already has a ledger reservation; refusing duplicate paid request")
                return old
            if self._charged(state) + amount > self.cap_usd + 1e-12:
                raise BudgetError("budget cap would be exceeded")
            rec = {"reserved_usd": amount, "status": "pending", "metadata": metadata or {}}
            state["reservations"][request] = rec
            return rec

    def mark_unknown(self, request: str) -> dict[str, Any]:
        with self._locked() as state:
            rec = state["reservations"].get(request)
            if not rec:
                raise BudgetError("unknown request ID")
            if rec["status"] in {"reconciled", "over_budget"}:
                return rec
            rec["status"] = "unknown"
            return rec

    def reconcile(self, request: str, actual_usd: float, *, verified: bool, receipt_id: str | None = None) -> dict[str, Any]:
        actual = _amount(actual_usd, "actual_usd")
        if not verified:
            raise BudgetError("only verified receipts can reconcile a reservation")
        with self._locked() as state:
            rec = state["reservations"].get(request)
            if not rec:
                raise BudgetError("unknown request ID")
            if rec["status"] in {"reconciled", "over_budget"}:
                if rec["actual_usd"] != actual:
                    raise BudgetError("request already reconciled with a different amount")
                return rec
            charged_without = self._charged(state) - rec["reserved_usd"]
            over_budget = charged_without + actual > self.cap_usd + 1e-12
            rec.update(status="over_budget" if over_budget else "reconciled", actual_usd=actual)
            if receipt_id is not None:
                if not isinstance(receipt_id, str) or not receipt_id.strip():
                    raise BudgetError("receipt_id must be a non-empty string")
                rec["receipt_id"] = receipt_id
            if over_budget:
                raise BudgetError("verified receipt exceeded budget cap; true usage retained and ledger locked")
            return rec

    def anchor_event(self, chain_id: str, expected_head: str, event: dict[str, Any]) -> dict[str, Any]:
        """CAS-anchor an external journal event without changing reservations."""
        if not isinstance(chain_id, str) or not chain_id.strip():
            raise BudgetError("chain_id must be a non-empty string")
        if not isinstance(expected_head, str) or len(expected_head) != 64 or any(c not in "0123456789abcdef" for c in expected_head):
            raise BudgetError("expected_head must be a lowercase sha256")
        if not isinstance(event, dict):
            raise BudgetError("event must be an object")
        event_hash = event.get("event_hash")
        previous_hash = event.get("previous_hash")
        if not isinstance(event_hash, str) or len(event_hash) != 64 or any(c not in "0123456789abcdef" for c in event_hash):
            raise BudgetError("event_hash must be a lowercase sha256")
        if previous_hash != expected_head:
            raise BudgetError("event previous_hash does not match expected_head")
        with self._locked() as state:
            heads = state.setdefault("audit_heads", {})
            current = heads.get(chain_id)
            current_head = current.get("head") if current else "0" * 64
            if current_head != expected_head:
                if current and current_head == event_hash and current.get("last_event") == event:
                    return dict(current)
                raise BudgetError("audit head compare-and-swap failed")
            if current and current_head == event_hash:
                if current.get("last_event") == event:
                    return dict(current)
                raise BudgetError("audit head hash already anchors a different event")
            entry = {"head": event_hash, "last_event": dict(event)}
            heads[chain_id] = entry
            return dict(entry)

    def read_event_anchor(self, chain_id: str) -> dict[str, Any]:
        """Read an anchored external journal head under the ledger lock."""
        if not isinstance(chain_id, str) or not chain_id.strip():
            raise BudgetError("chain_id must be a non-empty string")
        with self._locked() as state:
            entry = state.get("audit_heads", {}).get(chain_id)
            if entry is None:
                return {"head": "0" * 64, "last_event": None}
            return dict(entry)


def migrate_cap(path: str | Path, *, old_cap_usd: float, new_cap_usd: float,
                backup_path: str | Path, audit_path: str | Path,
                authorization: str) -> dict[str, Any]:
    """Migrate an authorized cap under lock, atomically replacing the ledger.

    The migration is intentionally separate from normal reservations. It refuses
    pending/unknown calls and preserves the pre-migration bytes before writing
    the new cap. Existing reservation amounts and receipts are never changed.
    """
    ledger_path = Path(path)
    old_cap = _amount(old_cap_usd, "old_cap_usd")
    new_cap = _amount(new_cap_usd, "new_cap_usd")
    if not isinstance(authorization, str) or not authorization.strip():
        raise BudgetError("authorization must be a non-empty string")
    if new_cap <= old_cap:
        raise BudgetError("new cap must be greater than old cap")
    backup = Path(backup_path); audit = Path(audit_path)
    lock_path = Path(str(ledger_path) + ".lock")
    if len({ledger_path.resolve(), lock_path.resolve(), backup.resolve(), audit.resolve()}) != 4:
        raise BudgetError("ledger, lock, backup and audit paths must be distinct")
    lock_path.touch(exist_ok=True)
    import datetime
    with lock_path.open("r+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if backup.exists() or audit.exists():
            raise BudgetError("backup or audit already exists; migration is immutable")
        state = BudgetLedger(ledger_path, old_cap)._read()
        statuses = {r.get("status") for r in state.get("reservations", {}).values()}
        if statuses & {"pending", "unknown"}:
            raise BudgetError("cannot migrate while calls are pending or unknown")
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ledger_path, backup)
        before_hash = hashlib.sha256(ledger_path.read_bytes()).hexdigest()
        state["cap_usd"] = new_cap
        BudgetLedger(ledger_path, new_cap)._write(state)
        after_hash = hashlib.sha256(ledger_path.read_bytes()).hexdigest()
        record = {
            "schema": "budget-cap-migration.v1",
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "ledger": str(ledger_path), "backup": str(backup),
            "old_cap_usd": old_cap, "new_cap_usd": new_cap,
            "before_sha256": before_hash, "after_sha256": after_hash,
            "reservation_count": len(state["reservations"]),
            "statuses": sorted(statuses),
            "charged_usd": BudgetLedger(ledger_path, new_cap)._charged(state),
            "authorization": authorization.strip(),
            "receipts_preserved": True,
        }
        audit.parent.mkdir(parents=True, exist_ok=True)
        audit.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        fcntl.flock(lock, fcntl.LOCK_UN)
    return record
