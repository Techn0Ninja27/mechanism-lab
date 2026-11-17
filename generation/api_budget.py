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

    def reserve(self, request: str, amount_usd: float, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
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
