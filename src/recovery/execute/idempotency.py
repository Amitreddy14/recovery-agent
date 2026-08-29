"""Idempotency.

The failure this prevents: an executor retries a call it believes failed,
when in fact the call succeeded and the response was lost. Without a key, the
customer is charged twice. This is the single worst defect available in a
payments system, and it is caused by ordinary network behaviour rather than
by anything exotic.

Two mechanisms, because either alone is insufficient:

* **A deterministic key** derived from the logical action, sent to Razorpay so
  the provider can collapse duplicates on its side. Deterministic, not random:
  a fresh UUID per attempt would make every retry look like a new charge,
  which is exactly the bug.
* **A local ledger** recording every key we have issued and what came back.
  The provider's guarantee only helps once the request reaches it; the ledger
  catches duplicates raised before the call and lets a crashed run resume
  without re-issuing work.

The key deliberately excludes the wall clock. Two attempts on the same case
with the same action and the same attempt number *are* the same logical
operation, however far apart they occur.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from recovery.domain.enums import ActionType

KEY_VERSION = "v1"


def idempotency_key(*, case_id: str, action: ActionType, attempt: int, amount_paise: int) -> str:
    """Stable key for one logical money operation.

    `amount_paise` is included so that a changed amount is treated as a
    genuinely different operation rather than silently deduplicated against
    the old one — a re-quoted amount must not be swallowed by a stale key.
    """
    payload = f"{KEY_VERSION}|{case_id}|{action.value}|{attempt}|{amount_paise}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]
    return f"rec_{digest}"


@dataclass(frozen=True)
class LedgerEntry:
    idempotency_key: str
    case_id: str
    action: ActionType
    attempt: int
    amount_paise: int
    status: str
    provider_reference: str | None
    response_digest: str | None
    created_at: datetime

    @property
    def is_terminal(self) -> bool:
        return self.status in {"succeeded", "failed_permanent"}


SCHEMA = """
CREATE TABLE IF NOT EXISTS idempotency (
    idempotency_key    TEXT PRIMARY KEY,
    case_id            TEXT NOT NULL,
    action             TEXT NOT NULL,
    attempt            INTEGER NOT NULL,
    amount_paise       INTEGER NOT NULL,
    status             TEXT NOT NULL,
    provider_reference TEXT,
    response_digest    TEXT,
    created_at         TEXT NOT NULL,
    updated_at         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_case ON idempotency(case_id);
"""


class IdempotencyLedger:
    """Append-mostly record of every money operation attempted.

    SQLite rather than an in-memory dict: the point is to survive the crash
    that happens between issuing a request and recording its result.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            yield conn
        finally:
            conn.close()

    def get(self, key: str) -> LedgerEntry | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM idempotency WHERE idempotency_key = ?", (key,)
            ).fetchone()
        return _to_entry(row) if row else None

    def reserve(
        self,
        *,
        key: str,
        case_id: str,
        action: ActionType,
        attempt: int,
        amount_paise: int,
    ) -> LedgerEntry | None:
        """Claim a key before calling the provider.

        Returns the existing entry if this operation has been seen before, in
        which case the caller must not issue the request. Returns None on a
        fresh claim.

        The row is written *before* the network call, not after. A crash
        mid-flight then leaves an `in_flight` row, which is recoverable; the
        reverse ordering leaves no trace at all and the operation repeats.
        """
        now = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT * FROM idempotency WHERE idempotency_key = ?", (key,)
            ).fetchone()
            if existing is not None:
                return _to_entry(existing)
            conn.execute(
                "INSERT INTO idempotency (idempotency_key, case_id, action, attempt,"
                " amount_paise, status, provider_reference, response_digest,"
                " created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?, 'in_flight', NULL, NULL, ?, ?)",
                (key, case_id, action.value, attempt, amount_paise, now, now),
            )
        return None

    def complete(
        self,
        *,
        key: str,
        status: str,
        provider_reference: str | None = None,
        response: object | None = None,
    ) -> None:
        digest = None
        if response is not None:
            digest = hashlib.sha256(
                json.dumps(response, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest()[:16]
        with self._connect() as conn:
            conn.execute(
                "UPDATE idempotency SET status = ?, provider_reference = ?,"
                " response_digest = ?, updated_at = ?"
                " WHERE idempotency_key = ?",
                (status, provider_reference, digest, datetime.now(UTC).isoformat(), key),
            )

    def in_flight(self) -> list[LedgerEntry]:
        """Operations that were issued but never resolved.

        These are the ones a restart must reconcile against the provider
        rather than reissue. Reissuing is what causes double charges.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM idempotency WHERE status = 'in_flight' ORDER BY created_at"
            ).fetchall()
        return [_to_entry(r) for r in rows]

    def for_case(self, case_id: str) -> list[LedgerEntry]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM idempotency WHERE case_id = ? ORDER BY created_at",
                (case_id,),
            ).fetchall()
        return [_to_entry(r) for r in rows]

    def attempts_used(self, case_id: str) -> int:
        """Attempts already spent, counted from the ledger rather than from
        case state. The ledger is the record of what actually reached the
        provider; in-memory case state can be stale after a crash."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM idempotency WHERE case_id = ?"
                " AND action IN ('retry_now', 'retry_scheduled', 'retry_alternate_rail')",
                (case_id,),
            ).fetchone()
        return int(row["n"])


def _to_entry(row: sqlite3.Row) -> LedgerEntry:
    return LedgerEntry(
        idempotency_key=row["idempotency_key"],
        case_id=row["case_id"],
        action=ActionType(row["action"]),
        attempt=int(row["attempt"]),
        amount_paise=int(row["amount_paise"]),
        status=row["status"],
        provider_reference=row["provider_reference"],
        response_digest=row["response_digest"],
        created_at=datetime.fromisoformat(row["created_at"]),
    )
