"""The append-only event store.

Insert-only by construction: there is no update or delete method, and the
schema enforces monotonic sequence numbers per case. If a fact turns out to
be wrong, the correction is a new event, not an edit — the same discipline a
financial ledger uses, for the same reason.

SQLite rather than Postgres because the whole run must be reproducible from a
file a reviewer can open. `recovery.db` ships with the demo; anyone can replay
it without standing up infrastructure.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from recovery.ledger.events import Event, EventType

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    sequence     INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id      TEXT NOT NULL,
    event_type   TEXT NOT NULL,
    occurred_at  TEXT NOT NULL,
    payload      TEXT NOT NULL,
    case_seq     INTEGER NOT NULL,
    UNIQUE (case_id, case_seq)
);
CREATE INDEX IF NOT EXISTS idx_events_case ON events(case_id, case_seq);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);

CREATE TRIGGER IF NOT EXISTS events_are_immutable
BEFORE UPDATE ON events
BEGIN
    SELECT RAISE(ABORT, 'events are append-only; correct with a new event');
END;

CREATE TRIGGER IF NOT EXISTS events_are_permanent
BEFORE DELETE ON events
BEGIN
    SELECT RAISE(ABORT, 'events are append-only; deletion is not permitted');
END;
"""


class LedgerError(RuntimeError):
    pass


class EventStore:
    """Append-only store. No update, no delete — enforced by SQL triggers as
    well as by the absence of methods, because a future contributor reaching
    for `conn.execute("UPDATE ...")` should hit a wall rather than a
    convention."""

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
            conn.execute("PRAGMA foreign_keys=ON")
            yield conn
        finally:
            conn.close()

    # -- writing -----------------------------------------------------------

    def append(self, event: Event) -> Event:
        """Append one event, assigning its per-case sequence number."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(case_seq), 0) AS n FROM events WHERE case_id = ?",
                (event.case_id,),
            ).fetchone()
            case_seq = int(row["n"]) + 1
            cursor = conn.execute(
                "INSERT INTO events (case_id, event_type, occurred_at, payload, case_seq)"
                " VALUES (?, ?, ?, ?, ?)",
                (
                    event.case_id,
                    event.event_type.value,
                    event.occurred_at.isoformat(),
                    json.dumps(event.payload, sort_keys=True, default=str),
                    case_seq,
                ),
            )
        return event.model_copy(update={"sequence": cursor.lastrowid})

    def append_many(self, events: Sequence[Event]) -> list[Event]:
        return [self.append(e) for e in events]

    # -- reading -----------------------------------------------------------

    def for_case(self, case_id: str) -> list[Event]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM events WHERE case_id = ? ORDER BY case_seq", (case_id,)
            ).fetchall()
        return [_to_event(r) for r in rows]

    def of_type(self, event_type: EventType, limit: int | None = None) -> list[Event]:
        sql = "SELECT * FROM events WHERE event_type = ? ORDER BY sequence"
        params: tuple[object, ...] = (event_type.value,)
        if limit is not None:
            sql += " LIMIT ?"
            params = (*params, limit)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_to_event(r) for r in rows]

    def case_ids(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute("SELECT DISTINCT case_id FROM events ORDER BY case_id").fetchall()
        return [r["case_id"] for r in rows]

    def count(self) -> int:
        with self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) AS n FROM events").fetchone()["n"])

    # -- integrity ---------------------------------------------------------

    def verify_sequences(self) -> list[str]:
        """Return case ids whose sequence numbers have a gap.

        A gap means an event was written outside this store or a transaction
        was lost. Neither should happen; both must be visible if they do,
        because a trail with a hole in it cannot be relied on for the parts
        that remain.
        """
        broken: list[str] = []
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT case_id, COUNT(*) AS n, MAX(case_seq) AS hi FROM events GROUP BY case_id"
            ).fetchall()
        for row in rows:
            if int(row["n"]) != int(row["hi"]):
                broken.append(row["case_id"])
        return broken


def _to_event(row: sqlite3.Row) -> Event:
    return Event(
        case_id=row["case_id"],
        event_type=EventType(row["event_type"]),
        occurred_at=datetime.fromisoformat(row["occurred_at"]),
        payload=json.loads(row["payload"]),
        sequence=int(row["sequence"]),
    )
