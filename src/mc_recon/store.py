"""Run store: run records + manifests, the oversight/validation event trail, the remediation log,
review submissions and API idempotency keys. SQLite for the PoC.

``events`` and ``remediation_log`` are **append-only**: SQLite triggers reject UPDATE and
DELETE, so the audit trail cannot be rewritten through this store. Writes that may be retried
(graph node re-execution on resume, API retries) carry a natural key and are idempotent.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    dataset     TEXT NOT NULL,
    created_by  TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    status      TEXT NOT NULL,
    error       TEXT,
    manifest    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    seq         INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT NOT NULL,
    event_type  TEXT NOT NULL,
    actor       TEXT NOT NULL,
    subject_id  TEXT,
    payload     TEXT NOT NULL,
    dedupe_key  TEXT UNIQUE,
    at          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS events_run ON events(run_id, seq);
CREATE TABLE IF NOT EXISTS remediation_log (
    run_id        TEXT NOT NULL,
    exception_id  TEXT NOT NULL,
    entry         TEXT NOT NULL,
    logged_at     TEXT NOT NULL,
    PRIMARY KEY (run_id, exception_id)
);
CREATE TABLE IF NOT EXISTS reviews (
    run_id        TEXT NOT NULL,
    exception_id  TEXT NOT NULL,
    round         INTEGER NOT NULL,
    analyst       TEXT NOT NULL,
    decision      TEXT NOT NULL,
    at            TEXT NOT NULL,
    PRIMARY KEY (run_id, exception_id, round)
);
CREATE TABLE IF NOT EXISTS idempotency (
    key           TEXT PRIMARY KEY,
    request_hash  TEXT NOT NULL,
    status_code   INTEGER NOT NULL,
    response      TEXT NOT NULL,
    at            TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS events_no_update BEFORE UPDATE ON events
    BEGIN SELECT RAISE(ABORT, 'events are append-only'); END;
CREATE TRIGGER IF NOT EXISTS events_no_delete BEFORE DELETE ON events
    BEGIN SELECT RAISE(ABORT, 'events are append-only'); END;
CREATE TRIGGER IF NOT EXISTS log_no_update BEFORE UPDATE ON remediation_log
    BEGIN SELECT RAISE(ABORT, 'remediation log is append-only'); END;
CREATE TRIGGER IF NOT EXISTS log_no_delete BEFORE DELETE ON remediation_log
    BEGIN SELECT RAISE(ABORT, 'remediation log is append-only'); END;
"""


def _utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=str)


class ConflictError(RuntimeError):
    """A retried write disagrees with what was already recorded."""


class RunStore:
    def __init__(self, path: str | Path = ":memory:", clock: Callable[[], str] = _utcnow):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._clock = clock
        with self._lock:
            if self.path != ":memory:":
                self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        self._conn.close()

    # --- runs ------------------------------------------------------------------------------

    def create_run(self, run_id: str, dataset: str, created_by: str, manifest: dict) -> None:
        now = self._clock()
        with self._lock:
            self._conn.execute(
                "INSERT INTO runs VALUES (?,?,?,?,?,?,?,?)",
                (run_id, dataset, created_by, now, now, "created", None, _dumps(manifest)),
            )

    def update_run(
        self,
        run_id: str,
        *,
        status: str | None = None,
        error: str | None = None,
        manifest: dict | None = None,
    ) -> None:
        with self._lock:
            row = self._conn.execute(
                "SELECT status, error, manifest FROM runs WHERE run_id=?", (run_id,)
            ).fetchone()
            if row is None:
                raise KeyError(run_id)
            merged = {**json.loads(row["manifest"]), **(manifest or {})}
            self._conn.execute(
                "UPDATE runs SET status=?, error=?, manifest=?, updated_at=? WHERE run_id=?",
                (
                    status or row["status"],
                    error if error is not None else row["error"],
                    _dumps(merged),
                    self._clock(),
                    run_id,
                ),
            )

    def get_run(self, run_id: str) -> dict | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            return None
        out = dict(row)
        out["manifest"] = json.loads(out["manifest"])
        return out

    # --- events (append-only) --------------------------------------------------------------

    def record_event(
        self,
        run_id: str,
        event_type: str,
        actor: str,
        payload: dict,
        *,
        subject_id: str | None = None,
        dedupe_key: str | None = None,
    ) -> bool:
        """Append an event. Returns False if ``dedupe_key`` was already recorded."""
        with self._lock:
            cur = self._conn.execute(
                "INSERT OR IGNORE INTO events "
                "(run_id, event_type, actor, subject_id, payload, dedupe_key, at) "
                "VALUES (?,?,?,?,?,?,?)",
                (run_id, event_type, actor, subject_id, _dumps(payload), dedupe_key, self._clock()),
            )
            return cur.rowcount == 1

    def events(self, run_id: str, event_type: str | None = None) -> list[dict]:
        sql = "SELECT * FROM events WHERE run_id=?"
        args: tuple = (run_id,)
        if event_type:
            sql += " AND event_type=?"
            args += (event_type,)
        with self._lock:
            rows = self._conn.execute(sql + " ORDER BY seq", args).fetchall()
        return [{**dict(r), "payload": json.loads(r["payload"])} for r in rows]

    # --- remediation log (append-only, idempotent) -----------------------------------------

    def append_log(self, run_id: str, exception_id: str, entry: dict) -> bool:
        """Insert once per ``(run_id, exception_id)``; a retry with the same entry is a no-op."""
        body = _dumps(entry)
        with self._lock:
            cur = self._conn.execute(
                "INSERT OR IGNORE INTO remediation_log VALUES (?,?,?,?)",
                (run_id, exception_id, body, self._clock()),
            )
            if cur.rowcount == 1:
                return True
            existing = self._conn.execute(
                "SELECT entry FROM remediation_log WHERE run_id=? AND exception_id=?",
                (run_id, exception_id),
            ).fetchone()
        if existing["entry"] != body:
            raise ConflictError(f"remediation log entry {run_id}/{exception_id} already differs")
        return False

    def remediation_log(self, run_id: str) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM remediation_log WHERE run_id=? ORDER BY exception_id", (run_id,)
            ).fetchall()
        return [{**json.loads(r["entry"]), "logged_at": r["logged_at"]} for r in rows]

    # --- reviews ---------------------------------------------------------------------------

    def put_review(
        self, run_id: str, exception_id: str, round_: int, analyst: str, decision: dict
    ) -> bool:
        body = _dumps(decision)
        with self._lock:
            cur = self._conn.execute(
                "INSERT OR IGNORE INTO reviews VALUES (?,?,?,?,?,?)",
                (run_id, exception_id, round_, analyst, body, self._clock()),
            )
            if cur.rowcount == 1:
                return True
            existing = self._conn.execute(
                "SELECT decision FROM reviews WHERE run_id=? AND exception_id=? AND round=?",
                (run_id, exception_id, round_),
            ).fetchone()
        if existing["decision"] != body:
            raise ConflictError(
                f"exception {exception_id} already reviewed in round {round_} with a different "
                "decision"
            )
        return False

    def reviews(self, run_id: str, round_: int) -> dict[str, dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT exception_id, decision FROM reviews WHERE run_id=? AND round=?",
                (run_id, round_),
            ).fetchall()
        return {r["exception_id"]: json.loads(r["decision"]) for r in rows}

    # --- idempotency keys ------------------------------------------------------------------

    def get_idempotent(self, key: str) -> dict | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM idempotency WHERE key=?", (key,)).fetchone()
        if row is None:
            return None
        return {
            "request_hash": row["request_hash"],
            "status_code": row["status_code"],
            "response": json.loads(row["response"]),
        }

    def put_idempotent(self, key: str, request_hash: str, status_code: int, response: Any) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO idempotency VALUES (?,?,?,?,?)",
                (key, request_hash, status_code, _dumps(response), self._clock()),
            )
