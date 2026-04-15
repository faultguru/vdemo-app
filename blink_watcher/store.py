import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id           TEXT PRIMARY KEY,
    camera       TEXT NOT NULL,
    ts           TEXT NOT NULL,
    source_url   TEXT,
    path         TEXT,
    priority     INTEGER NOT NULL DEFAULT 0,
    state        TEXT NOT NULL DEFAULT 'pending',
    attempts     INTEGER NOT NULL DEFAULT 0,
    last_error   TEXT,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_events_drain ON events(state, priority, ts DESC);
CREATE INDEX IF NOT EXISTS ix_events_camera_ts ON events(camera, ts DESC);

CREATE TABLE IF NOT EXISTS backfill_state (
    id                INTEGER PRIMARY KEY CHECK (id = 1),
    cursor_ts         TEXT NOT NULL,
    target_ts         TEXT NOT NULL,
    started_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL
);
"""

STATES = ("pending", "downloaded", "processed", "failed", "skipped")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(SCHEMA)

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        self._conn.execute("BEGIN")
        try:
            yield self._conn
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    def enqueue(self, event_id: str, camera: str, ts: datetime, source_url: Optional[str], priority: int) -> bool:
        now = _now()
        cur = self._conn.execute(
            "INSERT OR IGNORE INTO events(id, camera, ts, source_url, priority, state, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, 'pending', ?, ?)",
            (event_id, camera, ts.astimezone(timezone.utc).isoformat(), source_url, priority, now, now),
        )
        return cur.rowcount > 0

    def claim_next_pending(self) -> Optional[sqlite3.Row]:
        row = self._conn.execute(
            "SELECT * FROM events WHERE state='pending' ORDER BY priority ASC, ts DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        self._conn.execute(
            "UPDATE events SET state='claimed', updated_at=? WHERE id=? AND state='pending'",
            (_now(), row["id"]),
        )
        return row

    def mark_downloaded(self, event_id: str, path: str) -> None:
        self._conn.execute(
            "UPDATE events SET state='downloaded', path=?, updated_at=? WHERE id=?",
            (path, _now(), event_id),
        )

    def mark_processed(self, event_id: str) -> None:
        self._conn.execute(
            "UPDATE events SET state='processed', updated_at=? WHERE id=?",
            (_now(), event_id),
        )

    def mark_failed(self, event_id: str, error: str) -> None:
        self._conn.execute(
            "UPDATE events SET state='pending', attempts=attempts+1, last_error=?, updated_at=? WHERE id=?",
            (error[:500], _now(), event_id),
        )

    def latest_ts_for(self, camera: str) -> Optional[datetime]:
        row = self._conn.execute(
            "SELECT ts FROM events WHERE camera=? ORDER BY ts DESC LIMIT 1", (camera,)
        ).fetchone()
        if not row:
            return None
        return datetime.fromisoformat(row["ts"])

    def count_by_state(self) -> dict[str, int]:
        rows = self._conn.execute("SELECT state, COUNT(*) c FROM events GROUP BY state").fetchall()
        return {r["state"]: r["c"] for r in rows}

    # --- backfill state ---
    def load_backfill(self) -> Optional[sqlite3.Row]:
        return self._conn.execute("SELECT * FROM backfill_state WHERE id=1").fetchone()

    def save_backfill(self, cursor_ts: datetime, target_ts: datetime) -> None:
        now = _now()
        existing = self.load_backfill()
        if existing is None:
            self._conn.execute(
                "INSERT INTO backfill_state(id, cursor_ts, target_ts, started_at, updated_at) VALUES (1,?,?,?,?)",
                (cursor_ts.isoformat(), target_ts.isoformat(), now, now),
            )
        else:
            self._conn.execute(
                "UPDATE backfill_state SET cursor_ts=?, target_ts=?, updated_at=? WHERE id=1",
                (cursor_ts.isoformat(), target_ts.isoformat(), now),
            )

    def clear_backfill(self) -> None:
        self._conn.execute("DELETE FROM backfill_state WHERE id=1")
