"""Structured audit log (SQLite).

Every attribution decision is appended here BEFORE the API responds, so a record
exists even if the client disconnects (planning.md §1 step 8). A single
``audit_log`` table doubles as the lookup store that later milestones query:

  - M5 appeals look up a row by ``content_id`` and flip its ``status`` to
    ``under_review`` (and append an ``event_type='appeal'`` row beside it).
  - The per-creator interval check (M5) queries a creator's most recent entry.

M3 writes only ``event_type='classification'`` rows; ``stylo_score`` and
``appeal_reasoning`` stay NULL until M4/M5 fill them in.
"""

import sqlite3
from datetime import datetime, timezone

from config import DB_PATH

# Column order is reused by write_entry() and get_log().
_COLUMNS = (
    "content_id",
    "creator_id",
    "timestamp",
    "event_type",
    "attribution",
    "confidence",
    "llm_score",
    "stylo_score",
    "llm_status",
    "injection_suspected",
    "status",
    "appeal_reasoning",
)


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # rows behave like dicts
    return conn


def init_db():
    """Create the audit_log table if it does not exist. Idempotent."""
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_log (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                content_id          TEXT    NOT NULL,
                creator_id          TEXT,
                timestamp           TEXT    NOT NULL,
                event_type          TEXT    NOT NULL DEFAULT 'classification',
                attribution         TEXT,
                confidence          REAL,
                llm_score           REAL,
                stylo_score         REAL,
                llm_status          TEXT,
                injection_suspected INTEGER DEFAULT 0,
                status              TEXT    NOT NULL DEFAULT 'classified',
                appeal_reasoning    TEXT
            )
            """
        )
        # Speeds up the content_id lookups appeals will do in M5.
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_audit_content_id "
            "ON audit_log (content_id)"
        )


def now_iso():
    """Current UTC time as an ISO-8601 string with a trailing Z."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def write_entry(entry):
    """Append one structured row. Missing keys default to None / sensible values.

    ``entry`` is a dict using the keys in ``_COLUMNS``. ``timestamp`` is stamped
    here if not supplied.
    """
    entry = dict(entry)
    entry.setdefault("timestamp", now_iso())
    entry.setdefault("event_type", "classification")
    entry.setdefault("status", "classified")
    entry.setdefault("injection_suspected", 0)

    values = [entry.get(col) for col in _COLUMNS]
    placeholders = ", ".join("?" for _ in _COLUMNS)
    columns = ", ".join(_COLUMNS)
    with _connect() as conn:
        conn.execute(
            f"INSERT INTO audit_log ({columns}) VALUES ({placeholders})", values
        )


def get_log(limit=20):
    """Return the most recent log entries (newest first) as a list of dicts."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(row) for row in rows]
