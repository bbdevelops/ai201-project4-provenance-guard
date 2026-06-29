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
    "perplexity_score",
    "perplexity_status",
    "llm_status",
    "injection_suspected",
    "status",
    "appeal_reasoning",
)

# Columns added after the original schema shipped (Ensemble Detection stretch).
# CREATE TABLE IF NOT EXISTS won't add these to a pre-existing audit_log.db, so
# init_db() ALTERs them in idempotently. (column_name, SQL type).
_ADDED_COLUMNS = (
    ("perplexity_score", "REAL"),
    ("perplexity_status", "TEXT"),
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
                perplexity_score    REAL,
                perplexity_status   TEXT,
                llm_status          TEXT,
                injection_suspected INTEGER DEFAULT 0,
                status              TEXT    NOT NULL DEFAULT 'classified',
                appeal_reasoning    TEXT
            )
            """
        )
        # Migrate pre-existing databases: add any later columns the CREATE above
        # won't retrofit. Ignore "duplicate column" so init_db stays idempotent.
        for name, sql_type in _ADDED_COLUMNS:
            try:
                conn.execute(
                    f"ALTER TABLE audit_log ADD COLUMN {name} {sql_type}"
                )
            except sqlite3.OperationalError:
                pass  # column already exists
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


def get_submission(content_id):
    """Return the original classification row for ``content_id`` (or None).

    Used by /appeal to (a) confirm the content_id exists and (b) carry the
    original scores onto the appeal row. The idx_audit_content_id index makes
    this lookup fast.
    """
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM audit_log WHERE content_id = ? "
            "AND event_type = 'classification' ORDER BY id DESC LIMIT 1",
            (content_id,),
        ).fetchone()
    return dict(row) if row else None


def update_status(content_id, new_status):
    """Flip the status of every row for ``content_id`` (e.g. to under_review)."""
    with _connect() as conn:
        conn.execute(
            "UPDATE audit_log SET status = ? WHERE content_id = ?",
            (new_status, content_id),
        )
