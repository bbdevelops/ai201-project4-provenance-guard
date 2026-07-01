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
import time
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
        conn.execute("PRAGMA journal_mode=WAL;")
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


def _execute_with_retry(query, params=(), max_retries=5, initial_backoff=0.05):
    """Execute a query with exponential backoff on SQLite locking errors."""
    backoff = initial_backoff
    for attempt in range(max_retries):
        try:
            with _connect() as conn:
                conn.execute(query, params)
            return
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e).lower() and attempt < max_retries - 1:
                time.sleep(backoff)
                backoff *= 2
            else:
                raise


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
    _execute_with_retry(f"INSERT INTO audit_log ({columns}) VALUES ({placeholders})", values)


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
    _execute_with_retry(
        "UPDATE audit_log SET status = ? WHERE content_id = ?",
        (new_status, content_id),
    )


def get_dashboard_metrics():
    """Aggregate audit-log data for the analytics dashboard.

    Returns a dict with everything the ``/dashboard`` template needs:
      - Detection pattern: counts per attribution band + total classifications.
      - Appeal rate: total appeals / total classifications.
      - Injection-flagged rate: flagged classifications / total classifications.

    All rates default to 0.0 when the database is empty (no division-by-zero).
    """
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT 
                SUM(CASE WHEN event_type = 'classification' THEN 1 ELSE 0 END) AS total_classifications,
                SUM(CASE WHEN event_type = 'classification' AND attribution = 'likely_ai' THEN 1 ELSE 0 END) AS likely_ai_count,
                SUM(CASE WHEN event_type = 'classification' AND attribution = 'uncertain' THEN 1 ELSE 0 END) AS uncertain_count,
                SUM(CASE WHEN event_type = 'classification' AND attribution = 'likely_human' THEN 1 ELSE 0 END) AS likely_human_count,
                SUM(CASE WHEN event_type = 'appeal' THEN 1 ELSE 0 END) AS total_appeals,
                SUM(CASE WHEN event_type = 'classification' AND injection_suspected = 1 THEN 1 ELSE 0 END) AS total_injection_flagged
            FROM audit_log
            """
        ).fetchone()

    total_classifications = row["total_classifications"] or 0
    likely_ai = row["likely_ai_count"] or 0
    uncertain = row["uncertain_count"] or 0
    likely_human = row["likely_human_count"] or 0
    total_appeals = row["total_appeals"] or 0
    total_injection_flagged = row["total_injection_flagged"] or 0

    appeal_rate = (
        round(total_appeals / total_classifications, 4)
        if total_classifications > 0
        else 0.0
    )
    injection_rate = (
        round(total_injection_flagged / total_classifications, 4)
        if total_classifications > 0
        else 0.0
    )

    return {
        "total_classifications": total_classifications,
        "likely_ai_count": likely_ai,
        "uncertain_count": uncertain,
        "likely_human_count": likely_human,
        "total_appeals": total_appeals,
        "appeal_rate": appeal_rate,
        "total_injection_flagged": total_injection_flagged,
        "injection_rate": injection_rate,
    }
