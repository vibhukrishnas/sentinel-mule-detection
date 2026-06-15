"""
Durable case store — persists reviews/decisions beyond a single browser session.

Backed by SQLite at $SENTINEL_DB (default: a temp-dir file). On a real/local deployment
this is durable; on Streamlit Community Cloud the filesystem is ephemeral, so it persists
across refreshes/sessions within a running container but resets on reboot — a production
deployment would point SENTINEL_DB at a managed Postgres/SQL store. All calls are guarded
by the caller so a storage failure never breaks scoring.
"""
from __future__ import annotations
import os, sqlite3, tempfile
from pathlib import Path

DB_PATH = Path(os.getenv("SENTINEL_DB", str(Path(tempfile.gettempdir()) / "sentinel_cases.db")))
_DDL = ("CREATE TABLE IF NOT EXISTS cases (time_utc TEXT, account INTEGER, score INTEGER, "
        "band TEXT, probability REAL, threshold REAL, decision TEXT, ground_truth TEXT, "
        "ring TEXT, action TEXT, session TEXT)")


def _conn():
    c = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=5)
    c.execute(_DDL)
    return c


def log_event(row: dict, session: str):
    with _conn() as c:
        c.execute("INSERT INTO cases VALUES (?,?,?,?,?,?,?,?,?,?,?)", (
            row.get("time (UTC)"), int(row["account"]), int(row["score"]), row["band"],
            float(row["probability"]), float(row["threshold"]), row["decision"],
            row.get("ground_truth", "unknown"), row.get("ring", "—"),
            row.get("action", "reviewed"), session))


def load_all():
    import pandas as pd
    with _conn() as c:
        return pd.read_sql_query("SELECT * FROM cases ORDER BY rowid DESC", c)


def count() -> int:
    with _conn() as c:
        return int(c.execute("SELECT COUNT(*) FROM cases").fetchone()[0])
