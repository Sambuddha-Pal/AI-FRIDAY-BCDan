"""
Minimal SQLite persistence layer. Kept dependency-free (stdlib sqlite3)
so the demo runs with zero external DB setup.
"""
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

DB_PATH = "complaints.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS complaints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    original_text TEXT NOT NULL,
    anonymized_text TEXT NOT NULL,
    summary TEXT,
    category TEXT,
    severity TEXT,
    sentiment TEXT,
    root_cause_hypothesis TEXT,
    suggested_action TEXT,
    status TEXT DEFAULT 'analyzed',
    edited INTEGER DEFAULT 0
);
"""


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.execute(SCHEMA)


def insert_complaint(original_text: str, anonymized_text: str) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO complaints (created_at, original_text, anonymized_text, status) "
            "VALUES (?, ?, ?, 'pending')",
            (datetime.now(timezone.utc).isoformat(), original_text, anonymized_text),
        )
        return cur.lastrowid


def update_analysis(complaint_id: int, analysis: dict):
    with get_conn() as conn:
        conn.execute(
            """UPDATE complaints SET summary=?, category=?, severity=?, sentiment=?,
               root_cause_hypothesis=?, suggested_action=?, status='analyzed'
               WHERE id=?""",
            (
                analysis.get("summary"),
                analysis.get("category"),
                analysis.get("severity"),
                analysis.get("sentiment"),
                analysis.get("root_cause_hypothesis"),
                analysis.get("suggested_action"),
                complaint_id,
            ),
        )


def update_summary_edit(complaint_id: int, summary: str, category: str, severity: str):
    with get_conn() as conn:
        conn.execute(
            "UPDATE complaints SET summary=?, category=?, severity=?, edited=1 WHERE id=?",
            (summary, category, severity, complaint_id),
        )


def delete_complaint(complaint_id: int):
    with get_conn() as conn:
        conn.execute("DELETE FROM complaints WHERE id=?", (complaint_id,))


def list_complaints() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM complaints ORDER BY id DESC").fetchall()
        return [dict(r) for r in rows]


def get_complaint(complaint_id: int) -> dict | None:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM complaints WHERE id=?", (complaint_id,)).fetchone()
        return dict(row) if row else None
