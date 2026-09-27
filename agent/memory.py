"""Long-term memory (SQLite) - this is what lets the agent get better with use.

Two things are remembered between sessions:
  * runs    - every goal + final report + the full run details (plan, findings,
              trace) + the user's rating/feedback
  * lessons - short rules the LLM distils from user feedback, which are fed
              back into the planner and writer prompts on future runs.
"""
import json
import re
import sqlite3
from contextlib import closing
from datetime import datetime

from .config import MEMORY_DB

_STOPWORDS = {
    "the", "a", "an", "of", "and", "or", "in", "on", "for", "to", "is", "are",
    "what", "how", "why", "with", "about", "vs", "latest", "2024", "2025", "2026",
}


def _conn():
    conn = sqlite3.connect(MEMORY_DB)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT, goal TEXT, report TEXT,
            rating INTEGER, feedback TEXT)"""
    )
    # migrate databases created by earlier versions
    columns = {r[1] for r in conn.execute("PRAGMA table_info(runs)")}
    if "mode" not in columns:
        conn.execute("ALTER TABLE runs ADD COLUMN mode TEXT DEFAULT 'general'")
    if "details" not in columns:
        conn.execute("ALTER TABLE runs ADD COLUMN details TEXT")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS lessons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT, lesson TEXT UNIQUE)"""
    )
    return conn


def _keywords(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2 and w not in _STOPWORDS}


def save_run(goal: str, report: str, mode: str = "general", details: dict | None = None) -> int:
    """`details` holds the rest of the run (plan, findings, trace, critique...)
    so the UI can reopen a past run exactly as it finished."""
    with closing(_conn()) as conn, conn:
        cur = conn.execute(
            "INSERT INTO runs (created_at, goal, report, mode, details) VALUES (?, ?, ?, ?, ?)",
            (datetime.now().isoformat(timespec="seconds"), goal, report, mode, json.dumps(details or {})),
        )
        return cur.lastrowid


def get_run(run_id: int) -> dict | None:
    with closing(_conn()) as conn:
        row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    if row is None:
        return None
    run = dict(row)
    run["details"] = json.loads(run["details"] or "{}")  # runs from older versions have none
    return run


def count_runs() -> int:
    with closing(_conn()) as conn:
        return conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]


def clear_all() -> None:
    """Forget every run and lesson (used by the UI's 'Clear memory')."""
    with closing(_conn()) as conn, conn:
        conn.execute("DELETE FROM runs")
        conn.execute("DELETE FROM lessons")


def rate_run(run_id: int, rating: int, feedback: str) -> None:
    with closing(_conn()) as conn, conn:
        conn.execute("UPDATE runs SET rating = ?, feedback = ? WHERE id = ?", (rating, feedback, run_id))


def add_lesson(lesson: str) -> None:
    lesson = lesson.strip()
    if not lesson:
        return
    with closing(_conn()) as conn, conn:
        conn.execute(
            "INSERT OR IGNORE INTO lessons (created_at, lesson) VALUES (?, ?)",
            (datetime.now().isoformat(timespec="seconds"), lesson),
        )


def get_lessons(limit: int = 8) -> list[str]:
    with closing(_conn()) as conn:
        rows = conn.execute("SELECT lesson FROM lessons ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [r["lesson"] for r in rows]


def find_related_reports(query: str, limit: int = 2) -> list[dict]:
    """Cheap keyword-overlap retrieval; good enough for a local memory store.
    Poorly-rated runs are skipped so bad research is not reused."""
    q = _keywords(query)
    if not q:
        return []
    with closing(_conn()) as conn:
        rows = conn.execute(
            "SELECT id, goal, report, rating FROM runs WHERE rating IS NULL OR rating > 0"
        ).fetchall()
    scored = []
    for r in rows:
        overlap = len(q & _keywords(r["goal"]))
        if overlap:
            scored.append((overlap, dict(r)))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored[:limit]]


def list_runs(limit: int | None = 20) -> list[dict]:
    with closing(_conn()) as conn:
        rows = conn.execute(
            "SELECT id, created_at, goal, mode, rating, feedback FROM runs ORDER BY id DESC LIMIT ?",
            (-1 if limit is None else limit,),  # SQLite: LIMIT -1 means no limit
        ).fetchall()
    return [dict(r) for r in rows]
