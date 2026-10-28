"""SQLite memory: long-term facts and a rolling dialog history."""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

MAX_STORED_MESSAGES = 500  # older rows are dropped, history_len picks what the LLM sees

_SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


@dataclass(frozen=True)
class Fact:
    id: int
    text: str
    created_at: str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class MemoryStore:
    """Thread-safe wrapper around data/limi.db (tables: facts, messages)."""

    def __init__(self, db_path: str | Path) -> None:
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        with self._lock, self._conn:
            self._conn.executescript(_SCHEMA)

    # facts ------------------------------------------------------------------

    def add_fact(self, text: str) -> bool:
        """Save a fact. Returns False for empty text or an exact duplicate."""
        text = " ".join(text.split())
        if not text:
            return False
        with self._lock, self._conn:
            dup = self._conn.execute(
                "SELECT 1 FROM facts WHERE lower(text) = lower(?)", (text,)
            ).fetchone()
            if dup:
                return False
            self._conn.execute("INSERT INTO facts (text, created_at) VALUES (?, ?)", (text, _now()))
        return True

    def facts(self) -> list[Fact]:
        with self._lock:
            rows = self._conn.execute("SELECT id, text, created_at FROM facts ORDER BY id").fetchall()
        return [Fact(*row) for row in rows]

    def facts_block(self) -> str:
        """Text block for the system prompt, empty when nothing is known."""
        facts = self.facts()
        if not facts:
            return ""
        return "What you know about the user:\n" + "\n".join(f"- {f.text}" for f in facts)

    # dialog history ---------------------------------------------------------

    def add_message(self, role: str, content: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO messages (role, content, created_at) VALUES (?, ?, ?)",
                (role, content, _now()),
            )
            self._conn.execute(
                "DELETE FROM messages WHERE id <= "
                "(SELECT id FROM messages ORDER BY id DESC LIMIT 1 OFFSET ?)",
                (MAX_STORED_MESSAGES,),
            )

    def recent_messages(self, n: int) -> list[dict[str, str]]:
        """The last n messages, oldest first, as {'role', 'content'} dicts."""
        if n <= 0:
            return []
        with self._lock:
            rows = self._conn.execute(
                "SELECT role, content FROM messages ORDER BY id DESC LIMIT ?", (n,)
            ).fetchall()
        return [{"role": r, "content": c} for r, c in reversed(rows)]

    def message_count(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0])

    # wipe -------------------------------------------------------------------

    def wipe(self) -> None:
        """Delete every fact and every message."""
        with self._lock:
            with self._conn:
                self._conn.execute("DELETE FROM facts")
                self._conn.execute("DELETE FROM messages")
                self._conn.execute("DELETE FROM sqlite_sequence WHERE name IN ('facts', 'messages')")
            self._conn.execute("VACUUM")

    def close(self) -> None:
        with self._lock:
            self._conn.close()
