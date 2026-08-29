"""Persistent memory for the tagging agent.

Tagging the same image twice costs the same as tagging it once, which hurts
during development, demos, and re-runs after a partial failure. The agent
therefore remembers what it has already decided: results are keyed by a hash of
everything that could change the answer -- the image bytes, the model, the
product context, and the few-shot examples -- so a cache hit is only ever
returned for an identical request.

SQLite is used rather than a dict so memory survives across processes; it is in
the standard library, so this adds no dependency.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time

DEFAULT_PATH = ".agent_memory.sqlite3"


def make_key(image_b64: str, model: str, context: str | None,
             examples=None) -> str:
    """Fingerprint every input that can change the predicted tags."""
    h = hashlib.sha256()
    h.update(image_b64.encode())
    h.update(b"\x00" + (model or "").encode())
    h.update(b"\x00" + (context or "").encode())
    for ex_b64, _media, ex_tags in examples or []:
        h.update(b"\x00" + hashlib.sha256(ex_b64.encode()).digest())
        h.update(b"\x00" + json.dumps(sorted(ex_tags)).encode())
    return h.hexdigest()


class TagMemory:
    """A tiny key -> tags store, plus hit/miss counters for reporting."""

    def __init__(self, path: str = DEFAULT_PATH) -> None:
        self.path = path
        self.hits = 0
        self.misses = 0
        # run_folder can tag images in parallel, so every access is guarded:
        # one sqlite connection shared across threads, plus the counters.
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS tags ("
            "  key TEXT PRIMARY KEY,"
            "  tags TEXT NOT NULL,"
            "  model TEXT,"
            "  created REAL)")
        self._conn.commit()

    def get(self, key: str) -> list[str] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT tags FROM tags WHERE key = ?", (key,)).fetchone()
            if row is None:
                self.misses += 1
                return None
            self.hits += 1
            return json.loads(row[0])

    def put(self, key: str, tags: list[str], model: str = "") -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO tags (key, tags, model, created) "
                "VALUES (?, ?, ?, ?)",
                (key, json.dumps(tags), model, time.time()))
            self._conn.commit()

    def size(self) -> int:
        with self._lock:
            return self._conn.execute(
                "SELECT COUNT(*) FROM tags").fetchone()[0]

    def summary(self) -> str:
        total = self.hits + self.misses
        rate = (100 * self.hits / total) if total else 0.0
        return (f"Memory: {self.hits} hit(s), {self.misses} miss(es) "
                f"({rate:.0f}% reused), {self.size()} entries in "
                f"{os.path.basename(self.path)}")

    def close(self) -> None:
        self._conn.close()
