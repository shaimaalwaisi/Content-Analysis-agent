"""The database tool: where a run's tagged images are kept.

This is deliberately not the same store as `agent.memory`. That one is a cache
-- disposable, keyed by a fingerprint of the request, and cleared whenever you
want the model re-asked. This one is the content creator's record: one durable
row per image per run, which is what the results table in the UI reads.
Sharing one file would mean clearing the cache also deleted the history.

Rows carry only what the agent itself knows (name, path, category, product,
tags, the model's reasons). Price and view counts stay in meta_data.xlsx and
are joined at display time, so correcting the sheet never means re-tagging.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field

from agent.logconf import get_logger

log = get_logger(__name__)

DEFAULT_PATH = "results.sqlite3"


def new_run_id(now: float | None = None) -> str:
    """A sortable id for one batch, matching the naming used in results/."""
    return time.strftime("%Y%m%d-%H%M%S", time.localtime(now or time.time()))


@dataclass
class Tagging:
    """One image, as the results table shows it."""

    run_id: str
    image_path: str
    tags: list[str] = field(default_factory=list)
    highlights: list[str] = field(default_factory=list)
    rationale: dict[str, str] = field(default_factory=dict)
    category: str = ""
    product: str = ""
    description: str = ""
    specs: str = ""
    model: str = ""
    attempts: int = 1
    cached: bool = False
    created: float = 0.0

    @property
    def image_name(self) -> str:
        return os.path.basename(self.image_path)


class ResultStore:
    """A tiny append-only table of tagged images, plus the reads the UI needs.

    One connection shared across threads (run_folder tags in parallel) with a
    lock around every statement, exactly as TagMemory does.
    """

    def __init__(self, path: str = DEFAULT_PATH) -> None:
        self.path = path
        self.writes = 0
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS taggings ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  run_id TEXT NOT NULL,"
            "  created REAL NOT NULL,"
            "  image_name TEXT NOT NULL,"
            "  image_path TEXT NOT NULL,"
            "  category TEXT,"
            "  product TEXT,"
            "  highlights TEXT NOT NULL,"   # JSON list
            "  tags TEXT NOT NULL,"         # JSON list
            "  rationale TEXT NOT NULL,"    # JSON object, tag -> why
            "  description TEXT,"
            "  specs TEXT,"
            "  model TEXT,"
            "  attempts INTEGER,"
            "  cached INTEGER)")
        # Databases written before description and specs existed are still
        # readable; adding the columns is cheaper than asking for a re-run.
        existing = {row[1] for row in
                    self._conn.execute("PRAGMA table_info(taggings)")}
        for column in ("description", "specs"):
            if column not in existing:
                self._conn.execute(
                    f"ALTER TABLE taggings ADD COLUMN {column} TEXT")
        # A re-run of the same image inside one run replaces its row rather
        # than doubling it, so 10 images always render as 10 rows.
        self._conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS taggings_run_image "
            "ON taggings (run_id, image_path)")
        self._conn.commit()

    def put(self, row: Tagging) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO taggings (run_id, created, image_name, "
                " image_path, category, product, highlights, tags, rationale,"
                " description, specs, model, attempts, cached) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT (run_id, image_path) DO UPDATE SET "
                " created=excluded.created, highlights=excluded.highlights,"
                " tags=excluded.tags, rationale=excluded.rationale,"
                " category=excluded.category, product=excluded.product,"
                " description=excluded.description, specs=excluded.specs,"
                " model=excluded.model, attempts=excluded.attempts,"
                " cached=excluded.cached",
                (row.run_id, row.created or time.time(), row.image_name,
                 row.image_path, row.category, row.product,
                 json.dumps(row.highlights), json.dumps(row.tags),
                 json.dumps(row.rationale), row.description, row.specs,
                 row.model, row.attempts, int(row.cached)))
            self._conn.commit()
            self.writes += 1

    # ---- reads -----------------------------------------------------------
    def runs(self) -> list[dict]:
        """Every run, newest first: id, when it ran, and how many images."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT run_id, COUNT(*) AS images, MAX(created) AS created "
                "FROM taggings GROUP BY run_id ORDER BY created DESC").fetchall()
        return [dict(r) for r in rows]

    def latest_run(self) -> str | None:
        runs = self.runs()
        return runs[0]["run_id"] if runs else None

    def rows(self, run_id: str | None = None, limit: int | None = None
             ) -> list[dict]:
        """Rows for one run (the newest by default), in tagging order."""
        run_id = run_id or self.latest_run()
        if run_id is None:
            return []
        sql = ("SELECT * FROM taggings WHERE run_id = ? "
               "ORDER BY image_name ASC")
        params: tuple = (run_id,)
        if limit:
            sql += " LIMIT ?"
            params += (limit,)
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        out = []
        for r in rows:
            rec = dict(r)
            rec["tags"] = json.loads(rec["tags"])
            rec["highlights"] = json.loads(rec["highlights"])
            rec["rationale"] = json.loads(rec["rationale"])
            rec["cached"] = bool(rec["cached"])
            out.append(rec)
        return out

    def summary(self) -> str:
        return f"Stored {self.writes} row(s) in {self.path}"

    def close(self) -> None:
        with self._lock:
            self._conn.close()
