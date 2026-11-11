"""Concurrency-safe unit ledger (SQLite, WAL).

Claiming uses ``BEGIN IMMEDIATE`` so that N concurrent agents each receive a
distinct unit; loosening this transaction corrupts work assignment.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

from . import DB, STATE
from .util import ensure

SCHEMA = """
CREATE TABLE IF NOT EXISTS units(
  id TEXT PRIMARY KEY,
  module TEXT NOT NULL,
  kind TEXT NOT NULL CHECK(kind IN ('host','kernel','stub','shim','inline')),
  symbol TEXT NOT NULL,
  demangled TEXT,
  source_file TEXT,
  source_line INTEGER,
  address TEXT,
  byte_size INTEGER,
  status TEXT NOT NULL DEFAULT 'open'
    CHECK(status IN ('open','claimed','decompiled','verified','blocked','waived')),
  owner TEXT, claim_ts TEXT, update_ts TEXT,
  evidence TEXT,
  gates TEXT,
  notes TEXT
);
CREATE TABLE IF NOT EXISTS claims(
  id INTEGER PRIMARY KEY,
  unit_id TEXT NOT NULL,
  agent TEXT NOT NULL,
  ts TEXT NOT NULL,
  released INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS gate_runs(
  id INTEGER PRIMARY KEY,
  unit_id TEXT NOT NULL,
  gate TEXT NOT NULL,
  ts TEXT NOT NULL,
  ok INTEGER NOT NULL,
  detail TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS unknowns(
  id INTEGER PRIMARY KEY,
  unit_id TEXT NOT NULL,
  text TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'open',
  ts TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS units_status_module ON units(status, module);
CREATE INDEX IF NOT EXISTS gate_runs_unit ON gate_runs(unit_id, gate);
"""


def connect() -> sqlite3.Connection:
    ensure(STATE)
    conn = sqlite3.connect(str(DB), timeout=30.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init() -> None:
    with connect() as c:
        c.executescript(SCHEMA)


@contextmanager
def transaction(conn: sqlite3.Connection, immediate: bool = True):
    conn.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
    try:
        yield conn
    except Exception:
        conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")


def record_gate(unit_id: str, gate: str, ok: bool, detail: dict | str) -> None:
    import json
    from datetime import datetime, timezone
    detail_s = detail if isinstance(detail, str) else json.dumps(detail, sort_keys=True)
    with connect() as c:
        c.execute(
            "INSERT INTO gate_runs(unit_id, gate, ts, ok, detail) VALUES(?,?,?,?,?)",
            (unit_id, gate, datetime.now(timezone.utc).isoformat(timespec="seconds"),
             1 if ok else 0, detail_s),
        )
