"""The SQL migration (used by the PostGIS container) must match api/db.py's metadata."""

from __future__ import annotations

import re
from pathlib import Path

from sqlalchemy import inspect

from api.db import create_schema, make_engine, metadata

SQL = Path(__file__).resolve().parents[1] / "migrations" / "001_init.sql"
CONSTRAINT_WORDS = {"PRIMARY", "CONSTRAINT", "UNIQUE", "FOREIGN", "CHECK"}


def _sql_tables() -> dict[str, set[str]]:
    text = SQL.read_text()
    out: dict[str, set[str]] = {}
    for name, body in re.findall(r"CREATE TABLE IF NOT EXISTS (\w+) \((.*?)\n\);", text, re.S):
        cols = set()
        for line in body.splitlines():
            word = line.strip().split(" ")[0] if line.strip() else ""
            if word and word.upper() not in CONSTRAINT_WORDS and not word.startswith("--"):
                cols.add(word)
        out[name] = cols
    return out


def test_migration_matches_metadata():
    sql = _sql_tables()
    meta = {t.name: {c.name for c in t.columns} for t in metadata.sorted_tables}
    assert sql == meta


def test_create_schema_sqlite(tmp_path):
    eng = make_engine(f"sqlite:///{tmp_path / 'x.db'}")
    create_schema(eng)
    create_schema(eng)  # idempotent
    names = set(inspect(eng).get_table_names())
    assert {"players", "plans", "votes", "resident_reactions", "chats", "jobs", "households_real"} <= names
