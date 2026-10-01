"""Database schema (spec 9.2) and engine setup.

Works with Postgres (JSONB columns, `postgresql+psycopg://...`) and SQLite
(`sqlite:///...`, JSON columns) for local dev and tests. The same schema is
written as plain SQL in `api/migrations/001_init.sql` (used by the PostGIS
docker container's initdb); the app also calls `create_schema()` at startup,
which creates any missing tables and leaves existing ones alone.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    Engine,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    SmallInteger,
    String,
    Table,
    Text,
    create_engine,
    event,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.pool import StaticPool

JSONType = JSON().with_variant(JSONB(), "postgresql")

metadata = MetaData()


def utcnow() -> datetime:
    return datetime.now(UTC)


def _created() -> Column:
    return Column("created_at", DateTime(timezone=True), nullable=False, default=utcnow,
                  server_default=func.now())


def _updated() -> Column:
    return Column("updated_at", DateTime(timezone=True), nullable=False, default=utcnow,
                  onupdate=utcnow, server_default=func.now())


players = Table(
    "players", metadata,
    Column("id", String(36), primary_key=True),
    Column("display_name", String(80), nullable=True),
    _created(),
)

plans = Table(
    "plans", metadata,
    Column("id", String(36), primary_key=True),
    Column("mission", String(64), nullable=False),
    Column("title", String(200), nullable=False),
    Column("pitch", Text, nullable=False, server_default=""),
    Column("author_id", String(36), ForeignKey("players.id", ondelete="SET NULL"), nullable=True),
    Column("tools", JSONType, nullable=False),
    Column("report", JSONType, nullable=True),
    # API field name is `check`; the column avoids the SQL reserved word.
    Column("plan_check", JSONType, nullable=True),
    Column("status", String(16), nullable=False, server_default="draft"),
    Column("votes", Integer, nullable=False, server_default="0"),
    _created(),
    _updated(),
    Index("ix_plans_mission_created", "mission", "created_at"),
    Index("ix_plans_mission_votes", "mission", "votes"),
)

votes = Table(
    "votes", metadata,
    Column("plan_id", String(36), ForeignKey("plans.id", ondelete="CASCADE"), primary_key=True),
    Column("player_id", String(36), ForeignKey("players.id", ondelete="CASCADE"), primary_key=True),
    Column("value", SmallInteger, nullable=False),
    _created(),
    CheckConstraint("value IN (-1, 1)", name="ck_votes_value"),
)

resident_reactions = Table(
    "resident_reactions", metadata,
    Column("plan_id", String(36), ForeignKey("plans.id", ondelete="CASCADE"), primary_key=True),
    Column("persona_id", Integer, primary_key=True),
    Column("approval", Float, nullable=False),
    Column("approves", Boolean, nullable=False),
    Column("deltas", JSONType, nullable=True),
    Column("text", Text, nullable=True),
    _created(),
    _updated(),
)

chats = Table(
    "chats", metadata,
    Column("player_id", String(36), ForeignKey("players.id", ondelete="CASCADE"), primary_key=True),
    Column("persona_id", Integer, primary_key=True),
    Column("messages", JSONType, nullable=False),
    _updated(),
)

jobs = Table(
    "jobs", metadata,
    Column("id", String(36), primary_key=True),
    Column("plan_id", String(36), ForeignKey("plans.id", ondelete="CASCADE"), nullable=False),
    Column("status", String(16), nullable=False, server_default="queued"),
    Column("progress", Float, nullable=False, server_default="0"),
    Column("message", Text, nullable=False, server_default=""),
    Column("error", Text, nullable=True),
    Column("started_at", DateTime(timezone=True), nullable=True),
    Column("finished_at", DateTime(timezone=True), nullable=True),
    _created(),
    _updated(),
    Index("ix_jobs_plan_id", "plan_id"),
)

townhalls = Table(
    "townhalls", metadata,
    Column("plan_id", String(36), ForeignKey("plans.id", ondelete="CASCADE"), primary_key=True),
    Column("speakers", JSONType, nullable=False),
    _created(),
    _updated(),
)

# Reserved for real-household home claiming (spec 9.2, 13.4). Created empty; all nullable.
households_real = Table(
    "households_real", metadata,
    Column("id", String(36), primary_key=True),
    Column("player_id", String(36), ForeignKey("players.id", ondelete="SET NULL"), nullable=True),
    Column("building_id", BigInteger, nullable=True),
    Column("household_size", SmallInteger, nullable=True),
    Column("vehicles", SmallInteger, nullable=True),
    Column("n_kids", SmallInteger, nullable=True),
    Column("school_ids", JSONType, nullable=True),
    Column("work_area", JSONType, nullable=True),
    Column("departure_times", JSONType, nullable=True),
    Column("trip_data", JSONType, nullable=True),
    Column("opt_in", JSONType, nullable=True),
    Column("consent_at", DateTime(timezone=True), nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=True),
)


def make_engine(url: str) -> Engine:
    if url.startswith("sqlite"):
        kwargs: dict[str, Any] = {"connect_args": {"check_same_thread": False, "timeout": 30}}
        if url in ("sqlite://", "sqlite:///:memory:"):
            kwargs["poolclass"] = StaticPool
        engine = create_engine(url, **kwargs)

        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_conn: Any, _record: Any) -> None:
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            cur.execute("PRAGMA journal_mode=WAL")
            cur.close()

        return engine
    return create_engine(url, pool_pre_ping=True, pool_size=10, max_overflow=10)


def create_schema(engine: Engine) -> None:
    """Create missing tables (PostGIS is optional and only enabled by the SQL migration)."""
    metadata.create_all(engine, checkfirst=True)


def clean_json(obj: Any) -> Any:
    """Make sim output JSON safe: numpy scalars/arrays -> python, NaN/inf -> None."""
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [clean_json(v) for v in obj]
    if isinstance(obj, datetime):
        return iso(obj)
    item = getattr(obj, "item", None)  # numpy scalar
    if callable(item) and getattr(obj, "ndim", 1) == 0:
        return clean_json(item())
    tolist = getattr(obj, "tolist", None)  # numpy array
    if callable(tolist):
        return clean_json(tolist())
    if hasattr(obj, "model_dump"):
        return clean_json(obj.model_dump())
    return str(obj)


def iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).isoformat().replace("+00:00", "Z")
