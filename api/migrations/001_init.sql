-- Redraw schema v1 (spec 9.2). Mirrors api/db.py; keep both in sync.
-- Run automatically by the postgis docker container on first start
-- (mounted at /docker-entrypoint-initdb.d). The API also creates any missing
-- tables itself at startup (SQLAlchemy metadata.create_all), so this file is
-- safe to skip for SQLite dev databases.

-- PostGIS is optional for v1 tables; do not fail where the extension is absent.
DO $$
BEGIN
    CREATE EXTENSION IF NOT EXISTS postgis;
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'PostGIS not available (%); continuing without it', SQLERRM;
END
$$;

CREATE TABLE IF NOT EXISTS players (
    id            VARCHAR(36) PRIMARY KEY,
    display_name  VARCHAR(80),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS plans (
    id          VARCHAR(36) PRIMARY KEY,
    mission     VARCHAR(64)  NOT NULL,
    title       VARCHAR(200) NOT NULL,
    pitch       TEXT         NOT NULL DEFAULT '',
    author_id   VARCHAR(36)  REFERENCES players(id) ON DELETE SET NULL,
    tools       JSONB        NOT NULL,
    report      JSONB,
    plan_check  JSONB,
    status      VARCHAR(16)  NOT NULL DEFAULT 'draft',
    votes       INTEGER      NOT NULL DEFAULT 0,
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ  NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_plans_mission_created ON plans (mission, created_at);
CREATE INDEX IF NOT EXISTS ix_plans_mission_votes ON plans (mission, votes);

CREATE TABLE IF NOT EXISTS votes (
    plan_id     VARCHAR(36) NOT NULL REFERENCES plans(id) ON DELETE CASCADE,
    player_id   VARCHAR(36) NOT NULL REFERENCES players(id) ON DELETE CASCADE,
    value       SMALLINT    NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (plan_id, player_id),
    CONSTRAINT ck_votes_value CHECK (value IN (-1, 1))
);

CREATE TABLE IF NOT EXISTS resident_reactions (
    plan_id     VARCHAR(36) NOT NULL REFERENCES plans(id) ON DELETE CASCADE,
    persona_id  INTEGER     NOT NULL,
    approval    DOUBLE PRECISION NOT NULL,
    approves    BOOLEAN     NOT NULL,
    deltas      JSONB,
    text        TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (plan_id, persona_id)
);

CREATE TABLE IF NOT EXISTS chats (
    player_id   VARCHAR(36) NOT NULL REFERENCES players(id) ON DELETE CASCADE,
    persona_id  INTEGER     NOT NULL,
    messages    JSONB       NOT NULL,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (player_id, persona_id)
);

CREATE TABLE IF NOT EXISTS jobs (
    id           VARCHAR(36) PRIMARY KEY,
    plan_id      VARCHAR(36) NOT NULL REFERENCES plans(id) ON DELETE CASCADE,
    status       VARCHAR(16) NOT NULL DEFAULT 'queued',
    progress     DOUBLE PRECISION NOT NULL DEFAULT 0,
    message      TEXT        NOT NULL DEFAULT '',
    error        TEXT,
    started_at   TIMESTAMPTZ,
    finished_at  TIMESTAMPTZ,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_jobs_plan_id ON jobs (plan_id);

CREATE TABLE IF NOT EXISTS townhalls (
    plan_id     VARCHAR(36) PRIMARY KEY REFERENCES plans(id) ON DELETE CASCADE,
    speakers    JSONB       NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Reserved for real-household home claiming (spec 9.2, 13.4). Created empty; all nullable.
CREATE TABLE IF NOT EXISTS households_real (
    id               VARCHAR(36) PRIMARY KEY,
    player_id        VARCHAR(36) REFERENCES players(id) ON DELETE SET NULL,
    building_id      BIGINT,
    household_size   SMALLINT,
    vehicles         SMALLINT,
    n_kids           SMALLINT,
    school_ids       JSONB,
    work_area        JSONB,
    departure_times  JSONB,
    trip_data        JSONB,
    opt_in           JSONB,
    consent_at       TIMESTAMPTZ,
    created_at       TIMESTAMPTZ
);
