-- SmogAlert PK — Initial Schema
-- Run this in: supabase.com/dashboard/project/qykraxdfjrpdrhkoiccd/editor
-- This is idempotent (CREATE TABLE IF NOT EXISTS) — safe to run multiple times.

CREATE TABLE IF NOT EXISTS users (
    id          SERIAL PRIMARY KEY,
    clerk_user_id VARCHAR NOT NULL UNIQUE,
    email       VARCHAR NOT NULL UNIQUE,
    name        VARCHAR,
    phone_whatsapp VARCHAR,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX IF NOT EXISTS ix_users_clerk_user_id ON users (clerk_user_id);

CREATE TABLE IF NOT EXISTS user_locations (
    id          SERIAL PRIMARY KEY,
    user_id     INTEGER NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    city        VARCHAR NOT NULL,
    latitude    FLOAT,
    longitude   FLOAT,
    is_primary  BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS subscriptions (
    id                  SERIAL PRIMARY KEY,
    user_id             INTEGER NOT NULL UNIQUE REFERENCES users (id) ON DELETE CASCADE,
    plan                VARCHAR NOT NULL DEFAULT 'free',
    status              VARCHAR NOT NULL DEFAULT 'active',
    stripe_customer_id  VARCHAR,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS alert_preferences (
    id              SERIAL PRIMARY KEY,
    user_id         INTEGER NOT NULL UNIQUE REFERENCES users (id) ON DELETE CASCADE,
    channels        VARCHAR NOT NULL DEFAULT 'email',
    aqi_threshold   VARCHAR NOT NULL DEFAULT 'Unhealthy',
    advance_hours   INTEGER NOT NULL DEFAULT 6,
    language        VARCHAR NOT NULL DEFAULT 'en'
);

CREATE TABLE IF NOT EXISTS alerts_sent (
    id          SERIAL PRIMARY KEY,
    user_id     INTEGER NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    city        VARCHAR NOT NULL,
    aqi_level   VARCHAR NOT NULL,
    aqi_value   FLOAT,
    message_en  TEXT NOT NULL,
    message_ur  TEXT NOT NULL,
    channel     VARCHAR NOT NULL,
    status      VARCHAR NOT NULL DEFAULT 'pending',
    sent_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS aqi_readings (
    id              SERIAL PRIMARY KEY,
    city            VARCHAR NOT NULL,
    timestamp       TIMESTAMPTZ NOT NULL,
    pm25            FLOAT,
    pm10            FLOAT,
    no2             FLOAT,
    so2             FLOAT,
    co              FLOAT,
    o3              FLOAT,
    nh3             FLOAT,
    aqi_calculated  FLOAT,
    aqi_category    VARCHAR,
    is_anomaly      BOOLEAN NOT NULL DEFAULT FALSE,
    source          VARCHAR,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_aqi_city_timestamp_source UNIQUE (city, timestamp, source)
);
CREATE INDEX IF NOT EXISTS ix_aqi_readings_city      ON aqi_readings (city);
CREATE INDEX IF NOT EXISTS ix_aqi_readings_timestamp ON aqi_readings (timestamp);

CREATE TABLE IF NOT EXISTS forecasts (
    id              SERIAL PRIMARY KEY,
    city            VARCHAR NOT NULL,
    forecast_for    TIMESTAMPTZ NOT NULL,
    pm25_predicted  FLOAT NOT NULL,
    lower_ci        FLOAT NOT NULL,
    upper_ci        FLOAT NOT NULL,
    aqi_category    VARCHAR,
    generated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_forecasts_city ON forecasts (city);

CREATE TABLE IF NOT EXISTS model_versions (
    id          SERIAL PRIMARY KEY,
    model_type  VARCHAR NOT NULL,
    city        VARCHAR,
    mae         FLOAT,
    r2_score    FLOAT,
    trained_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    promoted    BOOLEAN NOT NULL DEFAULT FALSE,
    r2_key      VARCHAR
);

-- Alembic version table so alembic upgrade head is a no-op once we switch to Railway
CREATE TABLE IF NOT EXISTS alembic_version (
    version_num VARCHAR(32) NOT NULL,
    CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num)
);
INSERT INTO alembic_version (version_num) VALUES ('001')
ON CONFLICT DO NOTHING;
