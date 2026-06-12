"""Alembic migration environment — async SQLAlchemy setup."""

import asyncio
import os
from logging.config import fileConfig
from pathlib import Path

# Load .env from repo root so DATABASE_URL is available
from dotenv import load_dotenv
load_dotenv(Path(__file__).parents[3] / ".env")

from alembic import context
from sqlalchemy.ext.asyncio import async_engine_from_config
from sqlalchemy import pool

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Override the URL from environment variable (Railway/Supabase)
database_url = os.getenv("DATABASE_URL", config.get_main_option("sqlalchemy.url"))
# Alembic needs sync driver for migrations; swap asyncpg → psycopg2
sync_url = database_url.replace("postgresql+asyncpg://", "postgresql+psycopg2://")
config.set_main_option("sqlalchemy.url", sync_url)

# Import all models so Alembic can detect them
from core.database import Base
from models.user import User, UserLocation, Subscription, AlertPreferences, AlertSent  # noqa
from models.aqi import AqiReading, Forecast, ModelVersion  # noqa

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(url=url, target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    from sqlalchemy import create_engine
    connectable = create_engine(config.get_main_option("sqlalchemy.url"), poolclass=pool.NullPool)
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
