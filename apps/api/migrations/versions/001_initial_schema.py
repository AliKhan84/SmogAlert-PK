"""Initial database schema — all tables.

Revision ID: 001
Revises: —
Create Date: 2026-06-12
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers used by Alembic
revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── users ─────────────────────────────────────────────────────────────────
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("clerk_user_id", sa.String(), unique=True, nullable=False),
        sa.Column("email", sa.String(), unique=True, nullable=False),
        sa.Column("name", sa.String(), nullable=True),
        sa.Column("phone_whatsapp", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )
    op.create_index("ix_users_clerk_user_id", "users", ["clerk_user_id"])

    # ── user_locations ────────────────────────────────────────────────────────
    op.create_table(
        "user_locations",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("city", sa.String(), nullable=False),
        sa.Column("latitude", sa.Float(), nullable=True),
        sa.Column("longitude", sa.Float(), nullable=True),
        sa.Column("is_primary", sa.Boolean(), default=False),
    )

    # ── subscriptions ─────────────────────────────────────────────────────────
    op.create_table(
        "subscriptions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), unique=True, nullable=False),
        sa.Column("plan", sa.String(), default="free"),
        sa.Column("status", sa.String(), default="active"),
        sa.Column("stripe_customer_id", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )

    # ── alert_preferences ─────────────────────────────────────────────────────
    op.create_table(
        "alert_preferences",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), unique=True, nullable=False),
        sa.Column("channels", sa.String(), default="email"),
        sa.Column("aqi_threshold", sa.String(), default="Unhealthy"),
        sa.Column("advance_hours", sa.Integer(), default=6),
        sa.Column("language", sa.String(), default="en"),
    )

    # ── alerts_sent ───────────────────────────────────────────────────────────
    op.create_table(
        "alerts_sent",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("city", sa.String(), nullable=False),
        sa.Column("aqi_level", sa.String(), nullable=False),
        sa.Column("aqi_value", sa.Float(), nullable=True),
        sa.Column("message_en", sa.String(), nullable=False),
        sa.Column("message_ur", sa.String(), nullable=False),
        sa.Column("channel", sa.String(), nullable=False),
        sa.Column("status", sa.String(), default="pending"),
        sa.Column("sent_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )

    # ── aqi_readings ──────────────────────────────────────────────────────────
    op.create_table(
        "aqi_readings",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("city", sa.String(), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("pm25", sa.Float(), nullable=True),
        sa.Column("pm10", sa.Float(), nullable=True),
        sa.Column("no2", sa.Float(), nullable=True),
        sa.Column("so2", sa.Float(), nullable=True),
        sa.Column("co", sa.Float(), nullable=True),
        sa.Column("o3", sa.Float(), nullable=True),
        sa.Column("nh3", sa.Float(), nullable=True),
        sa.Column("aqi_calculated", sa.Float(), nullable=True),
        sa.Column("aqi_category", sa.String(), nullable=True),
        sa.Column("is_anomaly", sa.Boolean(), default=False),
        sa.Column("source", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )
    op.create_index("ix_aqi_readings_city", "aqi_readings", ["city"])
    op.create_index("ix_aqi_readings_timestamp", "aqi_readings", ["timestamp"])
    # Unique constraint to prevent duplicate readings from the same source
    op.create_unique_constraint("uq_aqi_city_timestamp_source", "aqi_readings", ["city", "timestamp", "source"])

    # ── forecasts ─────────────────────────────────────────────────────────────
    op.create_table(
        "forecasts",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("city", sa.String(), nullable=False),
        sa.Column("forecast_for", sa.DateTime(timezone=True), nullable=False),
        sa.Column("pm25_predicted", sa.Float(), nullable=False),
        sa.Column("lower_ci", sa.Float(), nullable=False),
        sa.Column("upper_ci", sa.Float(), nullable=False),
        sa.Column("aqi_category", sa.String(), nullable=True),
        sa.Column("generated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )
    op.create_index("ix_forecasts_city", "forecasts", ["city"])

    # ── model_versions ────────────────────────────────────────────────────────
    op.create_table(
        "model_versions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("model_type", sa.String(), nullable=False),
        sa.Column("city", sa.String(), nullable=True),
        sa.Column("mae", sa.Float(), nullable=True),
        sa.Column("r2_score", sa.Float(), nullable=True),
        sa.Column("trained_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("promoted", sa.Boolean(), default=False),
        sa.Column("r2_key", sa.String(), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("model_versions")
    op.drop_table("forecasts")
    op.drop_table("aqi_readings")
    op.drop_table("alerts_sent")
    op.drop_table("alert_preferences")
    op.drop_table("subscriptions")
    op.drop_table("user_locations")
    op.drop_table("users")
