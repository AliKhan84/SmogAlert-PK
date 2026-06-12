"""AQI data, forecast, and model version ORM models."""

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from core.database import Base


class AqiReading(Base):
    __tablename__ = "aqi_readings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    city: Mapped[str] = mapped_column(String, nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    pm25: Mapped[float | None] = mapped_column(Float)
    pm10: Mapped[float | None] = mapped_column(Float)
    no2: Mapped[float | None] = mapped_column(Float)
    so2: Mapped[float | None] = mapped_column(Float)
    co: Mapped[float | None] = mapped_column(Float)
    o3: Mapped[float | None] = mapped_column(Float)
    nh3: Mapped[float | None] = mapped_column(Float)
    aqi_calculated: Mapped[float | None] = mapped_column(Float)
    aqi_category: Mapped[str | None] = mapped_column(String)
    is_anomaly: Mapped[bool] = mapped_column(Boolean, default=False)
    source: Mapped[str | None] = mapped_column(String)  # "openaq" | "waqi" | "openmeteo" | "kaggle"
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Forecast(Base):
    __tablename__ = "forecasts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    city: Mapped[str] = mapped_column(String, nullable=False, index=True)
    forecast_for: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    pm25_predicted: Mapped[float] = mapped_column(Float)
    lower_ci: Mapped[float] = mapped_column(Float)
    upper_ci: Mapped[float] = mapped_column(Float)
    aqi_category: Mapped[str | None] = mapped_column(String)
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_type: Mapped[str] = mapped_column(String)  # "prophet" | "isolation_forest" | "random_forest"
    city: Mapped[str | None] = mapped_column(String)
    mae: Mapped[float | None] = mapped_column(Float)
    r2_score: Mapped[float | None] = mapped_column(Float)
    trained_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    promoted: Mapped[bool] = mapped_column(Boolean, default=False)
    r2_key: Mapped[str | None] = mapped_column(String)  # Cloudflare R2 object key
