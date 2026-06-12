"""Pydantic schemas for AQI and forecast endpoints."""

from datetime import datetime

from pydantic import BaseModel


class AqiReadingOut(BaseModel):
    city: str
    timestamp: datetime
    pm25: float | None
    pm10: float | None
    no2: float | None
    so2: float | None
    co: float | None
    o3: float | None
    aqi_calculated: float | None
    aqi_category: str | None
    is_anomaly: bool
    source: str | None

    model_config = {"from_attributes": True}


class CurrentAqiOut(BaseModel):
    city: str
    aqi: float | None
    category: str | None
    pm25: float | None
    pm10: float | None
    no2: float | None
    updated_at: datetime | None
    source: str | None


class ForecastPoint(BaseModel):
    hour: int
    timestamp: datetime
    pm25_predicted: float
    lower_ci: float
    upper_ci: float
    aqi_category: str | None


class ForecastResponse(BaseModel):
    city: str
    generated_at: datetime
    points: list[ForecastPoint]


class AlertHistoryOut(BaseModel):
    id: int
    city: str
    aqi_level: str
    aqi_value: float | None
    message_en: str
    channel: str
    status: str
    sent_at: datetime

    model_config = {"from_attributes": True}
