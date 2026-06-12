"""Pydantic response/request schemas for user endpoints."""

from datetime import datetime

from pydantic import BaseModel, EmailStr


class LocationOut(BaseModel):
    id: int
    city: str
    latitude: float | None
    longitude: float | None
    is_primary: bool

    model_config = {"from_attributes": True}


class UserOut(BaseModel):
    id: int
    clerk_user_id: str
    email: str
    name: str | None
    phone_whatsapp: str | None
    created_at: datetime
    locations: list[LocationOut] = []

    model_config = {"from_attributes": True}


class UserUpdate(BaseModel):
    name: str | None = None
    phone_whatsapp: str | None = None


class AlertPrefsOut(BaseModel):
    channels: str
    aqi_threshold: str
    advance_hours: int
    language: str

    model_config = {"from_attributes": True}


class AlertPrefsUpdate(BaseModel):
    channels: str | None = None
    aqi_threshold: str | None = None
    advance_hours: int | None = None
    language: str | None = None


class ClerkWebhookPayload(BaseModel):
    type: str
    data: dict
