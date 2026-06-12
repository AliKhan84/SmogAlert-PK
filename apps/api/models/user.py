"""User-related SQLAlchemy ORM models."""

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from core.database import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    clerk_user_id: Mapped[str] = mapped_column(String, unique=True, nullable=False, index=True)
    email: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    name: Mapped[str | None] = mapped_column(String)
    phone_whatsapp: Mapped[str | None] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    locations: Mapped[list["UserLocation"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    subscription: Mapped["Subscription | None"] = relationship(back_populates="user", uselist=False)
    alert_preferences: Mapped["AlertPreferences | None"] = relationship(back_populates="user", uselist=False)
    alerts_sent: Mapped[list["AlertSent"]] = relationship(back_populates="user", cascade="all, delete-orphan")


class UserLocation(Base):
    __tablename__ = "user_locations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    city: Mapped[str] = mapped_column(String, nullable=False)
    latitude: Mapped[float | None] = mapped_column(Float)
    longitude: Mapped[float | None] = mapped_column(Float)
    is_primary: Mapped[bool] = mapped_column(Boolean, default=False)

    user: Mapped["User"] = relationship(back_populates="locations")


class Subscription(Base):
    __tablename__ = "subscriptions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), unique=True, nullable=False)
    plan: Mapped[str] = mapped_column(String, default="free")  # "free" | "pro"
    status: Mapped[str] = mapped_column(String, default="active")
    stripe_customer_id: Mapped[str | None] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    user: Mapped["User"] = relationship(back_populates="subscription")


class AlertPreferences(Base):
    __tablename__ = "alert_preferences"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), unique=True, nullable=False)
    # Channels stored as comma-separated string: "email,whatsapp,sms"
    channels: Mapped[str] = mapped_column(String, default="email")
    aqi_threshold: Mapped[str] = mapped_column(String, default="Unhealthy")  # Moderate/Unhealthy/VeryUnhealthy/Hazardous
    advance_hours: Mapped[int] = mapped_column(Integer, default=6)
    language: Mapped[str] = mapped_column(String, default="en")  # "en" | "ur"

    user: Mapped["User"] = relationship(back_populates="alert_preferences")


class AlertSent(Base):
    __tablename__ = "alerts_sent"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    city: Mapped[str] = mapped_column(String, nullable=False)
    aqi_level: Mapped[str] = mapped_column(String)
    aqi_value: Mapped[float | None] = mapped_column(Float)
    message_en: Mapped[str] = mapped_column(String)
    message_ur: Mapped[str] = mapped_column(String)
    channel: Mapped[str] = mapped_column(String)  # "email" | "whatsapp" | "sms"
    status: Mapped[str] = mapped_column(String, default="pending")  # "delivered" | "failed" | "pending"
    sent_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    user: Mapped["User"] = relationship(back_populates="alerts_sent")
