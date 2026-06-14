"""User profile and preferences endpoints."""

import json

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from core.auth import get_current_user_id
from core.database import get_db
from models.user import AlertPreferences, User, UserLocation
from schemas.user import AlertPrefsOut, AlertPrefsUpdate, ClerkWebhookPayload, LocationOut, UserOut, UserUpdate

router = APIRouter(prefix="/users", tags=["users"])


async def _get_or_404(db: AsyncSession, clerk_user_id: str) -> User:
    result = await db.execute(
        select(User).where(User.clerk_user_id == clerk_user_id).options(selectinload(User.locations))
    )
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user


async def _get_or_create(db: AsyncSession, clerk_user_id: str) -> User:
    """Return user, creating a minimal record if the webhook hasn't fired yet."""
    from models.user import AlertPreferences, Subscription
    result = await db.execute(
        select(User).where(User.clerk_user_id == clerk_user_id).options(selectinload(User.locations))
    )
    user = result.scalar_one_or_none()
    if user is None:
        user = User(clerk_user_id=clerk_user_id, email=f"{clerk_user_id}@clerk.user")
        db.add(user)
        await db.flush()
        db.add(Subscription(user_id=user.id, plan="free"))
        db.add(AlertPreferences(user_id=user.id))
        await db.commit()
        await db.refresh(user)
    return user


@router.get("/me", response_model=UserOut)
async def get_me(
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    return await _get_or_create(db, user_id)


@router.put("/me", response_model=UserOut)
async def update_me(
    body: UserUpdate,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    user = await _get_or_404(db, user_id)
    if body.name is not None:
        user.name = body.name
    if body.phone_whatsapp is not None:
        user.phone_whatsapp = body.phone_whatsapp
    await db.commit()
    await db.refresh(user)
    return user


@router.get("/me/preferences", response_model=AlertPrefsOut)
async def get_preferences(
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    user = await _get_or_404(db, user_id)
    result = await db.execute(select(AlertPreferences).where(AlertPreferences.user_id == user.id))
    prefs = result.scalar_one_or_none()
    if prefs is None:
        # Return defaults
        return AlertPrefsOut(channels="email", aqi_threshold="Unhealthy", advance_hours=6, language="en")
    return prefs


@router.put("/me/preferences", response_model=AlertPrefsOut)
async def update_preferences(
    body: AlertPrefsUpdate,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    user = await _get_or_create(db, user_id)
    result = await db.execute(select(AlertPreferences).where(AlertPreferences.user_id == user.id))
    prefs = result.scalar_one_or_none()

    if prefs is None:
        prefs = AlertPreferences(user_id=user.id)
        db.add(prefs)

    if body.channels is not None:
        prefs.channels = body.channels
    if body.aqi_threshold is not None:
        prefs.aqi_threshold = body.aqi_threshold
    if body.advance_hours is not None:
        prefs.advance_hours = body.advance_hours
    if body.language is not None:
        prefs.language = body.language

    await db.commit()
    await db.refresh(prefs)
    return prefs


@router.post("/me/locations", response_model=LocationOut, status_code=status.HTTP_201_CREATED)
async def add_location(
    body: LocationOut,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    user = await _get_or_create(db, user_id)
    loc = UserLocation(
        user_id=user.id,
        city=body.city,
        latitude=body.latitude,
        longitude=body.longitude,
        is_primary=body.is_primary,
    )
    db.add(loc)
    await db.commit()
    await db.refresh(loc)
    return loc


@router.delete("/me/locations/{location_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_location(
    location_id: int,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    user = await _get_or_404(db, user_id)
    result = await db.execute(
        select(UserLocation).where(UserLocation.id == location_id, UserLocation.user_id == user.id)
    )
    loc = result.scalar_one_or_none()
    if loc is None:
        raise HTTPException(status_code=404, detail="Location not found")
    await db.delete(loc)
    await db.commit()
