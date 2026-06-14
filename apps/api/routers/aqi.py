"""AQI current readings, history, and forecast endpoints."""

import asyncio
import json
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.auth import get_current_user_id
from core.database import get_db, AsyncSessionLocal
from core.redis_client import cache_get, cache_set
from models.aqi import AqiReading, Forecast
from schemas.aqi import AqiReadingOut, CurrentAqiOut, ForecastPoint, ForecastResponse

router = APIRouter(prefix="/aqi", tags=["aqi"])

# Connected WebSocket clients — broadcast to all on AQI update
_ws_clients: set[WebSocket] = set()

PAKISTAN_CITIES = ["Islamabad", "Karachi", "Lahore", "Peshawar", "Quetta"]

# City coordinates for the map
CITY_COORDS = {
    "Islamabad": (33.6844, 73.0479),
    "Karachi": (24.8607, 67.0011),
    "Lahore": (31.5204, 74.3587),
    "Peshawar": (34.0151, 71.5249),
    "Quetta": (30.1798, 66.9750),
}


@router.get("/current", response_model=list[CurrentAqiOut])
async def get_current_aqi(db: AsyncSession = Depends(get_db)):
    """Return the most recent AQI reading for each city. Cached 5 minutes in Redis."""
    cache_key = "aqi:current:all"
    cached = await cache_get(cache_key)
    if cached:
        return cached

    results = []
    for city in PAKISTAN_CITIES:
        row = await db.execute(
            select(AqiReading)
            .where(AqiReading.city == city)
            .order_by(desc(AqiReading.timestamp))
            .limit(1)
        )
        reading = row.scalar_one_or_none()
        if reading:
            results.append(CurrentAqiOut(
                city=city,
                aqi=reading.aqi_calculated,
                category=reading.aqi_category,
                pm25=reading.pm25,
                pm10=reading.pm10,
                no2=reading.no2,
                updated_at=reading.timestamp,
                source=reading.source,
            ))
        else:
            results.append(CurrentAqiOut(
                city=city, aqi=None, category=None, pm25=None, pm10=None,
                no2=None, updated_at=None, source=None,
            ))

    serialized = [r.model_dump(mode="json") for r in results]
    await cache_set(cache_key, serialized, ttl_seconds=300)
    return results


@router.get("/{city}/history", response_model=list[AqiReadingOut])
async def get_city_history(
    city: str,
    days: int = 7,
    db: AsyncSession = Depends(get_db),
):
    """Return hourly AQI readings for the last N days (default 7). Auth not required."""
    if city not in PAKISTAN_CITIES:
        raise HTTPException(status_code=404, detail=f"Unknown city: {city}")
    if days > 30:
        days = 30

    since = datetime.now(timezone.utc) - timedelta(days=days)
    result = await db.execute(
        select(AqiReading)
        .where(AqiReading.city == city, AqiReading.timestamp >= since)
        .order_by(AqiReading.timestamp)
    )
    return result.scalars().all()


@router.get("/{city}/forecast", response_model=ForecastResponse)
async def get_city_forecast(city: str, db: AsyncSession = Depends(get_db)):
    """Return the 24-hour Prophet PM2.5 forecast for a city. Cached 1 hour."""
    if city not in PAKISTAN_CITIES:
        raise HTTPException(status_code=404, detail=f"Unknown city: {city}")

    cache_key = f"aqi:forecast:{city}"
    cached = await cache_get(cache_key)
    if cached:
        return cached

    # Check DB for recently generated forecast
    since = datetime.now(timezone.utc) - timedelta(hours=1)
    result = await db.execute(
        select(Forecast)
        .where(Forecast.city == city, Forecast.generated_at >= since)
        .order_by(Forecast.forecast_for)
    )
    rows = result.scalars().all()

    if not rows:
        # Trigger an on-demand forecast generation
        from services.forecast_service import generate_forecast
        try:
            rows = await generate_forecast(city, db)
        except Exception as exc:
            print(f"[forecast] generate_forecast failed for {city}: {exc}")
            rows = []

    if not rows:
        raise HTTPException(status_code=503, detail="Forecast not available yet — upload Prophet models to R2 to enable")

    points = [
        ForecastPoint(
            hour=i,
            timestamp=row.forecast_for,
            pm25_predicted=row.pm25_predicted,
            lower_ci=row.lower_ci,
            upper_ci=row.upper_ci,
            aqi_category=row.aqi_category,
        )
        for i, row in enumerate(rows)
    ]
    response = ForecastResponse(city=city, generated_at=rows[0].generated_at, points=points)
    await cache_set(cache_key, response.model_dump(mode="json"), ttl_seconds=3600)
    return response


@router.websocket("/ws")
async def aqi_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for real-time AQI updates.

    On connect: immediately sends current AQI for all 5 cities.
    Then pushes updates every 5 minutes (matching the REST cache TTL).
    Clients can also trigger an immediate refresh by sending any message.
    """
    await websocket.accept()
    _ws_clients.add(websocket)
    try:
        async def _send_current():
            async with AsyncSessionLocal() as db:
                results = []
                for city in PAKISTAN_CITIES:
                    row = await db.execute(
                        select(AqiReading)
                        .where(AqiReading.city == city)
                        .order_by(desc(AqiReading.timestamp))
                        .limit(1)
                    )
                    reading = row.scalar_one_or_none()
                    results.append({
                        "city": city,
                        "aqi": reading.aqi_calculated if reading else None,
                        "category": reading.aqi_category if reading else None,
                        "pm25": reading.pm25 if reading else None,
                        "updated_at": reading.timestamp.isoformat() if reading else None,
                    })
                await websocket.send_text(json.dumps({"type": "aqi_update", "data": results}))

        # Send initial data on connect
        await _send_current()

        # Keep alive — push updates every 5 minutes, or on client ping
        while True:
            try:
                # Wait for a client message or timeout after 300s
                await asyncio.wait_for(websocket.receive_text(), timeout=300)
            except asyncio.TimeoutError:
                pass
            await _send_current()

    except WebSocketDisconnect:
        pass
    finally:
        _ws_clients.discard(websocket)


async def broadcast_aqi_update(data: list[dict]) -> None:
    """
    Broadcast an AQI update to all connected WebSocket clients.

    Called by scrape_task after each successful scrape to push live data
    without waiting for the 5-minute poll cycle.
    """
    if not _ws_clients:
        return
    message = json.dumps({"type": "aqi_update", "data": data})
    dead: set[WebSocket] = set()
    for ws in _ws_clients:
        try:
            await ws.send_text(message)
        except Exception:
            dead.add(ws)
    _ws_clients -= dead
