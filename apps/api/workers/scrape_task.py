"""Celery task: scrape live AQI data from OpenAQ, WAQI, and OpenMeteo."""

import asyncio
from datetime import datetime, timezone

from workers.celery_app import celery_app


def _aqi_to_category(aqi: float | None) -> str | None:
    """Convert a US EPA AQI integer value to a category label."""
    if aqi is None:
        return None
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"


def _pm25_to_aqi(pm25: float | None) -> float | None:
    """Convert PM2.5 µg/m³ to US EPA AQI using linear interpolation between breakpoints."""
    if pm25 is None:
        return None
    # (pm25_low, pm25_high, aqi_low, aqi_high)
    breakpoints = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 500.4, 301, 500),
    ]
    for bp_lo, bp_hi, aqi_lo, aqi_hi in breakpoints:
        if pm25 <= bp_hi:
            return round(((aqi_hi - aqi_lo) / (bp_hi - bp_lo)) * (pm25 - bp_lo) + aqi_lo)
    return 500.0

PAKISTAN_CITIES = ["Islamabad", "Karachi", "Lahore", "Peshawar", "Quetta"]

# WAQI city slugs
WAQI_CITY_SLUGS = {
    "Islamabad": "islamabad",
    "Karachi": "karachi",
    "Lahore": "lahore",
    "Peshawar": "peshawar",
    "Quetta": "quetta",
}

# OpenMeteo coordinates
CITY_COORDS = {
    "Islamabad": (33.6844, 73.0479),
    "Karachi": (24.8607, 67.0011),
    "Lahore": (31.5204, 74.3587),
    "Peshawar": (34.0151, 71.5249),
    "Quetta": (30.1798, 66.9750),
}


@celery_app.task(name="workers.scrape_task.scrape_all_cities", bind=True, max_retries=3)
def scrape_all_cities(self):
    """Main hourly scrape task — tries all sources and stores results."""
    asyncio.run(_scrape_all_async())


async def _scrape_all_async():
    import httpx
    from core.config import settings
    from core.database import AsyncSessionLocal, engine
    from models.aqi import AqiReading
    from sqlalchemy import select

    # Dispose the connection pool so asyncpg creates fresh connections for this
    # event loop (Celery's asyncio.run() creates a new loop each task invocation).
    await engine.dispose()

    readings = []

    async with httpx.AsyncClient(timeout=30) as client:
        # --- WAQI (covers all 5 cities) ---
        if settings.waqi_token:
            for city, slug in WAQI_CITY_SLUGS.items():
                try:
                    resp = await client.get(
                        f"https://api.waqi.info/feed/{slug}/",
                        params={"token": settings.waqi_token},
                    )
                    data = resp.json()
                    if data.get("status") == "ok":
                        d = data["data"]
                        iaqi = d.get("iaqi", {})
                        aqi_val = d.get("aqi")
                        readings.append({
                            "city": city,
                            "timestamp": datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0),
                            "pm25": iaqi.get("pm25", {}).get("v"),
                            "pm10": iaqi.get("pm10", {}).get("v"),
                            "no2": iaqi.get("no2", {}).get("v"),
                            "so2": iaqi.get("so2", {}).get("v"),
                            "co": iaqi.get("co", {}).get("v"),
                            "o3": iaqi.get("o3", {}).get("v"),
                            "aqi_calculated": aqi_val,
                            "aqi_category": _aqi_to_category(aqi_val),
                            "source": "waqi",
                        })
                except Exception as exc:
                    print(f"[scrape] WAQI failed for {city}: {exc}")

        # --- OpenMeteo Air Quality (fallback for missing cities) ---
        covered = {r["city"] for r in readings}
        for city in PAKISTAN_CITIES:
            if city in covered:
                continue
            lat, lon = CITY_COORDS[city]
            try:
                resp = await client.get(
                    "https://air-quality-api.open-meteo.com/v1/air-quality",
                    params={
                        "latitude": lat,
                        "longitude": lon,
                        "hourly": "pm2_5,pm10,nitrogen_dioxide,sulphur_dioxide,carbon_monoxide,ozone",
                        "timezone": "Asia/Karachi",
                        "past_hours": 1,
                        "forecast_hours": 0,
                    },
                )
                d = resp.json()
                hourly = d.get("hourly", {})
                if hourly.get("pm2_5"):
                    readings.append({
                        "city": city,
                        "timestamp": datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0),
                        "pm25": hourly["pm2_5"][-1] if hourly["pm2_5"] else None,
                        "pm10": hourly["pm10"][-1] if hourly.get("pm10") else None,
                        "no2": hourly["nitrogen_dioxide"][-1] if hourly.get("nitrogen_dioxide") else None,
                        "so2": hourly["sulphur_dioxide"][-1] if hourly.get("sulphur_dioxide") else None,
                        "co": hourly["carbon_monoxide"][-1] if hourly.get("carbon_monoxide") else None,
                        "o3": hourly["ozone"][-1] if hourly.get("ozone") else None,
                        "aqi_calculated": None,
                        "aqi_category": None,
                        "source": "openmeteo",
                    })
            except Exception as exc:
                print(f"[scrape] OpenMeteo failed for {city}: {exc}")

    # Persist to DB (upsert by city + hour).
    # Two-pass approach: all SELECTs first, then all INSERTs.
    # Mixing db.add() and await db.execute() in the same loop triggers SQLAlchemy
    # autoflush mid-loop, which causes asyncpg "another operation in progress" errors.
    async with AsyncSessionLocal() as db:
        to_insert = []
        for r in readings:
            result = await db.execute(
                select(AqiReading.id).where(
                    AqiReading.city == r["city"],
                    AqiReading.timestamp == r["timestamp"],
                )
            )
            if result.scalar_one_or_none() is None:
                to_insert.append(r)

        for r in to_insert:
            db.add(AqiReading(**{k: v for k, v in r.items() if v is not None or k in ("city", "timestamp")}))
        await db.commit()

    print(f"[scrape] Stored {len(readings)} readings at {datetime.now(timezone.utc)}")
