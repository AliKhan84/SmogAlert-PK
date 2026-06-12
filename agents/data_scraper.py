"""
Autonomous data scraping agent — pulls live AQI data from multiple sources
and stores it in the PostgreSQL database.

Sources:
  1. WAQI (aqicn.org) — primary, covers all 5 PK cities
  2. OpenAQ v3     — secondary, covers Lahore/Karachi/Islamabad
  3. OpenMeteo     — fallback for any city not covered above

Run manually:
  python agents/data_scraper.py

Or scheduled by Celery Beat via workers/scrape_task.py.
"""

import asyncio
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx

# Allow imports from apps/api
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "apps" / "api"))

from core.config import settings
from core.database import AsyncSessionLocal
from models.aqi import AqiReading
from sqlalchemy import select


PAKISTAN_CITIES = ["Islamabad", "Karachi", "Lahore", "Peshawar", "Quetta"]

WAQI_SLUGS = {
    "Islamabad": "islamabad",
    "Karachi": "karachi",
    "Lahore": "lahore",
    "Peshawar": "peshawar",
    "Quetta": "quetta",
}

CITY_COORDS = {
    "Islamabad": (33.6844, 73.0479),
    "Karachi": (24.8607, 67.0011),
    "Lahore": (31.5204, 74.3587),
    "Peshawar": (34.0151, 71.5249),
    "Quetta": (30.1798, 66.9750),
}

# OpenAQ location IDs for Pakistani cities (from OpenAQ v3 API)
OPENAQ_LOCATION_IDS = {
    "Lahore": 8118,
    "Karachi": 8120,
    "Islamabad": 8119,
}


async def fetch_waqi(client: httpx.AsyncClient) -> list[dict]:
    """Fetch current AQI from WAQI for all 5 cities."""
    readings = []
    if not settings.waqi_token:
        print("[scraper] WAQI_TOKEN not set, skipping WAQI")
        return readings

    for city, slug in WAQI_SLUGS.items():
        try:
            resp = await client.get(
                f"https://api.waqi.info/feed/{slug}/",
                params={"token": settings.waqi_token},
            )
            data = resp.json()
            if data.get("status") != "ok":
                continue

            d = data["data"]
            iaqi = d.get("iaqi", {})
            hour = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
            readings.append({
                "city": city,
                "timestamp": hour,
                "pm25": iaqi.get("pm25", {}).get("v"),
                "pm10": iaqi.get("pm10", {}).get("v"),
                "no2": iaqi.get("no2", {}).get("v"),
                "so2": iaqi.get("so2", {}).get("v"),
                "co": iaqi.get("co", {}).get("v"),
                "o3": iaqi.get("o3", {}).get("v"),
                "aqi_calculated": float(d["aqi"]) if str(d.get("aqi", "")).lstrip("-").isdigit() else None,
                "source": "waqi",
            })
            print(f"[scraper] WAQI OK: {city} AQI={d.get('aqi')}")
        except Exception as exc:
            print(f"[scraper] WAQI error for {city}: {exc}")

    return readings


async def fetch_openaq(client: httpx.AsyncClient) -> list[dict]:
    """Fetch last reading from OpenAQ v3 for Lahore, Karachi, Islamabad."""
    readings = []
    headers = {}
    if settings.openaq_api_key:
        headers["X-API-Key"] = settings.openaq_api_key

    for city, location_id in OPENAQ_LOCATION_IDS.items():
        try:
            resp = await client.get(
                f"https://api.openaq.org/v3/locations/{location_id}/measurements",
                params={"limit": 8, "order_by": "datetime", "sort_order": "desc"},
                headers=headers,
            )
            data = resp.json()
            results = data.get("results", [])
            if not results:
                continue

            # Aggregate latest measurements into one reading
            reading: dict = {"city": city, "source": "openaq",
                             "timestamp": datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)}
            param_map = {"pm25": "pm25", "pm10": "pm10", "no2": "no2", "so2": "so2", "co": "co", "o3": "o3"}
            for m in results:
                param = m.get("parameter", {}).get("name", "")
                if param in param_map:
                    reading[param_map[param]] = m.get("value")

            readings.append(reading)
            print(f"[scraper] OpenAQ OK: {city} pm25={reading.get('pm25')}")
        except Exception as exc:
            print(f"[scraper] OpenAQ error for {city}: {exc}")

    return readings


async def fetch_openmeteo(client: httpx.AsyncClient, cities: list[str]) -> list[dict]:
    """Fetch last hour's air quality from OpenMeteo for given cities (fallback)."""
    readings = []
    for city in cities:
        lat, lon = CITY_COORDS[city]
        try:
            resp = await client.get(
                "https://air-quality-api.open-meteo.com/v1/air-quality",
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "hourly": "pm2_5,pm10,nitrogen_dioxide,sulphur_dioxide,carbon_monoxide,ozone",
                    "timezone": "Asia/Karachi",
                    "past_hours": 2,
                    "forecast_hours": 0,
                },
            )
            d = resp.json()
            hourly = d.get("hourly", {})
            pm25_vals = hourly.get("pm2_5", [])
            if not pm25_vals:
                continue

            hour = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
            readings.append({
                "city": city,
                "timestamp": hour,
                "pm25": pm25_vals[-1] if pm25_vals else None,
                "pm10": hourly.get("pm10", [None])[-1],
                "no2": hourly.get("nitrogen_dioxide", [None])[-1],
                "so2": hourly.get("sulphur_dioxide", [None])[-1],
                "co": hourly.get("carbon_monoxide", [None])[-1],
                "o3": hourly.get("ozone", [None])[-1],
                "source": "openmeteo",
            })
            print(f"[scraper] OpenMeteo OK: {city} pm25={pm25_vals[-1]}")
        except Exception as exc:
            print(f"[scraper] OpenMeteo error for {city}: {exc}")

    return readings


async def _normalise_and_store(all_readings: list[dict]) -> None:
    """Dedup by (city, timestamp) and upsert into aqi_readings table."""
    async with AsyncSessionLocal() as db:
        stored = 0
        for r in all_readings:
            # Skip if we already have a row for this city+hour
            existing = await db.execute(
                select(AqiReading).where(
                    AqiReading.city == r["city"],
                    AqiReading.timestamp == r["timestamp"],
                )
            )
            if existing.scalar_one_or_none() is not None:
                continue

            # Filter out None values that aren't city/timestamp
            clean = {k: v for k, v in r.items() if v is not None or k in ("city", "timestamp")}
            db.add(AqiReading(**clean))
            stored += 1

        await db.commit()
        print(f"[scraper] Stored {stored} new readings ({len(all_readings)} total fetched)")


async def run():
    """Main entry point: fetch from all sources and store."""
    async with httpx.AsyncClient(timeout=30) as client:
        waqi_readings = await fetch_waqi(client)
        covered = {r["city"] for r in waqi_readings}

        openaq_readings = await fetch_openaq(client)
        covered.update(r["city"] for r in openaq_readings)

        # OpenMeteo for any city not covered
        missing = [c for c in PAKISTAN_CITIES if c not in covered]
        fallback_readings = await fetch_openmeteo(client, missing) if missing else []

    all_readings = waqi_readings + openaq_readings + fallback_readings

    # Prefer WAQI/OpenAQ over OpenMeteo for same city+timestamp
    by_key: dict[tuple, dict] = {}
    for r in all_readings:
        key = (r["city"], r["timestamp"])
        existing = by_key.get(key)
        if existing is None or r["source"] in ("waqi", "openaq"):
            by_key[key] = r

    await _normalise_and_store(list(by_key.values()))


if __name__ == "__main__":
    asyncio.run(run())
