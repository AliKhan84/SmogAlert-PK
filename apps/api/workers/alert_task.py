"""
Celery task: hourly alert engine.

For each city, checks the 24-hour forecast. If any hour crosses a user's AQI threshold,
sends an alert — with Redis dedup to avoid spamming (one alert per user per city per 6 hours).
"""

import asyncio
from datetime import datetime, timedelta, timezone

from workers.celery_app import celery_app

AQI_THRESHOLD_ORDER = ["Moderate", "Unhealthy for Sensitive Groups", "Unhealthy", "Very Unhealthy", "Hazardous"]

URDU_CITY = {
    "Islamabad": "اسلام آباد",
    "Karachi": "کراچی",
    "Lahore": "لاہور",
    "Peshawar": "پشاور",
    "Quetta": "کوئٹہ",
}


def _threshold_met(predicted_category: str, user_threshold: str) -> bool:
    """Return True if predicted_category is at or above user_threshold."""
    try:
        return AQI_THRESHOLD_ORDER.index(predicted_category) >= AQI_THRESHOLD_ORDER.index(user_threshold)
    except ValueError:
        return False


def _build_alert_messages(city: str, aqi_level: str, hours_ahead: int, pm25: float) -> tuple[str, str]:
    en = (
        f"Air quality in {city} is forecast to reach {aqi_level} (PM2.5: {pm25:.0f} µg/m³) "
        f"within {hours_ahead} hours. Wear an N95 mask and limit outdoor activity."
    )
    ur_city = URDU_CITY.get(city, city)
    ur = (
        f"{ur_city} میں فضائی آلودگی اگلے {hours_ahead} گھنٹوں میں {aqi_level} سطح تک پہنچنے کی پیش گوئی ہے "
        f"(PM2.5: {pm25:.0f})۔ N95 ماسک پہنیں اور باہر نہ جائیں۔"
    )
    return en, ur


@celery_app.task(name="workers.alert_task.run_alert_engine", bind=True)
def run_alert_engine(self):
    asyncio.run(_run_alert_engine_async())


async def _run_alert_engine_async():
    from core.database import AsyncSessionLocal, engine
    from core.redis_client import cache_get, cache_set
    from models.aqi import Forecast
    from models.user import AlertPreferences, User, UserLocation
    from services.alert_dispatcher import dispatch_alert
    from sqlalchemy import select, and_

    # asyncpg connections are bound to a specific event loop. Celery uses asyncio.run()
    # per task which creates a NEW event loop each time. Disposing the pool forces
    # SQLAlchemy to create fresh connections for the current event loop.
    await engine.dispose()

    PAKISTAN_CITIES = ["Islamabad", "Karachi", "Lahore", "Peshawar", "Quetta"]
    now = datetime.now(timezone.utc)

    # Use a fresh session per city — avoids asyncpg "another operation in progress"
    # errors that occur when db.commit() inside dispatch_alert leaves the shared
    # session in an unclean state for the next city's queries.
    for city in PAKISTAN_CITIES:
        async with AsyncSessionLocal() as db:
            since = now - timedelta(minutes=30)
            until = now + timedelta(hours=24)
            result = await db.execute(
                select(Forecast).where(
                    and_(Forecast.city == city, Forecast.forecast_for >= since, Forecast.forecast_for <= until)
                ).order_by(Forecast.forecast_for)
            )
            forecast_rows = result.scalars().all()
            if not forecast_rows:
                print(f"[alert_engine] No forecast data for {city} — skipping")
                continue

            user_result = await db.execute(
                select(User)
                .join(UserLocation, UserLocation.user_id == User.id)
                .where(UserLocation.city == city)
            )
            users = user_result.scalars().unique().all()

            for user in users:
                prefs_result = await db.execute(
                    select(AlertPreferences).where(AlertPreferences.user_id == user.id)
                )
                prefs = prefs_result.scalar_one_or_none()
                threshold = prefs.aqi_threshold if prefs else "Unhealthy"
                channels = prefs.channels if prefs else "email"

                dedup_key = f"alert_sent:{user.id}:{city}"
                already_alerted = await cache_get(dedup_key)
                if already_alerted:
                    continue

                triggering_point = None
                for point in forecast_rows:
                    if point.aqi_category and _threshold_met(point.aqi_category, threshold):
                        triggering_point = point
                        break

                if triggering_point is None:
                    continue

                hours_ahead = max(1, int((triggering_point.forecast_for - now).total_seconds() / 3600))
                msg_en, msg_ur = _build_alert_messages(
                    city, triggering_point.aqi_category, hours_ahead, triggering_point.pm25_predicted
                )

                # dispatch_alert gets its own session internally — do not pass db
                await dispatch_alert(
                    user=user,
                    city=city,
                    aqi_level=triggering_point.aqi_category,
                    aqi_value=triggering_point.pm25_predicted,
                    message_en=msg_en,
                    message_ur=msg_ur,
                    channels=channels,
                )

                await cache_set(dedup_key, True, ttl_seconds=6 * 3600)
                print(f"[alert_engine] Alert sent to user {user.id} for {city} ({triggering_point.aqi_category})")
