"""
Load city Prophet models and generate 24-hour PM2.5 forecasts.

Model loading delegates to services/model_loader.py which handles:
- In-memory caching (1-hour TTL)
- Local disk fallback  →  /tmp/prophet_models/prophet_{city}.pkl
- Cloudflare R2 download if not found locally

If the Prophet model is unavailable or fails to load (e.g. pkl trained with an
older pandas version), we fall back to a statistical forecast derived from each
city's historical hour-of-day PM2.5 patterns stored in the aqi_readings table.
"""

from datetime import datetime, timedelta, timezone

import pandas as pd
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from models.aqi import AqiReading, Forecast


def _pm25_to_aqi_category(pm25: float) -> str:
    """Convert a PM2.5 µg/m³ value to a US EPA AQI category label."""
    if pm25 < 12:
        return "Good"
    elif pm25 < 35.5:
        return "Moderate"
    elif pm25 < 55.5:
        return "Unhealthy for Sensitive Groups"
    elif pm25 < 150.5:
        return "Unhealthy"
    elif pm25 < 250.5:
        return "Very Unhealthy"
    else:
        return "Hazardous"


async def _statistical_forecast(city: str, db: AsyncSession) -> list[Forecast]:
    """
    Fallback 24-hour forecast using hour-of-day PM2.5 averages from the DB.

    Steps:
    1. Pull the most recent 720 hourly readings (≈ 30 days) for the city.
    2. Group by hour-of-day and compute mean PM2.5 (seasonal day-cycle pattern).
    3. For any missing hours, use the city-wide average.
    4. Persist 24 Forecast rows starting from the next full UTC hour.
    """
    result = await db.execute(
        select(AqiReading)
        .where(AqiReading.city == city, AqiReading.pm25.is_not(None))
        .order_by(desc(AqiReading.timestamp))
        .limit(720)
    )
    readings = result.scalars().all()

    if not readings:
        return []

    # Build hour-of-day average
    hour_sums: dict[int, list[float]] = {}
    for r in readings:
        h = r.timestamp.hour
        hour_sums.setdefault(h, []).append(r.pm25)

    hour_means: dict[int, float] = {h: sum(v) / len(v) for h, v in hour_sums.items()}
    all_pm25 = [r.pm25 for r in readings]
    global_mean = sum(all_pm25) / len(all_pm25)

    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    rows: list[Forecast] = []
    for i in range(1, 25):
        ts = now + timedelta(hours=i)
        pm25 = round(hour_means.get(ts.hour, global_mean), 2)
        lower = round(max(0.0, pm25 * 0.85), 2)
        upper = round(pm25 * 1.15, 2)

        fc_row = Forecast(
            city=city,
            forecast_for=ts,
            pm25_predicted=pm25,
            lower_ci=lower,
            upper_ci=upper,
            aqi_category=_pm25_to_aqi_category(pm25),
        )
        db.add(fc_row)
        rows.append(fc_row)

    await db.commit()
    for r in rows:
        await db.refresh(r)

    return rows


async def generate_forecast(city: str, db: AsyncSession) -> list[Forecast]:
    """
    Generate a 24-hour PM2.5 forecast for a city.

    First attempts to use the Prophet ML model from R2/disk. If loading or
    inference fails (pkl trained with incompatible pandas version, model
    missing, etc.), falls back to the statistical hour-of-day forecast which
    only needs data already in the aqi_readings table.

    Args:
        city: City name — one of Islamabad, Karachi, Lahore, Peshawar, Quetta.
        db:   Async SQLAlchemy session.

    Returns:
        List of persisted Forecast ORM objects (24 rows).
    """
    # --- Try Prophet model first ---
    try:
        from services.model_loader import load_model

        model = load_model(city)

        # Cast ds to datetime64[ns] — pandas 2.x defaults to datetime64[us]
        # which causes dtype mismatch errors inside older Prophet pkl files.
        now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        future_timestamps_aware = [now + timedelta(hours=i) for i in range(1, 25)]
        future_df = pd.DataFrame({
            "ds": pd.to_datetime(
                [ts.replace(tzinfo=None) for ts in future_timestamps_aware]
            ).astype("datetime64[ns]")
        })

        forecast_df = model.predict(future_df)

        rows: list[Forecast] = []
        for i, (_, row) in enumerate(forecast_df.iterrows()):
            pm25 = max(0.0, float(row["yhat"]))
            lower = max(0.0, float(row["yhat_lower"]))
            upper = max(0.0, float(row["yhat_upper"]))

            fc_row = Forecast(
                city=city,
                forecast_for=future_timestamps_aware[i],
                pm25_predicted=pm25,
                lower_ci=lower,
                upper_ci=upper,
                aqi_category=_pm25_to_aqi_category(pm25),
            )
            db.add(fc_row)
            rows.append(fc_row)

        await db.commit()
        for r in rows:
            await db.refresh(r)

        print(f"[forecast] Prophet model forecast OK for {city}")
        return rows

    except Exception as exc:
        print(f"[forecast] Prophet failed for {city}: {exc!s:.200} — using statistical fallback")

    # --- Statistical fallback ---
    return await _statistical_forecast(city, db)
