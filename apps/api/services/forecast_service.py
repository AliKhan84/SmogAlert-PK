"""
Load city Prophet models and generate 24-hour PM2.5 forecasts.

Model loading delegates to packages/ml/model_loader.py which handles:
- In-memory caching (1-hour TTL)
- Local disk fallback (models/prophet_{city}.pkl)
- Cloudflare R2 download if not found locally
"""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from models.aqi import Forecast

# Add packages/ml to path so we can import model_loader
_ML_PKG = Path(__file__).resolve().parents[3] / "packages" / "ml"
if str(_ML_PKG) not in sys.path:
    sys.path.insert(0, str(_ML_PKG))

from model_loader import load_model  # noqa: E402


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


async def generate_forecast(city: str, db: AsyncSession) -> list[Forecast]:
    """
    Run the Prophet model for a city and persist 24 forecast rows to the DB.

    Uses model_loader so the model is served from memory cache on subsequent
    calls (avoiding repeated disk/R2 reads within the same process).

    Args:
        city: City name — one of Islamabad, Karachi, Lahore, Peshawar, Quetta.
        db:   Async SQLAlchemy session (provided by FastAPI dependency).

    Returns:
        List of persisted Forecast ORM objects (one per forecast hour).

    Raises:
        FileNotFoundError: If no Prophet model exists for the city.
    """
    model = load_model(city)

    # Build a 24-hour future DataFrame starting from the next full hour
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    future_timestamps = [now + timedelta(hours=i) for i in range(1, 25)]
    future_df = pd.DataFrame({"ds": future_timestamps})

    forecast_df = model.predict(future_df)

    rows: list[Forecast] = []
    for _, row in forecast_df.iterrows():
        pm25 = max(0.0, float(row["yhat"]))
        lower = max(0.0, float(row["yhat_lower"]))
        upper = max(0.0, float(row["yhat_upper"]))

        fc_row = Forecast(
            city=city,
            forecast_for=row["ds"].to_pydatetime().replace(tzinfo=timezone.utc),
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
