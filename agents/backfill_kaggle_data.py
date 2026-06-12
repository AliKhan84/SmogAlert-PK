"""
One-time script: load the existing Kaggle cleaned_data.csv into the aqi_readings table.

Run this once after deploying to Supabase to seed the database with historical data.
This gives the dashboard historical charts from Day 1 and gives Prophet enough data to forecast.

Usage:
  DATABASE_URL=<supabase-url> python agents/backfill_kaggle_data.py
"""

import asyncio
import sys
from datetime import timezone
from pathlib import Path

import pandas as pd

# Add apps/api to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "apps" / "api"))

from core.database import AsyncSessionLocal
from models.aqi import AqiReading
from sqlalchemy import select


DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "cleaned_data.csv"

COLUMN_MAP = {
    "PM2.5": "pm25",
    "PM10": "pm10",
    "NO2": "no2",
    "SO2": "so2",
    "CO": "co",
    "O3": "o3",
    "NH3": "nh3",
}


async def backfill():
    print(f"Loading {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])

    # Rename columns to DB schema names
    rename = {}
    for col in df.columns:
        lower = col.lower()
        if lower in COLUMN_MAP:
            rename[col] = COLUMN_MAP[lower]
        elif col in COLUMN_MAP:
            rename[col] = COLUMN_MAP[col]
    df = df.rename(columns=rename)

    # Ensure required columns exist
    if "city" not in df.columns or "timestamp" not in df.columns:
        print("ERROR: cleaned_data.csv must have 'city' and 'timestamp' columns")
        return

    # Normalize city names
    df["city"] = df["city"].str.strip().str.title()

    # Map AQI level to aqi_category
    if "aqi_level" in df.columns:
        df["aqi_category"] = df["aqi_level"]
    if "aqi_category" not in df.columns:
        df["aqi_category"] = None

    df["source"] = "kaggle"

    print(f"Inserting {len(df)} rows into aqi_readings ...")
    batch_size = 500
    inserted = 0

    async with AsyncSessionLocal() as db:
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i : i + batch_size]
            for _, row in batch.iterrows():
                ts = row["timestamp"]
                if hasattr(ts, "tzinfo") and ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)

                # Skip if already exists
                existing = await db.execute(
                    select(AqiReading).where(
                        AqiReading.city == row["city"],
                        AqiReading.timestamp == ts,
                        AqiReading.source == "kaggle",
                    )
                )
                if existing.scalar_one_or_none():
                    continue

                r = AqiReading(
                    city=row["city"],
                    timestamp=ts,
                    pm25=row.get("pm25") if pd.notna(row.get("pm25")) else None,
                    pm10=row.get("pm10") if pd.notna(row.get("pm10")) else None,
                    no2=row.get("no2") if pd.notna(row.get("no2")) else None,
                    so2=row.get("so2") if pd.notna(row.get("so2")) else None,
                    co=row.get("co") if pd.notna(row.get("co")) else None,
                    o3=row.get("o3") if pd.notna(row.get("o3")) else None,
                    nh3=row.get("nh3") if pd.notna(row.get("nh3")) else None,
                    aqi_calculated=row.get("aqi") if pd.notna(row.get("aqi", None)) else None,
                    aqi_category=row.get("aqi_category") if pd.notna(row.get("aqi_category", None)) else None,
                    source="kaggle",
                )
                db.add(r)
                inserted += 1

            await db.commit()
            print(f"  Batch {i//batch_size + 1}: committed {inserted} rows so far")

    print(f"Done. Inserted {inserted} rows.")


if __name__ == "__main__":
    asyncio.run(backfill())
