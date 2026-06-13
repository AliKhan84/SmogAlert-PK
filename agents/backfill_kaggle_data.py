"""
One-time script: load the existing Kaggle cleaned_data.csv into the aqi_readings table.

Uses bulk INSERT (executemany via asyncpg) instead of row-by-row upserts.
Inserts 127k rows in ~1-2 minutes instead of ~28 hours.

Usage:
  DATABASE_URL=<railway-url> python agents/backfill_kaggle_data.py
"""

import asyncio
import sys
from datetime import timezone
from pathlib import Path

import pandas as pd
from sqlalchemy import delete, text

# Add apps/api to path so we can import the ORM models
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "apps" / "api"))

from core.database import AsyncSessionLocal, engine
from models.aqi import AqiReading
from sqlalchemy.dialects.postgresql import insert as pg_insert


DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "cleaned_data.csv"

# Map cleaned_data.csv column names to DB column names
COLUMN_MAP = {
    "PM2.5": "pm25",
    "PM10": "pm10",
    "NO2":  "no2",
    "SO2":  "so2",
    "CO":   "co",
    "O3":   "o3",
    "NH3":  "nh3",
}

BATCH_SIZE = 5_000  # rows per INSERT batch — adjust down if you hit memory limits


def _safe_float(val):
    """Return float or None for NaN/missing values."""
    try:
        f = float(val)
        return None if (f != f) else f  # NaN check: NaN != NaN
    except (TypeError, ValueError):
        return None


async def backfill():
    print(f"Loading {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    print(f"  {len(df):,} rows loaded from CSV")

    # Rename columns to match DB schema
    rename = {}
    for col in df.columns:
        if col in COLUMN_MAP:
            rename[col] = COLUMN_MAP[col]
    df = df.rename(columns=rename)

    # Ensure timestamp is timezone-aware (UTC)
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    else:
        df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")

    # Normalize city names to Title Case (Islamabad, Lahore, etc.)
    df["city"] = df["city"].str.strip().str.title()

    # Use aqi_level column as aqi_category if present
    if "aqi_level" in df.columns and "aqi_category" not in df.columns:
        df["aqi_category"] = df["aqi_level"]
    elif "aqi_category" not in df.columns:
        df["aqi_category"] = None

    df["source"] = "kaggle"

    # Build list of plain dicts — much faster than creating ORM objects
    print("Building records list ...")
    records = []
    for _, row in df.iterrows():
        records.append({
            "city":           row["city"],
            "timestamp":      row["timestamp"].to_pydatetime(),
            "pm25":           _safe_float(row.get("pm25")),
            "pm10":           _safe_float(row.get("pm10")),
            "no2":            _safe_float(row.get("no2")),
            "so2":            _safe_float(row.get("so2")),
            "co":             _safe_float(row.get("co")),
            "o3":             _safe_float(row.get("o3")),
            "nh3":            _safe_float(row.get("nh3")),
            "aqi_calculated": _safe_float(row.get("aqi")),
            "aqi_category":   row.get("aqi_category") if pd.notna(row.get("aqi_category", None)) else None,
            "is_anomaly":     False,
            "source":         "kaggle",
        })
    print(f"  {len(records):,} records prepared")

    async with AsyncSessionLocal() as db:
        # Remove any previously inserted kaggle rows so re-runs are idempotent
        print("Clearing existing kaggle rows ...")
        await db.execute(delete(AqiReading).where(AqiReading.source == "kaggle"))
        await db.commit()
        print("  Done.")

        # Bulk-insert in batches using asyncpg executemany
        print(f"Inserting {len(records):,} rows in batches of {BATCH_SIZE:,} ...")
        total_inserted = 0
        num_batches = (len(records) + BATCH_SIZE - 1) // BATCH_SIZE

        for batch_num, start in enumerate(range(0, len(records), BATCH_SIZE), 1):
            batch = records[start : start + BATCH_SIZE]
            await db.execute(pg_insert(AqiReading), batch)
            await db.commit()
            total_inserted += len(batch)
            pct = total_inserted / len(records) * 100
            print(f"  Batch {batch_num}/{num_batches} — {total_inserted:,}/{len(records):,} rows ({pct:.1f}%)")

    print(f"\nDone. Inserted {total_inserted:,} rows into aqi_readings.")


if __name__ == "__main__":
    asyncio.run(backfill())
