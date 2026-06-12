"""
Standalone model retrain orchestrator for SmogAlert PK.

- Checks if ≥500 new DB rows have accumulated since the last retrain.
- Exports aqi_readings from PostgreSQL to CSV for the ML pipeline.
- Runs preprocess.py → model.py for each city.
- Evaluates Prophet MAE on recent data.
- Uploads new .pkl files to Cloudflare R2 if R2 is configured.
- Logs the retrain event to the model_versions table.

Run manually:    python agents/retrain_agent.py [--force] [--city Lahore]
Scheduled via:   workers/retrain_task.py (Celery Beat, daily 2 AM PKT)
                 .github/workflows/cron.yml (GitHub Actions fallback)
"""

import argparse
import asyncio
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

# ── Path setup ───────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent   # SmogAlert-PK/
ML_DIR = REPO_ROOT / "packages" / "ml"
MODELS_DIR = REPO_ROOT / "models"
API_DIR = REPO_ROOT / "apps" / "api"

# Add api/ to path so we can import SQLAlchemy models
sys.path.insert(0, str(API_DIR))

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

# ── Config from env ───────────────────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL", "")
R2_ACCESS_KEY = os.getenv("CLOUDFLARE_R2_ACCESS_KEY", "")
R2_SECRET_KEY = os.getenv("CLOUDFLARE_R2_SECRET_KEY", "")
R2_BUCKET = os.getenv("CLOUDFLARE_R2_BUCKET", "smogalert-pk")
R2_ENDPOINT = os.getenv("CLOUDFLARE_R2_ENDPOINT", "")

CITIES = ["Islamabad", "Karachi", "Lahore", "Peshawar", "Quetta"]
MIN_NEW_ROWS = 500  # Minimum new rows required before triggering a retrain


# ── R2 helpers ────────────────────────────────────────────────────────────────

def _get_r2_client():
    """Return a boto3 S3-compatible client pointed at Cloudflare R2, or None."""
    if not (R2_ACCESS_KEY and R2_ENDPOINT):
        return None
    try:
        import boto3
        return boto3.client(
            "s3",
            endpoint_url=R2_ENDPOINT,
            aws_access_key_id=R2_ACCESS_KEY,
            aws_secret_access_key=R2_SECRET_KEY,
            region_name="auto",
        )
    except Exception as exc:
        print(f"[retrain_agent] Could not create R2 client: {exc}")
        return None


def _upload_model_to_r2(r2_client, city: str) -> str | None:
    """Upload prophet_{city}.pkl to R2. Returns the R2 object key on success."""
    local_path = MODELS_DIR / f"prophet_{city}.pkl"
    if not local_path.exists():
        print(f"[retrain_agent] No model at {local_path} — skipping R2 upload")
        return None

    r2_key = f"models/prophet_{city}.pkl"
    try:
        r2_client.upload_file(str(local_path), R2_BUCKET, r2_key)
        print(f"[retrain_agent] Uploaded {city} model → R2:{r2_key}")
        return r2_key
    except Exception as exc:
        print(f"[retrain_agent] R2 upload failed for {city}: {exc}")
        return None


# ── Model evaluation ──────────────────────────────────────────────────────────

def _evaluate_prophet_mae(city: str) -> float | None:
    """
    Load the freshly trained Prophet model and compute MAE on the last 24
    rows of the cleaned CSV as a quick sanity check.
    """
    try:
        model_path = MODELS_DIR / f"prophet_{city}.pkl"
        if not model_path.exists():
            return None

        model = joblib.load(model_path)

        cleaned_csv = REPO_ROOT / "data" / "cleaned_data.csv"
        if not cleaned_csv.exists():
            return None

        df = pd.read_csv(cleaned_csv)
        df = df[df["city"] == city].copy()

        # Normalise column names — legacy CSV uses 'date', live export uses 'timestamp'
        ts_col = "timestamp" if "timestamp" in df.columns else "date"
        df = df.rename(columns={ts_col: "ds", "pm25": "y"})
        df["ds"] = pd.to_datetime(df["ds"])
        df = df[["ds", "y"]].dropna().sort_values("ds")

        if len(df) < 48:
            return None

        eval_df = df.tail(24).reset_index(drop=True)
        predictions = model.predict(eval_df[["ds"]])
        mae = float((eval_df["y"] - predictions["yhat"]).abs().mean())
        return mae
    except Exception as exc:
        print(f"[retrain_agent] MAE eval failed for {city}: {exc}")
        return None


# ── Database helpers ──────────────────────────────────────────────────────────

async def _count_new_rows(engine) -> int:
    """Count aqi_readings rows added since the last promoted retrain."""
    from models.aqi import AqiReading, ModelVersion

    async with AsyncSession(engine) as db:
        # Find the last successful retrain timestamp
        result = await db.execute(
            select(ModelVersion.trained_at)
            .where(ModelVersion.promoted.is_(True))
            .order_by(ModelVersion.trained_at.desc())
            .limit(1)
        )
        last_retrain = result.scalar_one_or_none()

        if last_retrain:
            count_q = select(func.count(AqiReading.id)).where(
                AqiReading.created_at > last_retrain
            )
        else:
            count_q = select(func.count(AqiReading.id))

        count = (await db.execute(count_q)).scalar_one()
        return count or 0


async def _export_db_to_csv(engine) -> bool:
    """Export all aqi_readings to packages/ml/data/raw_aqi_data.csv for ML pipeline."""
    from models.aqi import AqiReading

    async with AsyncSession(engine) as db:
        result = await db.execute(
            select(AqiReading).order_by(AqiReading.timestamp)
        )
        rows = result.scalars().all()

    if not rows:
        print("[retrain_agent] No DB rows to export — keeping existing CSV")
        return False

    records = [
        {
            "city": r.city,
            "timestamp": r.timestamp.isoformat(),
            "pm25": r.pm25,
            "pm10": r.pm10,
            "no2": r.no2,
            "so2": r.so2,
            "co": r.co,
            "o3": r.o3,
            "nh3": r.nh3,
            "aqi_calculated": r.aqi_calculated,
            "aqi_category": r.aqi_category,
        }
        for r in rows
    ]

    out_path = ML_DIR / "data" / "raw_aqi_data.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(out_path, index=False)
    print(f"[retrain_agent] Exported {len(records)} rows → {out_path}")
    return True


async def _log_model_version(engine, city: str, mae: float | None, r2_key: str | None, promoted: bool):
    """Insert a row into model_versions to record this retrain event."""
    from models.aqi import ModelVersion

    async with AsyncSession(engine) as db:
        mv = ModelVersion(
            model_type="prophet",
            city=city,
            mae=mae,
            trained_at=datetime.now(timezone.utc),
            promoted=promoted,
            r2_key=r2_key,
        )
        db.add(mv)
        await db.commit()
        print(f"[retrain_agent] Logged model_version: city={city} mae={mae} promoted={promoted}")


# ── Main retrain logic ────────────────────────────────────────────────────────

async def retrain(force: bool = False, city: str | None = None) -> None:
    """
    Entry point for model retraining.

    Args:
        force: If True, skip the row-count gate and always retrain.
        city:  If set, retrain only this city's Prophet model.
    """
    if not DATABASE_URL:
        print("[retrain_agent] DATABASE_URL not configured — aborting")
        return

    print(f"\n{'='*60}")
    print(f"[retrain_agent] Starting retrain — {datetime.now(timezone.utc).isoformat()}")
    print(f"[retrain_agent] force={force} city={city or 'all'}")
    print(f"{'='*60}\n")

    engine = create_async_engine(DATABASE_URL, echo=False)

    # Gate: only retrain if enough new data has arrived
    if not force:
        new_rows = await _count_new_rows(engine)
        print(f"[retrain_agent] New rows since last retrain: {new_rows}")
        if new_rows < MIN_NEW_ROWS:
            print(f"[retrain_agent] Below threshold ({MIN_NEW_ROWS}) — skipping retrain")
            await engine.dispose()
            return

    # Export DB → CSV so the ML scripts can read it
    await _export_db_to_csv(engine)

    # Run preprocessing step
    print("\n[retrain_agent] Step 1: preprocess.py ...")
    proc = subprocess.run(
        [sys.executable, str(ML_DIR / "src" / "preprocess.py")],
        capture_output=True, text=True, cwd=str(ML_DIR),
    )
    if proc.returncode != 0:
        print(f"[retrain_agent] preprocess.py failed:\n{proc.stderr}")
        await engine.dispose()
        return
    print("[retrain_agent] preprocess.py — OK")

    # Train per-city Prophet models
    r2 = _get_r2_client()
    if not r2:
        print("[retrain_agent] R2 not configured — models will be local only")

    cities_to_train = [city] if city else CITIES
    for c in cities_to_train:
        print(f"\n[retrain_agent] Step 2: Training Prophet for {c} ...")
        start = time.time()

        train_proc = subprocess.run(
            [sys.executable, str(ML_DIR / "src" / "model.py"), "--city", c],
            capture_output=True, text=True, cwd=str(ML_DIR),
        )

        if train_proc.returncode != 0:
            print(f"[retrain_agent] model.py failed for {c}:\n{train_proc.stderr}")
            await _log_model_version(engine, c, None, None, promoted=False)
            continue

        elapsed = time.time() - start
        print(f"[retrain_agent] {c} trained in {elapsed:.1f}s")

        # Evaluate MAE
        mae = _evaluate_prophet_mae(c)
        if mae is not None:
            print(f"[retrain_agent] {c} Prophet MAE = {mae:.2f} µg/m³")

        # Upload to R2
        r2_key = _upload_model_to_r2(r2, c) if r2 else None

        # Log to DB
        await _log_model_version(engine, c, mae, r2_key, promoted=True)

    await engine.dispose()
    print(f"\n[retrain_agent] Retrain complete — {datetime.now(timezone.utc).isoformat()}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SmogAlert PK — model retrain agent")
    parser.add_argument("--force", action="store_true", help="Skip row-count check, always retrain")
    parser.add_argument("--city", type=str, default=None, choices=CITIES + [None],
                        help="Retrain only this city (default: all)")
    args = parser.parse_args()
    asyncio.run(retrain(force=args.force, city=args.city))
