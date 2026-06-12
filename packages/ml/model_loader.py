"""
Prophet model loader with in-memory caching and Cloudflare R2 fallback.

Load priority:
  1. In-memory cache (if not older than MODEL_CACHE_TTL seconds)
  2. Local disk  → models/prophet_{city}.pkl
  3. Cloudflare R2 → download to disk, then load

Usage:
  from model_loader import load_model, invalidate_cache
  model = load_model("Lahore")
  forecast_df = model.predict(future_df)
"""

import os
import time
from pathlib import Path
from typing import Any

import joblib

# models/ lives two levels above packages/ml/
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

# In-memory cache keyed by city name
_cache: dict[str, dict[str, Any]] = {}
MODEL_CACHE_TTL = 3600  # seconds — reload from disk/R2 after 1 hour


# ── R2 helpers ────────────────────────────────────────────────────────────────

def _r2_client():
    """Return boto3 S3-compatible R2 client, or None if env vars missing."""
    endpoint = os.getenv("CLOUDFLARE_R2_ENDPOINT", "")
    access_key = os.getenv("CLOUDFLARE_R2_ACCESS_KEY", "")
    secret_key = os.getenv("CLOUDFLARE_R2_SECRET_KEY", "")
    if not (endpoint and access_key):
        return None
    try:
        import boto3
        return boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name="auto",
        )
    except Exception:
        return None


def _download_from_r2(city: str) -> bool:
    """
    Download prophet_{city}.pkl from R2 to the local models/ directory.

    Returns True if the download succeeded, False otherwise.
    """
    client = _r2_client()
    if not client:
        return False

    bucket = os.getenv("CLOUDFLARE_R2_BUCKET", "smogalert-pk")
    r2_key = f"models/prophet_{city}.pkl"
    local_path = MODELS_DIR / f"prophet_{city}.pkl"

    try:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        client.download_file(bucket, r2_key, str(local_path))
        print(f"[model_loader] Downloaded R2:{r2_key} → {local_path}")
        return True
    except Exception as exc:
        print(f"[model_loader] R2 download failed for {city}: {exc}")
        return False


# ── Public API ────────────────────────────────────────────────────────────────

def load_model(city: str) -> Any:
    """
    Load the Prophet PM2.5 model for a given city.

    Checks in-memory cache first, then local disk, then Cloudflare R2.

    Args:
        city: City name — one of Islamabad, Karachi, Lahore, Peshawar, Quetta.

    Returns:
        Loaded Prophet model object ready for .predict(future_df).

    Raises:
        FileNotFoundError: If the model cannot be found locally or on R2.
    """
    # 1. In-memory cache hit
    entry = _cache.get(city)
    if entry and (time.time() - entry["ts"]) < MODEL_CACHE_TTL:
        return entry["model"]

    # 2. Local disk
    local_path = MODELS_DIR / f"prophet_{city}.pkl"
    if not local_path.exists():
        print(f"[model_loader] {local_path} not found — trying R2 ...")
        if not _download_from_r2(city):
            raise FileNotFoundError(
                f"Prophet model for {city} not found on disk or R2. "
                "Run: python agents/retrain_agent.py --force --city " + city
            )

    # 3. Load from disk and cache
    model = joblib.load(local_path)
    _cache[city] = {"model": model, "ts": time.time()}
    print(f"[model_loader] Loaded {city} model (cache miss)")
    return model


def invalidate_cache(city: str | None = None) -> None:
    """
    Evict cached models so the next call to load_model() re-reads from disk.

    Args:
        city: If provided, evict only this city. If None, evict all.
    """
    if city:
        _cache.pop(city, None)
        print(f"[model_loader] Cache evicted for {city}")
    else:
        _cache.clear()
        print("[model_loader] Full cache cleared")
