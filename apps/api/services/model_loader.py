"""
Prophet model loader with in-memory caching and Cloudflare R2 fallback.

This is a Railway-deployed copy of packages/ml/model_loader.py.
The key difference: MODELS_DIR points to /tmp/prophet_models/ so pkl files
don't collide with the apps/api/models/ Python package directory.

Load priority:
  1. In-memory cache (if not older than MODEL_CACHE_TTL seconds)
  2. Local disk  → MODELS_DIR/prophet_{city}.pkl
  3. Cloudflare R2 → download to disk, then load
"""

import os
import time
from pathlib import Path
from typing import Any

import joblib

# Monkey-patch StringDtype to accept extra positional args from pickled models
# trained with pandas versions that had a different __init__ signature.
# This lets older .pkl files (which stored `StringDtype(storage, na_value)`)
# unpickle correctly when the current pandas only accepts `StringDtype(storage)`.
try:
    import pandas as _pd
    _orig_sd_init = _pd.core.arrays.string_.StringDtype.__init__

    def _compat_sd_init(self, *args, **kwargs):
        _orig_sd_init(self, args[0] if args else kwargs.get("storage"))

    _pd.core.arrays.string_.StringDtype.__init__ = _compat_sd_init
except Exception:
    pass

# Use /tmp so pkl files don't conflict with the apps/api/models/ Python package.
# Falls back to MODELS_DIR_PATH env var if set (useful for local dev).
MODELS_DIR = Path(os.getenv("MODELS_DIR_PATH", "/tmp/prophet_models"))

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
    Download prophet_{city}.pkl from R2 to the local MODELS_DIR.

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
                "Upload models via: python agents/retrain_agent.py --force --city " + city
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
