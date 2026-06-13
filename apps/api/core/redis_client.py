"""Redis client using Upstash REST API (works in serverless too) or standard redis-py."""

import json
from typing import Any

import redis.asyncio as aioredis

from .config import settings

# Single shared async Redis connection pool
_redis_client: aioredis.Redis | None = None


def get_redis() -> aioredis.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = aioredis.from_url(settings.redis_url, decode_responses=True)
    return _redis_client


async def cache_get(key: str) -> Any | None:
    """Return decoded JSON value or None if missing/expired/unreachable."""
    try:
        r = get_redis()
        raw = await r.get(key)
        return json.loads(raw) if raw else None
    except Exception:
        return None


async def cache_set(key: str, value: Any, ttl_seconds: int = 300) -> None:
    """Store value serialised as JSON with TTL. No-op if Redis is unreachable."""
    try:
        r = get_redis()
        await r.set(key, json.dumps(value), ex=ttl_seconds)
    except Exception:
        pass


async def cache_delete(key: str) -> None:
    try:
        r = get_redis()
        await r.delete(key)
    except Exception:
        pass
