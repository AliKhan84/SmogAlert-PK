"""Clerk JWT verification middleware for FastAPI."""

import httpx
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from .config import settings

bearer_scheme = HTTPBearer(auto_error=False)

_jwks_cache: dict | None = None


async def _get_clerk_jwks() -> dict:
    """Fetch Clerk's JWKS (cached after first fetch)."""
    global _jwks_cache
    if _jwks_cache:
        return _jwks_cache
    # Clerk JWKS endpoint derived from the secret key prefix
    # e.g. sk_live_xxx → instance is xxx
    async with httpx.AsyncClient() as client:
        resp = await client.get("https://api.clerk.com/v1/jwks", headers={
            "Authorization": f"Bearer {settings.clerk_secret_key}"
        })
        resp.raise_for_status()
        _jwks_cache = resp.json()
    return _jwks_cache


async def get_current_user_id(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> str:
    """Validate Clerk session JWT and return the Clerk user ID (sub claim)."""
    if credentials is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")

    token = credentials.credentials
    try:
        # Decode without verification first to get the key ID
        header = jwt.get_unverified_header(token)
        jwks = await _get_clerk_jwks()

        # Find the matching key
        key = next((k for k in jwks.get("keys", []) if k.get("kid") == header.get("kid")), None)
        if key is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token key")

        payload = jwt.decode(token, key, algorithms=["RS256"])
        user_id: str = payload.get("sub", "")
        if not user_id:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload")
        return user_id

    except JWTError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token") from exc


async def require_admin(user_id: str = Depends(get_current_user_id)) -> str:
    """Dependency that only allows admin users (IDs set in env ADMIN_USER_IDS)."""
    if user_id not in settings.admin_ids:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")
    return user_id
