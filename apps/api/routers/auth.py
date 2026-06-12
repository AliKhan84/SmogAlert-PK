"""Clerk webhook handler — creates user row on signup."""

import hashlib
import hmac
import json

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import settings
from core.database import get_db
from models.user import AlertPreferences, Subscription, User

router = APIRouter(prefix="/auth", tags=["auth"])


def _verify_clerk_signature(payload: bytes, svix_id: str, svix_timestamp: str, svix_signature: str) -> bool:
    """Verify Clerk webhook signature using svix headers."""
    signed_content = f"{svix_id}.{svix_timestamp}.{payload.decode()}"
    secret = settings.clerk_webhook_secret.removeprefix("whsec_")
    import base64
    secret_bytes = base64.b64decode(secret)
    expected = hmac.new(secret_bytes, signed_content.encode(), hashlib.sha256).digest()
    expected_b64 = base64.b64encode(expected).decode()
    # svix_signature may contain multiple space-separated v1,<sig> pairs
    for sig in svix_signature.split(" "):
        if sig.startswith("v1,"):
            if hmac.compare_digest(sig[3:], expected_b64):
                return True
    return False


@router.post("/webhook", status_code=status.HTTP_200_OK)
async def clerk_webhook(
    request: Request,
    db: AsyncSession = Depends(get_db),
    svix_id: str = Header(None),
    svix_timestamp: str = Header(None),
    svix_signature: str = Header(None),
):
    """Receive Clerk webhook events and sync user data to our database."""
    payload = await request.body()

    # Verify signature in production
    if settings.clerk_webhook_secret and not _verify_clerk_signature(
        payload, svix_id or "", svix_timestamp or "", svix_signature or ""
    ):
        raise HTTPException(status_code=400, detail="Invalid webhook signature")

    event = json.loads(payload)
    event_type = event.get("type")
    data = event.get("data", {})

    if event_type == "user.created":
        email = (data.get("email_addresses") or [{}])[0].get("email_address", "")
        name = f"{data.get('first_name', '')} {data.get('last_name', '')}".strip()

        user = User(
            clerk_user_id=data["id"],
            email=email,
            name=name or None,
        )
        db.add(user)
        await db.flush()  # Get the user ID

        # Create default subscription (free) and alert preferences
        db.add(Subscription(user_id=user.id, plan="free"))
        db.add(AlertPreferences(user_id=user.id))
        await db.commit()

    elif event_type == "user.deleted":
        from sqlalchemy import select
        result = await db.execute(select(User).where(User.clerk_user_id == data["id"]))
        user = result.scalar_one_or_none()
        if user:
            await db.delete(user)
            await db.commit()

    return {"status": "ok"}
