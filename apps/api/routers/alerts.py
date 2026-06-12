"""Alert history and test-send endpoints."""

from fastapi import APIRouter, Depends
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.auth import get_current_user_id
from core.database import get_db
from models.user import AlertSent, User
from schemas.aqi import AlertHistoryOut

router = APIRouter(prefix="/alerts", tags=["alerts"])


@router.get("", response_model=list[AlertHistoryOut])
async def get_alert_history(
    limit: int = 50,
    offset: int = 0,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    """Return the current user's alert history, newest first."""
    result = await db.execute(select(User).where(User.clerk_user_id == user_id))
    user = result.scalar_one_or_none()
    if user is None:
        return []

    rows = await db.execute(
        select(AlertSent)
        .where(AlertSent.user_id == user.id)
        .order_by(desc(AlertSent.sent_at))
        .limit(limit)
        .offset(offset)
    )
    return rows.scalars().all()


@router.post("/test")
async def send_test_alert(
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    """Send a test alert to verify the user's configured channels."""
    result = await db.execute(select(User).where(User.clerk_user_id == user_id))
    user = result.scalar_one_or_none()
    if user is None:
        return {"status": "error", "detail": "User not found"}

    from services.alert_dispatcher import dispatch_alert
    await dispatch_alert(
        user=user,
        city="Lahore",
        aqi_level="Test",
        aqi_value=155.0,
        message_en="This is a test alert from SmogAlert PK. Your notifications are working!",
        message_ur="یہ SmogAlert PK کا ایک ٹیسٹ الرٹ ہے۔ آپ کی اطلاعات کام کر رہی ہیں!",
        db=db,
    )
    return {"status": "ok", "detail": "Test alert dispatched"}
