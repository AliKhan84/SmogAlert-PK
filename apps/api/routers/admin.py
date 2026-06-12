"""Admin-only endpoints: model status, retrain trigger, ingestion health."""

from fastapi import APIRouter, Depends
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.auth import require_admin
from core.database import get_db
from models.aqi import ModelVersion

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/model-status")
async def model_status(
    _admin: str = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Return model version history ordered by newest first."""
    result = await db.execute(
        select(ModelVersion).order_by(desc(ModelVersion.trained_at)).limit(50)
    )
    rows = result.scalars().all()
    return [
        {
            "id": r.id,
            "model_type": r.model_type,
            "city": r.city,
            "mae": r.mae,
            "r2_score": r.r2_score,
            "trained_at": r.trained_at.isoformat(),
            "promoted": r.promoted,
            "r2_key": r.r2_key,
        }
        for r in rows
    ]


@router.post("/retrain")
async def trigger_retrain(
    city: str | None = None,
    _admin: str = Depends(require_admin),
):
    """Manually trigger a model retrain. Queues a Celery task."""
    from workers.celery_app import celery_app
    task = celery_app.send_task("workers.retrain_task.retrain_models", kwargs={"city": city})
    return {"status": "queued", "task_id": task.id}


@router.get("/ingestion-status")
async def ingestion_status(
    _admin: str = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Return the most recent data ingestion timestamp per city and source."""
    from models.aqi import AqiReading
    from sqlalchemy import func
    result = await db.execute(
        select(
            AqiReading.city,
            AqiReading.source,
            func.max(AqiReading.timestamp).label("last_seen"),
            func.count(AqiReading.id).label("total_rows"),
        ).group_by(AqiReading.city, AqiReading.source)
    )
    rows = result.all()
    return [
        {"city": r.city, "source": r.source, "last_seen": r.last_seen.isoformat() if r.last_seen else None, "total_rows": r.total_rows}
        for r in rows
    ]
