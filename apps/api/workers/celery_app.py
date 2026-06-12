"""Celery application factory with scheduled tasks (Celery Beat)."""

from celery import Celery
from celery.schedules import crontab

from core.config import settings

celery_app = Celery(
    "smokalert",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=[
        "workers.scrape_task",
        "workers.alert_task",
        "workers.retrain_task",
    ],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="Asia/Karachi",
    enable_utc=True,
    beat_schedule={
        # Scrape live AQI data every hour
        "scrape-aqi-hourly": {
            "task": "workers.scrape_task.scrape_all_cities",
            "schedule": crontab(minute=5),  # 5 past every hour
        },
        # Run alert engine every hour (after scrape)
        "alert-engine-hourly": {
            "task": "workers.alert_task.run_alert_engine",
            "schedule": crontab(minute=15),  # 15 past every hour
        },
        # Retrain models daily at 2 AM PKT
        "retrain-daily": {
            "task": "workers.retrain_task.retrain_models",
            "schedule": crontab(hour=2, minute=0),
        },
    },
)
