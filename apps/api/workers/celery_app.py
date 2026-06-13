"""Celery application factory with scheduled tasks (Celery Beat)."""

import ssl

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

# Upstash Redis uses rediss:// (TLS). Celery requires ssl_cert_reqs to be
# set explicitly — CERT_NONE because Upstash uses managed certificates.
_ssl_config = {"ssl_cert_reqs": ssl.CERT_NONE}
_use_ssl = settings.redis_url.startswith("rediss://")

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="Asia/Karachi",
    enable_utc=True,
    **({"broker_use_ssl": _ssl_config, "redis_backend_use_ssl": _ssl_config} if _use_ssl else {}),
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
