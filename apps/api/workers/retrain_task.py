"""Celery task: daily model retraining — delegates to agents/retrain_agent.py."""

import asyncio
import subprocess
import sys
from pathlib import Path

from workers.celery_app import celery_app

# repo root is 3 levels up from apps/api/workers/
REPO_ROOT = Path(__file__).resolve().parents[3]
RETRAIN_AGENT = REPO_ROOT / "agents" / "retrain_agent.py"


@celery_app.task(name="workers.retrain_task.retrain_models", bind=True)
def retrain_models(self, city: str | None = None, force: bool = False):
    """
    Trigger model retraining by running agents/retrain_agent.py as a subprocess.

    Running as a subprocess keeps the heavy ML dependencies (Prophet, sklearn)
    out of the Celery worker process memory between retrains.

    Args:
        city:  If set, retrain only this city's Prophet model.
        force: Skip the new-row count check and always retrain.
    """
    cmd = [sys.executable, str(RETRAIN_AGENT)]
    if force:
        cmd.append("--force")
    if city:
        cmd.extend(["--city", city])

    print(f"[retrain_task] Running: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )

    print(result.stdout)
    if result.returncode != 0:
        print(f"[retrain_task] retrain_agent.py exited with code {result.returncode}:\n{result.stderr}")
        raise RuntimeError(f"Retrain failed (exit {result.returncode})")

    print("[retrain_task] Retrain completed successfully")
