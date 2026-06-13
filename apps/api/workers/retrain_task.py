"""Celery task: daily model retraining — delegates to agents/retrain_agent.py."""

import subprocess
import sys
from pathlib import Path

from workers.celery_app import celery_app

# In the monorepo the file lives at apps/api/workers/retrain_task.py (parents[3] = repo root).
# In the Railway container only apps/api is copied to /app, so parents[3] doesn't exist;
# fall back to /app so the module can at least import cleanly.
_this = Path(__file__).resolve()
REPO_ROOT = _this.parents[3] if len(_this.parents) > 3 else _this.parents[1]
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
