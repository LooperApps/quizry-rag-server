import logging

from fastapi import APIRouter

from app.apps import all_apps
from app.db.chroma import get_collection
from app.log_utils import log_step

router = APIRouter(tags=["health"])
logger = logging.getLogger(__name__)


@router.get("/health")
async def health() -> dict:
    """
    Render health-check endpoint.
    Returns 200 OK when the server and ChromaDB are operational.

    Reports a per-app breakdown, but no app ids beyond their names and counts —
    this route is unauthenticated because Render's health checker cannot carry
    a credential.
    """
    log_step(logger, "HEALTH", "Health check requested")
    try:
        per_app = {}
        total = 0
        for app in all_apps():
            count = get_collection(app.collection).count()
            per_app[app.id] = {"collection": app.collection, "totalDocs": count}
            total += count
        log_step(logger, "HEALTH", f"ChromaDB connected, totalDocs={total}")
        return {
            "status": "ok",
            "chromadb": "connected",
            "totalDocs": total,
            "apps": per_app,
        }
    except Exception as exc:
        log_step(logger, "HEALTH", f"ChromaDB degraded: {exc}")
        return {"status": "degraded", "chromadb": str(exc)}
