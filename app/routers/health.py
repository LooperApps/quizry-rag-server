from fastapi import APIRouter

from app.db.chroma import get_collection

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict:
    """
    Render health-check endpoint.
    Returns 200 OK when the server and ChromaDB are operational.
    """
    try:
        col = get_collection()
        doc_count = col.count()
        return {"status": "ok", "chromadb": "connected", "totalDocs": doc_count}
    except Exception as exc:
        # Return 200 but with error details so Render doesn't restart the server
        # for a transient ChromaDB issue. Change to raise if you want hard failures.
        return {"status": "degraded", "chromadb": str(exc)}
