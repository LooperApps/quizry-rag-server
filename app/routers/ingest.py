"""
Ingestion router.

POST   /ingest                               — ingest a source file (sync, returns after done)
DELETE /ingest/notebooks/{notebookId}        — delete all chunks for a notebook
DELETE /ingest/notebooks/{notebookId}/sources/{sourceId} — delete chunks for one source
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.auth import verify_api_key
from app.models.ingest import IngestRequest, IngestResponse
from app.services.ingest_pipeline import (
    _delete_source_chunks,
    delete_notebook_chunks,
    run_ingest,
)

router = APIRouter(prefix="/ingest", tags=["ingest"])
logger = logging.getLogger(__name__)


@router.post(
    "",
    response_model=IngestResponse,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(verify_api_key)],
    summary="Ingest a source file into ChromaDB",
    description=(
        "Downloads the file from `storageUrl`, extracts text, chunks it via LLM, "
        "embeds the chunks, and stores them in ChromaDB keyed by `notebookId` + `sourceId`. "
        "This call is **synchronous** — it returns only after ingestion completes. "
        "The Cloud Function should set the source status to `ready` on 200, `error` on 5xx."
    ),
)
async def ingest(req: IngestRequest) -> IngestResponse:
    logger.info(
        f"[ingest] Request notebook={req.notebookId} source={req.sourceId} file={req.fileName}"
    )
    try:
        chunk_count = await run_ingest(
            notebook_id=req.notebookId,
            source_id=req.sourceId,
            storage_url=req.storageUrl,
            file_name=req.fileName,
        )
        return IngestResponse(
            status="ok",
            sourceId=req.sourceId,
            chunkCount=chunk_count,
        )
    except ValueError as exc:
        # Validation / extraction errors — 422 so Cloud Function treats them as permanent
        logger.warning(f"[ingest] Validation error for source={req.sourceId}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.error(
            f"[ingest] Unexpected error for source={req.sourceId}: {exc}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {exc}",
        ) from exc


@router.delete(
    "/notebooks/{notebook_id}",
    dependencies=[Depends(verify_api_key)],
    summary="Delete all chunks for a notebook",
    description="Called by onNotebookDeleted Cloud Function trigger.",
)
async def delete_notebook(notebook_id: str) -> dict:
    logger.info(f"[ingest] Delete notebook chunks: notebook={notebook_id}")
    delete_notebook_chunks(notebook_id)
    return {"status": "ok", "notebookId": notebook_id}


@router.delete(
    "/notebooks/{notebook_id}/sources/{source_id}",
    dependencies=[Depends(verify_api_key)],
    summary="Delete chunks for one source file",
    description="Called when a source document is removed from a notebook.",
)
async def delete_source(notebook_id: str, source_id: str) -> dict:
    logger.info(f"[ingest] Delete source chunks: notebook={notebook_id} source={source_id}")
    _delete_source_chunks(notebook_id, source_id)
    return {"status": "ok", "notebookId": notebook_id, "sourceId": source_id}
