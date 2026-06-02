"""
Ingestion router.

POST   /ingest                               — ingest a source file (sync, returns after done)
DELETE /ingest/notebooks/{notebookId}        — delete all chunks for a notebook
DELETE /ingest/notebooks/{notebookId}/sources/{sourceId} — delete chunks for one source
"""

import logging
import time

from fastapi import APIRouter, Depends, HTTPException, status

from app.auth import verify_api_key
from app.log_utils import get_trace_id
from app.models.ingest import IngestRequest, IngestResponse
from app.services.ingest_pipeline import (
    _delete_source_chunks,
    delete_notebook_chunks,
    run_ingest,
)

router = APIRouter(prefix="/ingest", tags=["ingest"])
logger = logging.getLogger(__name__)
LINE = "=" * 80


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
    trace = get_trace_id()
    t_start = time.perf_counter()
    logger.info(
        f"\n{LINE}\n"
        f"[INGEST-ROUTER] trace={trace} ENTER\n"
        f"notebook_id={req.notebookId!r}\n"
        f"source_id={req.sourceId!r}\n"
        f"file_name={req.fileName!r}\n"
        f"storage_url={req.storageUrl!r}\n"
        f"file_type={req.fileType!r}\n"
        f"{LINE}"
    )
    try:
        chunk_count = await run_ingest(
            notebook_id=req.notebookId,
            source_id=req.sourceId,
            storage_url=req.storageUrl,
            file_name=req.fileName,
        )
        elapsed_ms = int((time.perf_counter() - t_start) * 1000)
        logger.info(
            f"\n{LINE}\n"
            f"[INGEST-ROUTER] trace={trace} SUCCESS\n"
            f"chunks_stored={chunk_count}\n"
            f"elapsed={elapsed_ms}ms\n"
            f"{LINE}"
        )
        return IngestResponse(
            status="ok",
            sourceId=req.sourceId,
            chunkCount=chunk_count,
        )
    except ValueError as exc:
        elapsed_ms = int((time.perf_counter() - t_start) * 1000)
        logger.warning(
            f"\n{LINE}\n"
            f"[INGEST-ROUTER] trace={trace} VALIDATION_ERROR elapsed={elapsed_ms}ms\n"
            f"error={exc}\n"
            f"{LINE}"
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        elapsed_ms = int((time.perf_counter() - t_start) * 1000)
        logger.error(
            f"\n{LINE}\n"
            f"[INGEST-ROUTER] trace={trace} FAILED elapsed={elapsed_ms}ms\n"
            f"error={exc}\n"
            f"{LINE}",
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
    trace = get_trace_id()
    logger.info(f"[INGEST-ROUTER] trace={trace} DELETE notebook={notebook_id!r}")
    delete_notebook_chunks(notebook_id)
    logger.info(f"[INGEST-ROUTER] trace={trace} DELETE notebook={notebook_id!r} DONE")
    return {"status": "ok", "notebookId": notebook_id}


@router.delete(
    "/notebooks/{notebook_id}/sources/{source_id}",
    dependencies=[Depends(verify_api_key)],
    summary="Delete chunks for one source file",
    description="Called when a source document is removed from a notebook.",
)
async def delete_source(notebook_id: str, source_id: str) -> dict:
    trace = get_trace_id()
    logger.info(
        f"[INGEST-ROUTER] trace={trace} DELETE SOURCE "
        f"notebook={notebook_id!r} source={source_id!r}"
    )
    _delete_source_chunks(notebook_id, source_id)
    logger.info(
        f"[INGEST-ROUTER] trace={trace} DELETE SOURCE notebook={notebook_id!r} source={source_id!r} DONE"
    )
    return {"status": "ok", "notebookId": notebook_id, "sourceId": source_id}
