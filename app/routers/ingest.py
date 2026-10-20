"""
Ingestion router.

POST   /ingest                               — ingest a source file (sync, returns after done)
POST   /ingest/text                          — index page text extracted elsewhere
DELETE /ingest/notebooks/{notebookId}        — delete all chunks for a notebook
DELETE /ingest/notebooks/{notebookId}/sources/{sourceId} — delete chunks for one source
"""

import logging
import time

from fastapi import APIRouter, Depends, HTTPException, status
from starlette.concurrency import run_in_threadpool

from app.apps import App, CollectionAccessError, resolve_collection
from app.auth import resolve_app
from app.db.chroma import MaintenanceError
from app.log_utils import get_trace_id, redact_url
from app.models.ingest import IngestRequest, IngestResponse, IngestTextRequest, IngestTextResponse
from app.services.ingest_pipeline import (
    SourceConflictError,
    _delete_source_chunks,
    delete_notebook_chunks,
    ingest_text,
    run_ingest,
)

router = APIRouter(prefix="/ingest", tags=["ingest"])
logger = logging.getLogger(__name__)
LINE = "=" * 80


@router.post(
    "",
    response_model=IngestResponse,
    status_code=status.HTTP_200_OK,
    summary="Ingest a source file into ChromaDB",
    description=(
        "Downloads the file from `storageUrl`, extracts text, chunks it via LLM, "
        "embeds the chunks, and stores them in ChromaDB keyed by `notebookId` + `sourceId`. "
        "This call is **synchronous** — it returns only after ingestion completes. "
        "The Cloud Function should set the source status to `ready` on 200, `error` on 5xx."
    ),
)
async def ingest(
    req: IngestRequest,
    rag_app: App = Depends(resolve_app),
) -> IngestResponse:
    trace = get_trace_id()
    t_start = time.perf_counter()
    logger.info(
        f"\n{LINE}\n"
        f"[INGEST-ROUTER] trace={trace} ENTER\n"
        f"app={rag_app.id} collection={rag_app.collection}\n"
        f"notebook_id={req.notebookId!r}\n"
        f"source_id={req.sourceId!r}\n"
        f"file_name={req.fileName!r}\n"
        f"storage_url={redact_url(req.storageUrl)!r}\n"
        f"file_type={req.fileType!r}\n"
        f"{LINE}"
    )
    try:
        chunk_count = await run_ingest(
            notebook_id=req.notebookId,
            source_id=req.sourceId,
            storage_url=req.storageUrl,
            file_name=req.fileName,
            collection=rag_app.collection,
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
    except SourceConflictError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except MaintenanceError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
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


@router.post(
    "/text",
    response_model=IngestTextResponse,
    summary="Index page text that was extracted elsewhere",
    description=(
        "Chunks the given pages page-aware (pages_v1), embeds them with the target "
        "collection's flags and upserts them under deterministic ids "
        "`{sourceId}:{chunkIndex:05d}`, then deletes the source's leftover ids. "
        "Idempotent: re-sending the same source converges on the same chunks. "
        "`collection` may name the app's live collection or one of its "
        "`<live>__<suffix>` staging collections (created on first use)."
    ),
)
async def ingest_text_route(
    req: IngestTextRequest,
    rag_app: App = Depends(resolve_app),
) -> IngestTextResponse:
    trace = get_trace_id()
    t_start = time.perf_counter()
    try:
        collection = resolve_collection(rag_app, req.collection)
    except CollectionAccessError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    logger.info(
        f"[INGEST-ROUTER] trace={trace} TEXT app={rag_app.id} collection={collection} "
        f"notebook_id={req.notebookId!r} source_id={req.sourceId!r} file_name={req.fileName!r} "
        f"pages={len(req.pages)}"
    )
    try:
        chunk_count, page_count = await ingest_text(
            notebook_id=req.notebookId,
            source_id=req.sourceId,
            file_name=req.fileName,
            pages=[(p.page, p.text) for p in req.pages],
            collection=collection,
        )
    except SourceConflictError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except MaintenanceError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except ValueError as exc:
        logger.warning(f"[INGEST-ROUTER] trace={trace} TEXT VALIDATION_ERROR error={exc}")
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(f"[INGEST-ROUTER] trace={trace} TEXT FAILED error={exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {exc}",
        ) from exc
    logger.info(
        f"[INGEST-ROUTER] trace={trace} TEXT SUCCESS source_id={req.sourceId!r} chunks={chunk_count} "
        f"elapsed={int((time.perf_counter() - t_start) * 1000)}ms"
    )
    return IngestTextResponse(
        status="ok",
        sourceId=req.sourceId,
        chunkCount=chunk_count,
        pageCount=page_count,
        collection=collection,
    )


@router.delete(
    "/notebooks/{notebook_id}",
    summary="Delete all chunks for a notebook",
    description="Called by onNotebookDeleted Cloud Function trigger.",
)
async def delete_notebook(
    notebook_id: str,
    rag_app: App = Depends(resolve_app),
) -> dict:
    trace = get_trace_id()
    logger.info(
        f"[INGEST-ROUTER] trace={trace} DELETE app={rag_app.id} notebook={notebook_id!r}"
    )
    await run_in_threadpool(delete_notebook_chunks, notebook_id, rag_app.collection)
    logger.info(f"[INGEST-ROUTER] trace={trace} DELETE notebook={notebook_id!r} DONE")
    return {"status": "ok", "notebookId": notebook_id}


@router.delete(
    "/notebooks/{notebook_id}/sources/{source_id}",
    summary="Delete chunks for one source file",
    description="Called when a source document is removed from a notebook.",
)
async def delete_source(
    notebook_id: str,
    source_id: str,
    rag_app: App = Depends(resolve_app),
) -> dict:
    trace = get_trace_id()
    logger.info(
        f"[INGEST-ROUTER] trace={trace} DELETE SOURCE app={rag_app.id} "
        f"notebook={notebook_id!r} source={source_id!r}"
    )
    await run_in_threadpool(_delete_source_chunks, notebook_id, source_id, rag_app.collection)
    logger.info(
        f"[INGEST-ROUTER] trace={trace} DELETE SOURCE notebook={notebook_id!r} source={source_id!r} DONE"
    )
    return {"status": "ok", "notebookId": notebook_id, "sourceId": source_id}
