"""
Retrieval router.

POST /retrieve — retrieve top-k RAG context chunks for a question.
"""

import logging
import time

from fastapi import APIRouter, Depends, HTTPException, status
from starlette.concurrency import run_in_threadpool

from app.apps import App, CollectionAccessError, resolve_collection
from app.auth import resolve_app
from app.db.chroma import collection_exists
from app.log_utils import get_trace_id, log_step, log_step_data
from app.models.retrieve import ChunkResult, RetrieveRequest, RetrieveResponse
from app.services.retriever import ChunkResult as ServiceChunk
from app.services.retriever import fetch_context, fetch_multi

router = APIRouter(prefix="/retrieve", tags=["retrieve"])
logger = logging.getLogger(__name__)
LINE = "=" * 80


@router.post(
    "",
    response_model=RetrieveResponse,
    summary="Retrieve relevant context chunks",
    description=(
        "Performs query rewriting, dual-vector search, deduplication, and LLM re-ranking "
        "against the ChromaDB collection. Returns the top-k chunks for the given "
        "notebook(s).\n\n"
        "- Use `notebookId` for single-notebook queries (chat, summary, study guide, etc.).\n"
        "- Use `notebookIds` for multi-notebook research feature (`researchPlan` / `researchContent`).\n"
        "- `sourceIds` adds chunks of those sources (OR-ed with the notebooks).\n"
        "- `queries` switches to multi-query retrieval (one embedding batch, one search, "
        "round-robin merge, no rewrite/rerank)."
    ),
)
async def retrieve(
    req: RetrieveRequest,
    rag_app: App = Depends(resolve_app),
) -> RetrieveResponse:
    # The model validator always populates both lists (either may be empty).
    notebook_ids: list[str] = req.notebookIds or []
    source_ids: list[str] = req.sourceIds or []
    t_start = time.perf_counter()
    trace = get_trace_id()

    try:
        collection = resolve_collection(rag_app, req.collection)
    except CollectionAccessError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    if collection != rag_app.collection and not await run_in_threadpool(collection_exists, collection):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No collection {collection!r}")

    logger.info(
        f"\n{LINE}\n"
        f"[RETRIEVE-ROUTER] trace={trace} ENTER\n"
        f"app={rag_app.id} collection={collection}\n"
        f"notebook_ids={notebook_ids!r}\n"
        f"source_ids={source_ids!r}\n"
        f"k={req.k}\n"
        f"question={req.question!r}\n"
        f"queries={req.queries!r}\n"
        f"{LINE}"
    )

    try:
        chunks: list[ServiceChunk]
        rewritten: str | None
        # Blocking work (embedding, Chroma, rewrite/rerank LLM calls) runs in
        # the threadpool so concurrent requests are not serialised.
        if req.queries:
            chunks = await run_in_threadpool(
                fetch_multi,
                notebook_ids=notebook_ids,
                queries=req.queries,
                k=req.k,
                collection=collection,
                source_ids=source_ids,
            )
            rewritten = None
        else:
            chunks, rewritten = await run_in_threadpool(
                fetch_context,
                notebook_ids=notebook_ids,
                question=req.question,
                k=req.k,
                history=req.history,
                collection=collection,
                source_ids=source_ids,
                rewrite=req.rewrite,
            )

        total_ms = int((time.perf_counter() - t_start) * 1000)
        rewritten_log = repr(rewritten) if rewritten and rewritten != req.question else "(unchanged)"

        # Log each returned chunk in detail
        chunks_detail = "\n".join(
            f"  chunk[{i}] source={c.source!r} sourceId={c.sourceId!r}\n"
            f"    content[:200]={c.content[:200]!r}"
            for i, c in enumerate(chunks)
        )
        logger.info(
            f"\n{LINE}\n"
            f"[RETRIEVE-ROUTER] trace={trace} RESPONSE\n"
            f"total_chunks={len(chunks)}\n"
            f"rewritten_query={rewritten_log}\n"
            f"total_elapsed={total_ms}ms\n"
            f"--- CHUNKS ---\n{chunks_detail}\n"
            f"{LINE}"
        )

        return RetrieveResponse(
            chunks=[
                ChunkResult(
                    content=c.content,
                    source=c.source,
                    sourceId=c.sourceId,
                    notebookId=c.notebookId,
                    pageStart=c.pageStart,
                    pageEnd=c.pageEnd,
                    distance=c.distance,
                )
                for c in chunks
            ],
            rewrittenQuery=rewritten if rewritten != req.question else None,
        )
    except Exception as exc:
        total_ms = int((time.perf_counter() - t_start) * 1000)
        logger.error(
            f"\n{LINE}\n"
            f"[RETRIEVE-ROUTER] trace={trace} FAILED elapsed={total_ms}ms\n"
            f"error={exc}\n"
            f"{LINE}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retrieval failed: {exc}",
        ) from exc
