"""
Retrieval router.

POST /retrieve — retrieve top-k RAG context chunks for a question.
"""

import logging
import time

from fastapi import APIRouter, Depends, HTTPException, status

from app.auth import verify_api_key
from app.log_utils import get_trace_id, log_step, log_step_data
from app.models.retrieve import ChunkResult, RetrieveRequest, RetrieveResponse
from app.services.retriever import ChunkResult as ServiceChunk
from app.services.retriever import fetch_context

router = APIRouter(prefix="/retrieve", tags=["retrieve"])
logger = logging.getLogger(__name__)
LINE = "=" * 80


@router.post(
    "",
    response_model=RetrieveResponse,
    dependencies=[Depends(verify_api_key)],
    summary="Retrieve relevant context chunks",
    description=(
        "Performs query rewriting, dual-vector search, deduplication, and LLM re-ranking "
        "against the ChromaDB collection. Returns the top-k chunks for the given "
        "notebook(s).\n\n"
        "- Use `notebookId` for single-notebook queries (chat, summary, study guide, etc.).\n"
        "- Use `notebookIds` for multi-notebook research feature (`researchPlan` / `researchContent`)."
    ),
)
async def retrieve(req: RetrieveRequest) -> RetrieveResponse:
    # req.notebookIds is always populated by the model validator
    notebook_ids: list[str] = req.notebookIds  # type: ignore[assignment]
    t_start = time.perf_counter()
    trace = get_trace_id()

    logger.info(
        f"\n{LINE}\n"
        f"[RETRIEVE-ROUTER] trace={trace} ENTER\n"
        f"notebook_ids={notebook_ids!r}\n"
        f"k={req.k}\n"
        f"question={req.question!r}\n"
        f"{LINE}"
    )

    try:
        chunks: list[ServiceChunk]
        rewritten: str
        chunks, rewritten = fetch_context(
            notebook_ids=notebook_ids,
            question=req.question,
            k=req.k,
        )

        total_ms = int((time.perf_counter() - t_start) * 1000)
        rewritten_log = repr(rewritten) if rewritten != req.question else "(unchanged)"

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
