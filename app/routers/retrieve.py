"""
Retrieval router.

POST /retrieve — retrieve top-k RAG context chunks for a question.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.auth import verify_api_key
from app.models.retrieve import ChunkResult, RetrieveRequest, RetrieveResponse
from app.services.retriever import ChunkResult as ServiceChunk
from app.services.retriever import fetch_context

router = APIRouter(prefix="/retrieve", tags=["retrieve"])
logger = logging.getLogger(__name__)


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

    logger.info(
        f"[retrieve] notebooks={notebook_ids} k={req.k} "
        f"question={req.question[:80]!r}"
    )

    try:
        chunks: list[ServiceChunk]
        rewritten: str
        chunks, rewritten = fetch_context(
            notebook_ids=notebook_ids,
            question=req.question,
            k=req.k,
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
        logger.error(
            f"[retrieve] Failed for notebooks={notebook_ids}: {exc}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retrieval failed: {exc}",
        ) from exc
