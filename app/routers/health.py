import logging

from fastapi import APIRouter

from app.db.chroma import get_collection
from app.log_utils import get_trace_id, log_prompt, log_step, log_step_data

router = APIRouter(tags=["health"])
logger = logging.getLogger(__name__)
LINE = "=" * 80

_DEBUG_NOTEBOOK = "university_requirements"
_DEBUG_QUERY = "علم الحاسوب التخنيون"


@router.get("/health")
async def health() -> dict:
    """
    Render health-check endpoint.
    Returns 200 OK when the server and ChromaDB are operational.
    """
    trace = get_trace_id()
    log_step(logger, "HEALTH", "Health check requested")
    try:
        col = get_collection()
        doc_count = col.count()
        log_step(logger, "HEALTH", f"ChromaDB connected, totalDocs={doc_count}")
        return {"status": "ok", "chromadb": "connected", "totalDocs": doc_count}
    except Exception as exc:
        log_step(logger, "HEALTH", f"ChromaDB degraded: {exc}")
        return {"status": "degraded", "chromadb": str(exc)}


@router.get("/debug/university")
async def debug_university() -> dict:
    """No-auth test: query university_requirements and return chunks + final LLM answer."""
    from app.services.retriever import fetch_context
    from litellm import completion

    trace = get_trace_id()
    logger.info(
        f"\n{LINE}\n"
        f"[DEBUG_UNIVERSITY] trace={trace} START query={_DEBUG_QUERY!r}\n"
        f"{LINE}"
    )
    try:
        chunks, rewritten = fetch_context(
            notebook_ids=[_DEBUG_NOTEBOOK],
            question=_DEBUG_QUERY,
        )

        # Build context string from retrieved chunks
        context = "\n\n---\n\n".join(c.content for c in chunks)

        system_prompt = (
            "أنت مساعد ذكي مفيد. أجب على أسئلة المستخدم بناءً على المحتوى المرجعي المتوفر.\n\n"
            f"# المحتوى المرجعي:\n{context}"
        )

        log_step(
            logger, "DEBUG_UNIVERSITY",
            f"Calling Gemini for final answer with {len(chunks)} chunks context",
        )
        log_prompt(
            logger, "debug_university (Gemini answer)",
            prompt=f"System: {system_prompt}\n\nUser: {_DEBUG_QUERY}",
            model="gemini/gemini-2.5-flash",
        )

        response = completion(
            model="gemini/gemini-2.5-flash",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": _DEBUG_QUERY},
            ],
        )
        answer = response.choices[0].message.content.strip()

        logger.info(
            f"\n{LINE}\n"
            f"[DEBUG_UNIVERSITY] trace={trace} DONE\n"
            f"chunks_retrieved={len(chunks)}\n"
            f"rewritten_query={rewritten!r}\n"
            f"answer_preview={answer[:500]!r}\n"
            f"{LINE}"
        )

        return {
            "notebook": _DEBUG_NOTEBOOK,
            "query": _DEBUG_QUERY,
            "rewrittenQuery": rewritten if rewritten != _DEBUG_QUERY else None,
            "chunkCount": len(chunks),
            "chunks": [
                {
                    "source": c.source,
                    "sourceId": c.sourceId,
                    "content": c.content[:400],
                }
                for c in chunks
            ],
            "answer": answer,
        }
    except Exception as exc:
        logger.error(
            f"\n{LINE}\n"
            f"[DEBUG_UNIVERSITY] trace={trace} FAILED: {exc}\n"
            f"{LINE}",
            exc_info=True,
        )
        return {"error": str(exc)}


@router.get("/debug/reset")
async def debug_reset() -> dict:
    """Wipe ALL documents from ChromaDB (drops and recreates the collection)."""
    from app.db.chroma import reset_collection
    trace = get_trace_id()
    log_step(logger, "DEBUG_RESET", "Resetting ChromaDB collection")
    deleted = reset_collection()
    log_step(logger, "DEBUG_RESET", f"Deleted {deleted} documents, collection recreated")
    return {"status": "ok", "deletedDocs": deleted}
