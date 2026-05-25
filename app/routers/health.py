import logging

from fastapi import APIRouter

from app.db.chroma import get_collection

router = APIRouter(tags=["health"])
logger = logging.getLogger(__name__)

_DEBUG_NOTEBOOK = "university_requirements"
_DEBUG_QUERY = "علم الحاسوب التخنيون"


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


@router.get("/debug/university")
async def debug_university() -> dict:
    """No-auth test: query university_requirements and return chunks + final LLM answer."""
    from app.services.retriever import fetch_context
    from litellm import completion

    logger.info(f"[debug] university test: {_DEBUG_QUERY!r}")
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

        response = completion(
            model="gemini/gemini-2.5-flash",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": _DEBUG_QUERY},
            ],
        )
        answer = response.choices[0].message.content.strip()

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
        logger.error(f"[debug] university test failed: {exc}", exc_info=True)
        return {"error": str(exc)}


@router.get("/debug/reset")
async def debug_reset() -> dict:
    """Wipe ALL documents from ChromaDB (drops and recreates the collection)."""
    from app.db.chroma import reset_collection
    deleted = reset_collection()
    return {"status": "ok", "deletedDocs": deleted}
