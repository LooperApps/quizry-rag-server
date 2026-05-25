"""
RAG retrieval pipeline.

Ports the retrieval logic from week5/pro_implementation/answer.py with these
production enhancements:
  - Multi-notebook support (for research feature with vectorStoreIds)
  - ChromaDB metadata-based isolation by notebookId
  - Graceful degradation when rewrite or rerank fails
  - Efficient pre-count to avoid ChromaDB "n_results > index size" errors
"""

import logging
import time

from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from app.db.chroma import get_collection, count_for_notebooks, _build_where
from app.services.embedder import embed_query
from app.services.rewriter import rewrite_query
from app.config import settings

logger = logging.getLogger(__name__)


# ─── Result model (internal — mirrored by the router's response model) ────────

class ChunkResult(BaseModel):
    content: str
    source: str    # original file name
    sourceId: str  # Firestore source document ID


# ─── Reranker ─────────────────────────────────────────────────────────────────

class _RankOrder(BaseModel):
    order: list[int] = Field(
        description="Chunk IDs ordered from most to least relevant (1-based)"
    )


@retry(
    wait=wait_exponential(multiplier=1, min=5, max=60),
    stop=stop_after_attempt(2),
    reraise=True,
)
def _rerank(question: str, chunks: list[ChunkResult]) -> list[ChunkResult]:
    """Re-rank `chunks` by relevance to `question` using the LLM."""
    if len(chunks) <= 1:
        return chunks

    system_prompt = (
        "You are a document re-ranker for an educational app. "
        "Rank the provided text chunks by relevance to the student's question, "
        "most relevant first. Include ALL chunk IDs. "
        "Reply only with the JSON object {\"order\": [1, 3, 2, ...]}."
    )

    user_lines = [f"Question: {question}\n\nChunks:\n"]
    for i, chunk in enumerate(chunks, 1):
        # Truncate each chunk preview to keep the prompt manageable
        preview = chunk.content[:400].replace("\n", " ")
        user_lines.append(f"# CHUNK ID: {i}\n{preview}\n")

    response = completion(
        model=settings.litellm_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "\n".join(user_lines)},
        ],
        response_format=_RankOrder,
    )

    order = _RankOrder.model_validate_json(response.choices[0].message.content).order
    # Defensive: ignore out-of-range or duplicate indices
    seen: set[int] = set()
    reranked: list[ChunkResult] = []
    for idx in order:
        if 1 <= idx <= len(chunks) and idx not in seen:
            reranked.append(chunks[idx - 1])
            seen.add(idx)
    # Append any chunks missed by the model (shouldn't happen, but safety net)
    for i, chunk in enumerate(chunks, 1):
        if i not in seen:
            reranked.append(chunk)

    return reranked


# ─── ChromaDB query ───────────────────────────────────────────────────────────

def _query_chroma(
    notebook_ids: list[str],
    query_vector: list[float],
    k: int,
) -> list[ChunkResult]:
    """
    Query ChromaDB for the top-k chunks matching `query_vector`,
    filtered to the given notebook IDs.
    """
    collection = get_collection()
    where = _build_where(notebook_ids)

    # Count first to avoid ChromaDB error when n_results > collection size
    matching_ids = collection.get(where=where, include=[])["ids"]
    n = len(matching_ids)
    if n == 0:
        return []

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=min(k, n),
        where=where,
        include=["documents", "metadatas"],
    )

    chunks: list[ChunkResult] = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    for doc, meta in zip(docs, metas):
        chunks.append(
            ChunkResult(
                content=doc,
                source=meta.get("source", ""),
                sourceId=meta.get("sourceId", ""),
            )
        )
    return chunks


# ─── Merge / deduplication ────────────────────────────────────────────────────

def _merge_deduplicate(
    primary: list[ChunkResult],
    secondary: list[ChunkResult],
) -> list[ChunkResult]:
    """Merge two result lists, removing duplicates by content."""
    seen = {c.content for c in primary}
    merged = list(primary)
    for chunk in secondary:
        if chunk.content not in seen:
            seen.add(chunk.content)
            merged.append(chunk)
    return merged


# ─── Public API ───────────────────────────────────────────────────────────────

def fetch_context(
    notebook_ids: list[str],
    question: str,
    k: int | None = None,
    history: list[dict] | None = None,
) -> tuple[list[ChunkResult], str]:
    """
    Full RAG retrieval pipeline:
      1. Rewrite query for better recall
      2. Dual-vector search (original + rewritten)
      3. Merge & deduplicate
      4. LLM re-rank
      5. Return top-k chunks + rewritten query

    Returns (chunks, rewritten_query).
    Gracefully degrades: if rewrite or rerank fails, falls back to simpler step.
    """
    final_k = k or settings.final_k
    retrieval_k = settings.retrieval_k
    t_total = time.perf_counter()

    logger.info(
        f"[retriever] START notebooks={notebook_ids} question={question!r} "
        f"retrieval_k={retrieval_k} final_k={final_k}"
    )

    # Step 0: Fast-path — nothing ingested yet for these notebooks
    t0 = time.perf_counter()
    total = count_for_notebooks(notebook_ids)
    logger.info(f"[retriever] step=count docs_in_index={total} elapsed={_ms(t0)}ms")
    if total == 0:
        logger.info(f"[retriever] No documents indexed for notebooks {notebook_ids}")
        return [], question

    # Step 1: Query rewrite (translates Arabic → Hebrew to match Hebrew KB content)
    rewritten = question
    try:
        t1 = time.perf_counter()
        rewritten = rewrite_query(question, history)
        logger.info(
            f"[retriever] step=rewrite elapsed={_ms(t1)}ms "
            f"original={question!r} rewritten={rewritten!r}"
        )
    except Exception as exc:
        logger.warning(f"[retriever] step=rewrite FAILED, using original: {exc}")

    # Step 2: Embed the rewritten query
    t2 = time.perf_counter()
    vec = embed_query(rewritten)
    logger.info(f"[retriever] step=embed elapsed={_ms(t2)}ms dims={len(vec)}")

    # Step 3: ChromaDB similarity search
    t3 = time.perf_counter()
    chunks = _query_chroma(notebook_ids, vec, retrieval_k)
    logger.info(
        f"[retriever] step=chroma_query elapsed={_ms(t3)}ms "
        f"retrieved={len(chunks)} sources={list({c.source for c in chunks})}"
    )

    if not chunks:
        logger.info(f"[retriever] No chunks returned from ChromaDB")
        return [], rewritten

    # Step 4: Trim to final_k (ChromaDB already orders by cosine distance)
    result = chunks[:final_k]
    logger.info(
        f"[retriever] DONE total_elapsed={_ms(t_total)}ms "
        f"returned={len(result)}/{len(chunks)} chunks"
    )
    return result, rewritten


def _ms(t: float) -> int:
    """Milliseconds since a perf_counter snapshot."""
    return int((time.perf_counter() - t) * 1000)
