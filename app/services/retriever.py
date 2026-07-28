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

from litellm import completion
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from app.db.chroma import get_collection, count_for_notebooks, _build_where
from app.log_utils import (
    get_trace_id,
    log_chroma_query,
    log_prompt,
    log_step,
    log_step_data,
)
from app.services.embedder import embed_query
from app.services.rewriter import rewrite_query
from app.config import settings

logger = logging.getLogger(__name__)
LINE = "=" * 80


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
    wait=wait_exponential(multiplier=1, min=1, max=4),
    stop=stop_after_attempt(2),
    reraise=True,
)
def _rerank(question: str, chunks: list[ChunkResult]) -> list[ChunkResult]:
    """Re-rank `chunks` by relevance to `question` using the LLM."""
    if len(chunks) <= 1:
        return chunks

    trace = get_trace_id()
    log_step(logger, "RERANK", f"Re-ranking {len(chunks)} chunks for question={question!r}")

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

    full_prompt = "\n".join([system_prompt] + user_lines)

    t0 = time.perf_counter()
    # Plain json_object mode: DeepSeek does not support json_schema-style
    # structured outputs, so passing the pydantic model made every call fail
    # (then retry after a 5s+ backoff) and rerank silently degrade.
    response = completion(
        model=settings.litellm_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "\n".join(user_lines)},
        ],
        response_format={"type": "json_object"},
    )

    raw_response = response.choices[0].message.content
    elapsed = int((time.perf_counter() - t0) * 1000)

    log_prompt(
        logger, "_rerank",
        prompt=full_prompt,
        response=raw_response,
        model=settings.litellm_model,
        elapsed_ms=elapsed,
    )

    order = _RankOrder.model_validate_json(raw_response).order
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

    log_step(
        logger, "RERANK",
        f"Done — input={len(chunks)} chunks, output={len(reranked)} chunks, "
        f"order={order}, elapsed={elapsed}ms",
    )
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
    trace = get_trace_id()
    collection = get_collection()
    where = _build_where(notebook_ids)
    log_step(logger, "CHROMA_QUERY", f"notebook_ids={notebook_ids!r} k={k} where={where!r}")

    # Count first to avoid ChromaDB error when n_results > collection size
    matching_ids = collection.get(where=where, include=[])["ids"]
    n = len(matching_ids)
    log_step(logger, "CHROMA_QUERY", f"Matching docs in collection: {n}")
    if n == 0:
        log_step(logger, "CHROMA_QUERY", "No matching docs found — returning empty")
        return []

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=min(k, n),
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    # Log the full ChromaDB query result
    log_chroma_query(
        logger, "_query_chroma",
        notebook_ids=notebook_ids,
        query_vector=query_vector,
        n_results=min(k, n),
        where_filter=where,
        results=results,
    )

    chunks: list[ChunkResult] = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]
    for i, (doc, meta) in enumerate(zip(docs, metas)):
        dist = dists[i] if i < len(dists) else "?"
        chunks.append(
            ChunkResult(
                content=doc,
                source=meta.get("source", ""),
                sourceId=meta.get("sourceId", ""),
            )
        )
        logger.debug(
            f"[CHROMA_QUERY] trace={trace} result[{i}] "
            f"distance={dist if isinstance(dist, str) else f'{dist:.4f}'} "
            f"source={meta.get('source', '')!r} "
            f"sourceId={meta.get('sourceId', '')!r}"
        )

    return chunks


# ─── Merge / deduplication ────────────────────────────────────────────────────

def _merge_deduplicate(
    primary: list[ChunkResult],
    secondary: list[ChunkResult],
) -> list[ChunkResult]:
    """Merge two result lists, removing duplicates by content."""
    trace = get_trace_id()
    seen = {c.content for c in primary}
    merged = list(primary)
    deduped = 0
    for chunk in secondary:
        if chunk.content not in seen:
            seen.add(chunk.content)
            merged.append(chunk)
        else:
            deduped += 1
    log_step(
        logger, "MERGE_DEDUP",
        f"primary={len(primary)} secondary={len(secondary)} "
        f"deduped={deduped} final={len(merged)}",
    )
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
      2. Embed the (rewritten) query
      3. ChromaDB similarity search
      4. LLM re-rank
      5. Return top-k chunks + rewritten query

    Returns (chunks, rewritten_query).
    Gracefully degrades: if rewrite or rerank fails, falls back to simpler step.
    """
    final_k = k or settings.final_k
    retrieval_k = settings.retrieval_k
    t_total = time.perf_counter()
    trace = get_trace_id()

    logger.info(
        f"\n{LINE}\n"
        f"[FETCH_CONTEXT] trace={trace} START\n"
        f"notebook_ids={notebook_ids!r}\n"
        f"question={question!r}\n"
        f"retrieval_k={retrieval_k}\n"
        f"final_k={final_k}\n"
        f"history_provided={history is not None}\n"
        f"{LINE}"
    )

    # Step 0: Fast-path — nothing ingested yet for these notebooks
    t0 = time.perf_counter()
    total = count_for_notebooks(notebook_ids)
    log_step(
        logger, "STEP0_COUNT",
        f"docs_in_index={total} elapsed={_ms(t0)}ms",
    )
    if total == 0:
        log_step(logger, "STEP0_COUNT", f"No documents indexed for notebooks {notebook_ids} — returning empty")
        return [], question

    # Step 1: Query rewrite (multilingual keyword query matching the KB language)
    rewritten = question
    try:
        t1 = time.perf_counter()
        rewritten = rewrite_query(question, history)
        log_step(
            logger, "STEP1_REWRITE",
            f"original={question!r} rewritten={rewritten!r} elapsed={_ms(t1)}ms",
        )
    except Exception as exc:
        log_step(logger, "STEP1_REWRITE", f"FAILED, using original: {exc}")

    # Step 2: Embed the rewritten query
    t2 = time.perf_counter()
    vec = embed_query(rewritten)
    log_step(
        logger, "STEP2_EMBED",
        f"text={rewritten!r} dims={len(vec)} elapsed={_ms(t2)}ms",
    )

    # Step 3: ChromaDB similarity search
    t3 = time.perf_counter()
    chunks = _query_chroma(notebook_ids, vec, retrieval_k)
    log_step(
        logger, "STEP3_CHROMA_SEARCH",
        f"retrieved={len(chunks)} sources={list({c.source for c in chunks})} elapsed={_ms(t3)}ms",
    )

    if not chunks:
        log_step(logger, "STEP3_CHROMA_SEARCH", "No chunks returned from ChromaDB")
        return [], rewritten

    # Step 4: Re-rank via LLM (graceful degradation on failure).
    # Skipped when everything retrieved fits in the final result anyway —
    # the LLM call would only reorder chunks that all get returned.
    if len(chunks) <= final_k:
        log_step(
            logger, "STEP4_RERANK",
            f"SKIPPED — retrieved {len(chunks)} <= final_k {final_k}",
        )
        reranked = chunks
    else:
        try:
            t4 = time.perf_counter()
            reranked = _rerank(rewritten, chunks)
            log_step(
                logger, "STEP4_RERANK",
                f"input={len(chunks)} output={len(reranked)} elapsed={_ms(t4)}ms",
            )
        except Exception as exc:
            log_step(logger, "STEP4_RERANK", f"FAILED, using ChromaDB ordering: {exc}")
            reranked = chunks

    # Step 5: Trim to final_k
    result = reranked[:final_k]
    log_step(
        logger, "STEP5_TRIM",
        f"reranked={len(reranked)} final={len(result)} chunks",
    )

    # Log final result summary
    chunks_detail = "\n".join(
        f"  chunk[{i}] source={c.source!r} sourceId={c.sourceId!r}\n"
        f"    content_preview={c.content[:200]!r}"
        for i, c in enumerate(result)
    )
    logger.info(
        f"\n{LINE}\n"
        f"[FETCH_CONTEXT] trace={trace} DONE\n"
        f"total_elapsed={_ms(t_total)}ms\n"
        f"returned_chunks={len(result)}\n"
        f"rewritten_query={rewritten!r}\n"
        f"--- FINAL CHUNKS ---\n{chunks_detail}\n"
        f"{LINE}"
    )

    return result, rewritten


def _ms(t: float) -> int:
    """Milliseconds since a perf_counter snapshot."""
    return int((time.perf_counter() - t) * 1000)
