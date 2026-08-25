"""
Admin router - low-level access to the ChromaDB collection.

Exists because the vector store is an embedded PersistentClient on this
process's disk: nothing outside this container can see a chunk, count a
notebook or trace a query. The Node manager (quizry-db-manager) is built
entirely on these endpoints.

Everything here is read-only except the explicit DELETE routes. All routes sit
behind `verify_admin_key`, which prefers ADMIN_API_KEY and falls back to
RAG_API_KEY so an existing deployment keeps working without new config.

    GET    /admin/stats                       collection + disk + embedding info
    GET    /admin/notebooks                   notebook -> source tree with counts
    GET    /admin/notebooks/{id}              one notebook, with its sources
    GET    /admin/chunks                      paginated chunk browser + filters
    GET    /admin/chunks/{chunk_id}           one chunk, optionally with vector
    GET    /admin/chunks/{chunk_id}/similar   nearest neighbours of a chunk
    POST   /admin/query                       traced retrieval (what gets found)
    DELETE /admin/chunks                      delete explicit chunk ids
    DELETE /admin/notebooks/{id}              delete a notebook's chunks
    DELETE /admin/notebooks/{id}/sources/{sid}  delete one source's chunks
"""

import logging
import time

from fastapi import APIRouter, Body, Depends, HTTPException, Query

from app.auth import verify_admin_key
from app.config import settings
from app.db.chroma import _build_where, get_collection
from app.log_utils import get_trace_id, log_step
from app.models.admin import (
    AdminQueryRequest,
    DeleteChunksRequest,
    DeleteResponse,
)
from app.services.admin_index import (
    col_get,
    disk_usage,
    embedding_info,
    get_snapshot,
    invalidate_snapshot,
)

# The embedder, rewriter and reranker are imported inside the handlers that use
# them. Only POST /admin/query touches an LLM; keeping those imports out of
# module scope means the inspection endpoints stay usable when a model provider
# is misconfigured, which is exactly when you want to look at the data.

router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(verify_admin_key)])
logger = logging.getLogger(__name__)

# Chunk text is unbounded; the browser sends previews unless asked for the body.
PREVIEW_CHARS = 400


def _where_for(notebook_id: str | None, source_id: str | None) -> dict | None:
    """Build a Chroma where-filter from the optional notebook/source filters."""
    clauses = []
    if notebook_id:
        clauses.append({"notebookId": {"$eq": notebook_id}})
    if source_id:
        clauses.append({"sourceId": {"$eq": source_id}})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _chunk_view(
    chunk_id: str,
    document: str | None,
    metadata: dict | None,
    full: bool,
    embedding: list | None = None,
) -> dict:
    text = document or ""
    meta = metadata or {}
    view = {
        "id": chunk_id,
        "notebookId": meta.get("notebookId", ""),
        "sourceId": meta.get("sourceId", ""),
        "source": meta.get("source", ""),
        "chars": len(text),
        "bytes": len(text.encode("utf-8")),
        "preview": text if full else text[:PREVIEW_CHARS],
        "truncated": (not full) and len(text) > PREVIEW_CHARS,
        "metadata": meta,
    }
    if full:
        view["content"] = text
    if embedding is not None:
        view["embedding"] = embedding
        view["embeddingDimensions"] = len(embedding)
    return view


# --- Read: stats --------------------------------------------------------------

@router.get("/stats", summary="Collection, disk and embedding statistics")
async def stats(refresh: bool = Query(False, description="Force a full rescan")) -> dict:
    col = get_collection()
    snapshot = get_snapshot(force=refresh)
    return {
        "collection": snapshot["collection"],
        "count": col.count(),
        "totals": snapshot["totals"],
        "snapshot": {
            "generatedAt": snapshot["generatedAt"],
            "elapsedMs": snapshot["elapsedMs"],
            "cached": snapshot["cached"],
            "ttlSeconds": 60,
        },
        "embedding": embedding_info(),
        "disk": disk_usage(),
        "config": {
            "chromaPath": settings.chroma_path,
            "embeddingModel": settings.embedding_model,
            "llmModel": settings.litellm_model,
            "chunkMode": settings.chunk_mode,
            "retrievalK": settings.retrieval_k,
            "finalK": settings.final_k,
        },
    }


# --- Read: notebooks ----------------------------------------------------------

@router.get("/notebooks", summary="Every notebook in the index, with its sources")
async def list_notebooks(
    refresh: bool = Query(False, description="Force a full rescan"),
    includeSources: bool = Query(True, description="Embed the per-source breakdown"),
) -> dict:
    snapshot = get_snapshot(force=refresh)
    notebooks = snapshot["notebooks"]
    if not includeSources:
        notebooks = [{k: v for k, v in nb.items() if k != "sources"} for nb in notebooks]
    return {
        "generatedAt": snapshot["generatedAt"],
        "cached": snapshot["cached"],
        "totals": snapshot["totals"],
        "notebooks": notebooks,
    }


@router.get("/notebooks/{notebook_id}", summary="One notebook and its sources")
async def get_notebook(
    notebook_id: str,
    refresh: bool = Query(False),
) -> dict:
    snapshot = get_snapshot(force=refresh)
    for nb in snapshot["notebooks"]:
        if nb["notebookId"] == notebook_id:
            return {"generatedAt": snapshot["generatedAt"], "notebook": nb}
    # Not an error: a notebook can exist in Firestore with nothing indexed yet.
    return {
        "generatedAt": snapshot["generatedAt"],
        "notebook": {
            "notebookId": notebook_id,
            "chunks": 0,
            "chars": 0,
            "bytes": 0,
            "avgChunkChars": 0,
            "minChunkChars": None,
            "maxChunkChars": None,
            "sourceCount": 0,
            "sources": [],
        },
    }


# --- Read: chunks -------------------------------------------------------------

@router.get("/chunks", summary="Browse chunks with filters and pagination")
async def list_chunks(
    notebookId: str | None = Query(None),
    sourceId: str | None = Query(None),
    contains: str | None = Query(None, description="Substring match on chunk text"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    full: bool = Query(False, description="Return whole chunk text, not a preview"),
) -> dict:
    where = _where_for(notebookId, sourceId)
    where_document = {"$contains": contains} if contains else None

    kwargs: dict = {"limit": limit, "offset": offset, "include": ["documents", "metadatas"]}
    if where:
        kwargs["where"] = where
    if where_document:
        kwargs["where_document"] = where_document

    t0 = time.perf_counter()
    page = col_get(**kwargs)

    # Chroma cannot count a filtered set without fetching it, so the total is a
    # second ids-only pass. Cheap relative to pulling documents.
    count_kwargs: dict = {"include": []}
    if where:
        count_kwargs["where"] = where
    if where_document:
        count_kwargs["where_document"] = where_document
    matched = len(col_get(**count_kwargs)["ids"])

    chunks = [
        _chunk_view(cid, page["documents"][i], page["metadatas"][i], full)
        for i, cid in enumerate(page["ids"])
    ]
    log_step(
        logger, "ADMIN_CHUNKS",
        f"notebookId={notebookId!r} sourceId={sourceId!r} contains={contains!r} "
        f"returned={len(chunks)} matched={matched} elapsed={int((time.perf_counter() - t0) * 1000)}ms",
    )
    return {
        "total": matched,
        "limit": limit,
        "offset": offset,
        "returned": len(chunks),
        "hasMore": offset + len(chunks) < matched,
        "filters": {"notebookId": notebookId, "sourceId": sourceId, "contains": contains},
        "chunks": chunks,
    }


@router.get("/chunks/{chunk_id}", summary="One chunk in full")
async def get_chunk(
    chunk_id: str,
    includeEmbedding: bool = Query(False, description="Include the raw vector"),
) -> dict:
    include = ["documents", "metadatas"] + (["embeddings"] if includeEmbedding else [])
    page = col_get(ids=[chunk_id], include=include)
    if not page["ids"]:
        raise HTTPException(status_code=404, detail=f"No chunk with id {chunk_id!r}")
    embedding = page["embeddings"][0] if includeEmbedding and page["embeddings"] else None
    return {
        "chunk": _chunk_view(
            page["ids"][0], page["documents"][0], page["metadatas"][0],
            full=True, embedding=embedding,
        )
    }


@router.get("/chunks/{chunk_id}/similar", summary="Nearest neighbours of a chunk")
async def similar_chunks(
    chunk_id: str,
    k: int = Query(10, ge=1, le=100),
    sameNotebook: bool = Query(True, description="Restrict to the chunk's own notebook"),
) -> dict:
    """
    Uses the chunk's stored vector, so no embedding call is made. Useful for
    spotting near-duplicate chunks produced by the ingest overlap.
    """
    page = col_get(ids=[chunk_id], include=["documents", "metadatas", "embeddings"])
    if not page["ids"]:
        raise HTTPException(status_code=404, detail=f"No chunk with id {chunk_id!r}")
    vector = page["embeddings"][0] if page["embeddings"] else None
    if not vector:
        raise HTTPException(status_code=409, detail="Chunk has no stored embedding")

    meta = page["metadatas"][0] or {}
    where = _where_for(meta.get("notebookId") if sameNotebook else None, None)
    results = _raw_query(vector, where, k + 1)
    # The chunk is its own nearest neighbour; drop it and close the rank gap.
    neighbours = [r for r in results if r["id"] != chunk_id][:k]
    for position, row in enumerate(neighbours, 1):
        row["rank"] = position
    return {
        "chunk": _chunk_view(page["ids"][0], page["documents"][0], meta, full=False),
        "neighbours": neighbours,
    }


# --- Read: query tracing ------------------------------------------------------

def _raw_query(vector: list[float], where: dict | None, k: int) -> list[dict]:
    """One Chroma similarity search, flattened into ranked rows with distances."""
    col = get_collection()
    kwargs: dict = {
        "query_embeddings": [vector],
        "n_results": k,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where
    raw = col.query(**kwargs)

    ids = (raw.get("ids") or [[]])[0]
    docs = (raw.get("documents") or [[]])[0]
    metas = (raw.get("metadatas") or [[]])[0]
    dists = (raw.get("distances") or [[]])[0]

    rows = []
    for i, cid in enumerate(ids):
        meta = metas[i] if i < len(metas) else {}
        distance = float(dists[i]) if i < len(dists) else None
        row = _chunk_view(cid, docs[i] if i < len(docs) else "", meta, full=True)
        row["rank"] = i + 1
        row["distance"] = distance
        # Chroma's cosine space returns 1 - cosine_similarity.
        row["similarity"] = round(1.0 - distance, 6) if distance is not None else None
        rows.append(row)
    return rows


@router.post("/query", summary="Run a retrieval and see exactly what comes back")
async def admin_query(req: AdminQueryRequest = Body(...)) -> dict:
    """
    The debugging endpoint the manager's query playground is built on.

    Unlike POST /retrieve, this returns the whole trace: the rewritten query,
    the embedding timing, every candidate with its raw distance and source
    attribution, and - when reranking is on - the order before and after the
    LLM moved things around. Each stage is optional so a bad answer can be
    bisected between "the vectors are wrong" and "the rerank is wrong".
    """
    trace = get_trace_id()
    t_total = time.perf_counter()
    notebook_ids = req.resolved_notebook_ids()
    stages: list[dict] = []

    # Stage 1 - optional query rewrite.
    effective_query = req.question
    if req.rewrite:
        t0 = time.perf_counter()
        try:
            from app.services.rewriter import rewrite_query
            effective_query = rewrite_query(req.question, req.history)
            stages.append({
                "stage": "rewrite", "ok": True,
                "elapsedMs": int((time.perf_counter() - t0) * 1000),
                "input": req.question, "output": effective_query,
                "changed": effective_query != req.question,
            })
        except Exception as exc:
            stages.append({
                "stage": "rewrite", "ok": False,
                "elapsedMs": int((time.perf_counter() - t0) * 1000),
                "error": str(exc), "output": req.question,
            })
            effective_query = req.question
    else:
        stages.append({"stage": "rewrite", "ok": True, "skipped": True, "output": effective_query})

    # Stage 2 - embed.
    t0 = time.perf_counter()
    try:
        from app.services.embedder import embed_query
        vector = embed_query(effective_query)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Embedding failed: {exc}") from exc
    stages.append({
        "stage": "embed", "ok": True,
        "elapsedMs": int((time.perf_counter() - t0) * 1000),
        "model": settings.embedding_model, "dimensions": len(vector),
    })

    # Stage 3 - vector search.
    where = _build_where(notebook_ids) if notebook_ids else None
    if req.sourceId:
        source_clause = {"sourceId": {"$eq": req.sourceId}}
        where = {"$and": [where, source_clause]} if where else source_clause

    pool_kwargs: dict = {"include": []}
    if where:
        pool_kwargs["where"] = where
    pool = len(col_get(**pool_kwargs)["ids"])

    t0 = time.perf_counter()
    candidates = _raw_query(vector, where, min(req.k, pool)) if pool else []
    stages.append({
        "stage": "search", "ok": True,
        "elapsedMs": int((time.perf_counter() - t0) * 1000),
        "candidatePool": pool, "requested": req.k, "returned": len(candidates),
        "where": where,
    })

    # Stage 4 - optional LLM rerank, reported as a before/after permutation.
    results = candidates
    if req.rerank and len(candidates) > 1:
        t0 = time.perf_counter()
        try:
            from app.services.retriever import ChunkResult, _rerank
            as_chunks = [
                ChunkResult(content=c["content"], source=c["source"], sourceId=c["sourceId"])
                for c in candidates
            ]
            reranked = _rerank(effective_query, as_chunks)
            by_content: dict[str, dict] = {}
            for c in candidates:
                by_content.setdefault(c["content"], c)
            ordered = [by_content[c.content] for c in reranked if c.content in by_content]
            for new_rank, row in enumerate(ordered, 1):
                row["rerankedRank"] = new_rank
                row["rankDelta"] = row["rank"] - new_rank
            results = ordered
            stages.append({
                "stage": "rerank", "ok": True,
                "elapsedMs": int((time.perf_counter() - t0) * 1000),
                "model": settings.litellm_model,
                "orderBefore": [c["id"] for c in candidates],
                "orderAfter": [c["id"] for c in ordered],
            })
        except Exception as exc:
            stages.append({
                "stage": "rerank", "ok": False,
                "elapsedMs": int((time.perf_counter() - t0) * 1000),
                "error": str(exc),
            })
    else:
        stages.append({"stage": "rerank", "ok": True, "skipped": True})

    final = results[: req.finalK] if req.finalK else results

    # Which files actually contributed - the question most often being asked.
    attribution: dict[str, dict] = {}
    for row in final:
        key = row["sourceId"] or row["source"] or "(unknown)"
        entry = attribution.setdefault(key, {
            "sourceId": row["sourceId"], "source": row["source"],
            "notebookId": row["notebookId"], "chunks": 0, "bestDistance": None,
        })
        entry["chunks"] += 1
        if row["distance"] is not None and (
            entry["bestDistance"] is None or row["distance"] < entry["bestDistance"]
        ):
            entry["bestDistance"] = row["distance"]

    log_step(
        logger, "ADMIN_QUERY",
        f"trace={trace} notebooks={notebook_ids!r} q={req.question!r} "
        f"pool={pool} returned={len(final)}",
    )
    return {
        "question": req.question,
        "effectiveQuery": effective_query,
        "notebookIds": notebook_ids,
        "totalElapsedMs": int((time.perf_counter() - t_total) * 1000),
        "stages": stages,
        "attribution": sorted(attribution.values(), key=lambda a: a["chunks"], reverse=True),
        "results": final,
    }


# --- Write: deletion ----------------------------------------------------------

@router.post(
    "/chunks/delete",
    response_model=DeleteResponse,
    summary="Delete chunks by id",
    description=(
        "Bulk delete by explicit id. Exposed as POST as well as DELETE because "
        "a DELETE carrying a request body is the one shape HTTP proxies are "
        "known to strip, and this service sits behind one. Prefer this form "
        "over the wire; the DELETE alias exists for hand-written requests."
    ),
)
@router.delete("/chunks", response_model=DeleteResponse, summary="Delete chunks by id")
async def delete_chunks(req: DeleteChunksRequest = Body(...)) -> DeleteResponse:
    existing = col_get(ids=req.ids, include=[])["ids"]
    if not existing:
        return DeleteResponse(status="ok", deleted=0, detail="No matching chunk ids")
    get_collection().delete(ids=existing)
    invalidate_snapshot()
    log_step(logger, "ADMIN_DELETE_CHUNKS", f"deleted {len(existing)} chunks")
    return DeleteResponse(status="ok", deleted=len(existing))


@router.delete(
    "/notebooks/{notebook_id}",
    response_model=DeleteResponse,
    summary="Delete every chunk of one notebook",
)
async def delete_notebook(notebook_id: str) -> DeleteResponse:
    where = _build_where([notebook_id])
    count = len(col_get(where=where, include=[])["ids"])
    if count:
        get_collection().delete(where=where)
        invalidate_snapshot()
    log_step(logger, "ADMIN_DELETE_NOTEBOOK", f"notebook={notebook_id} deleted={count}")
    return DeleteResponse(status="ok", deleted=count, notebookId=notebook_id)


@router.delete(
    "/notebooks/{notebook_id}/sources/{source_id}",
    response_model=DeleteResponse,
    summary="Delete every chunk of one source file",
)
async def delete_source(notebook_id: str, source_id: str) -> DeleteResponse:
    where = {
        "$and": [
            {"notebookId": {"$eq": notebook_id}},
            {"sourceId": {"$eq": source_id}},
        ]
    }
    count = len(col_get(where=where, include=[])["ids"])
    if count:
        get_collection().delete(where=where)
        invalidate_snapshot()
    log_step(
        logger, "ADMIN_DELETE_SOURCE",
        f"notebook={notebook_id} source={source_id} deleted={count}",
    )
    return DeleteResponse(
        status="ok", deleted=count, notebookId=notebook_id, sourceId=source_id,
    )
