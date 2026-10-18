"""
Admin aggregation service.

ChromaDB exposes no GROUP BY, so the notebook -> source -> chunk tree that the
manager UI is built around has to be produced by paging through the whole
collection once and tallying metadata. That scan is O(collection), and every
screen of the UI wants some slice of it, so the result is cached behind a TTL
and invalidated whenever this process mutates the collection.

Nothing here writes; deletion lives in the router so it can log and report.
"""

import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Iterator

from app.config import settings
from app.db.chroma import COLLECTION_NAME, get_collection
from app.log_utils import log_step

logger = logging.getLogger(__name__)

# Rows pulled per collection.get() call while scanning. Large enough that a
# 100k-chunk collection is ~50 round trips, small enough not to hold the whole
# document set in memory twice.
SCAN_PAGE_SIZE = 2_000

# How long a snapshot stays fresh. The collection only changes on ingest or
# delete, both of which invalidate explicitly - the TTL is just a backstop for
# writes made by another process (e.g. a Cloud Function calling /ingest).
SNAPSHOT_TTL_SECONDS = 60

# Cached per collection: the manager switches between apps constantly, and a
# single slot would make every switch pay for a full rescan of both.
_snapshots: dict[str, tuple[dict[str, Any], float]] = {}
_snapshot_lock = threading.Lock()


# --- Chroma result normalisation ---------------------------------------------

def embeddings_to_lists(embeddings) -> list:
    """
    chromadb 1.x hands back numpy arrays for embeddings, which are not JSON
    serialisable and whose truthiness raises. Normalise to plain lists.
    """
    if embeddings is None:
        return []
    out = []
    for vec in embeddings:
        if vec is None:
            out.append(None)
        elif hasattr(vec, "tolist"):
            out.append(vec.tolist())
        else:
            out.append(list(vec))
    return out


def col_get(collection: str | None = None, **kwargs) -> dict:
    """collection.get() with the keys we rely on always present as lists."""
    raw = get_collection(collection).get(**kwargs)
    n = len(raw.get("ids") or [])
    return {
        "ids": list(raw.get("ids") or []),
        "documents": list(raw.get("documents") or []) or [None] * n,
        "metadatas": list(raw.get("metadatas") or []) or [None] * n,
        "embeddings": embeddings_to_lists(raw.get("embeddings")),
    }


def iter_chunks(
    where: dict | None = None,
    where_document: dict | None = None,
    include: list[str] | None = None,
    page_size: int = SCAN_PAGE_SIZE,
    collection: str | None = None,
) -> Iterator[dict]:
    """
    Yield chunks one at a time, paging through the collection so the caller
    never materialises more than `page_size` rows at once.
    """
    include = include if include is not None else ["documents", "metadatas"]
    offset = 0
    while True:
        kwargs: dict[str, Any] = {"limit": page_size, "offset": offset, "include": include}
        if where:
            kwargs["where"] = where
        if where_document:
            kwargs["where_document"] = where_document
        page = col_get(collection=collection, **kwargs)
        ids = page["ids"]
        if not ids:
            return
        for i, chunk_id in enumerate(ids):
            yield {
                "id": chunk_id,
                "document": page["documents"][i] if i < len(page["documents"]) else None,
                "metadata": page["metadatas"][i] if i < len(page["metadatas"]) else None,
                "embedding": page["embeddings"][i] if i < len(page["embeddings"]) else None,
            }
        if len(ids) < page_size:
            return
        offset += len(ids)


# --- Snapshot ----------------------------------------------------------------

def _blank_stats() -> dict:
    return {"chunks": 0, "chars": 0, "bytes": 0, "minChunkChars": None, "maxChunkChars": None}


def _tally(stats: dict, text: str) -> None:
    chars = len(text)
    stats["chunks"] += 1
    stats["chars"] += chars
    stats["bytes"] += len(text.encode("utf-8"))
    if stats["minChunkChars"] is None or chars < stats["minChunkChars"]:
        stats["minChunkChars"] = chars
    if stats["maxChunkChars"] is None or chars > stats["maxChunkChars"]:
        stats["maxChunkChars"] = chars


def _finalise(stats: dict) -> dict:
    stats["avgChunkChars"] = round(stats["chars"] / stats["chunks"], 1) if stats["chunks"] else 0
    return stats


def build_snapshot(collection: str | None = None) -> dict:
    """Page the whole collection and tally it into a notebook -> source tree."""
    t0 = time.perf_counter()
    notebooks: dict[str, dict] = {}
    totals = _blank_stats()
    unassigned = 0

    for chunk in iter_chunks(include=["documents", "metadatas"], collection=collection):
        meta = chunk["metadata"] or {}
        text = chunk["document"] or ""
        notebook_id = str(meta.get("notebookId") or "")
        source_id = str(meta.get("sourceId") or "")
        source_name = str(meta.get("source") or "")

        if not notebook_id:
            unassigned += 1
            notebook_id = "(no notebookId)"

        nb = notebooks.setdefault(
            notebook_id,
            {"notebookId": notebook_id, **_blank_stats(), "sources": {}},
        )
        src_key = source_id or f"(no sourceId){source_name}"
        src = nb["sources"].setdefault(
            src_key,
            {
                "sourceId": source_id,
                "source": source_name,
                "sampleChunkId": chunk["id"],
                **_blank_stats(),
            },
        )
        # A re-ingest can change the file name while keeping the sourceId.
        if source_name and not src["source"]:
            src["source"] = source_name

        _tally(src, text)
        _tally(nb, text)
        _tally(totals, text)

    notebook_list = []
    for nb in notebooks.values():
        sources = [_finalise(s) for s in nb["sources"].values()]
        sources.sort(key=lambda s: s["chunks"], reverse=True)
        nb["sources"] = sources
        nb["sourceCount"] = len(sources)
        notebook_list.append(_finalise(nb))
    notebook_list.sort(key=lambda n: n["chunks"], reverse=True)

    snapshot = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "elapsedMs": int((time.perf_counter() - t0) * 1000),
        "collection": collection or COLLECTION_NAME,
        "totals": {
            **_finalise(totals),
            "notebooks": len(notebook_list),
            "sources": sum(n["sourceCount"] for n in notebook_list),
            "chunksWithoutNotebookId": unassigned,
        },
        "notebooks": notebook_list,
    }
    log_step(
        logger, "ADMIN_SNAPSHOT",
        f"scanned {totals['chunks']} chunks across {len(notebook_list)} notebooks "
        f"in collection {snapshot['collection']} in {snapshot['elapsedMs']}ms",
    )
    return snapshot


def get_snapshot(force: bool = False, collection: str | None = None) -> dict:
    """Return the cached snapshot for one collection, rebuilding when stale."""
    key = collection or COLLECTION_NAME
    with _snapshot_lock:
        cached = _snapshots.get(key)
        if cached and not force and (time.time() - cached[1]) < SNAPSHOT_TTL_SECONDS:
            return {**cached[0], "cached": True}
        snapshot = build_snapshot(key)
        _snapshots[key] = (snapshot, time.time())
        return {**snapshot, "cached": False}


def invalidate_snapshot(collection: str | None = None) -> None:
    """
    Drop the cache after a mutation so the next read reflects it.
    Passing no collection clears every app's — used after a cross-app move.
    """
    with _snapshot_lock:
        if collection is None:
            _snapshots.clear()
        else:
            _snapshots.pop(collection, None)


# --- Disk usage --------------------------------------------------------------

def human_bytes(n: int | None) -> str | None:
    if n is None:
        return None
    step = 1024.0
    size = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < step:
            return f"{int(size)} B" if unit == "B" else f"{size:.1f} {unit}"
        size /= step
    return f"{size:.1f} PB"


def _dir_size(path: str) -> int:
    size = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                size += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return size


def disk_usage() -> dict:
    """
    Walk the ChromaDB directory and report its footprint.

    The sqlite file holds documents and metadata; the per-collection UUID
    directories hold the HNSW index, which is what actually grows with the
    embedding dimension. Reporting them separately makes it obvious whether a
    bloated disk is text or vectors.
    """
    from app.db.chroma import get_client

    path = settings.chroma_path
    if not os.path.isdir(path):
        # _resolve_chroma_path may have fallen back to a temp dir.
        try:
            resolved = getattr(get_client(), "_identifier", None)
            if isinstance(resolved, str) and os.path.isdir(resolved):
                path = resolved
        except Exception:
            pass

    entries: list[dict] = []
    total = 0
    if os.path.isdir(path):
        for name in sorted(os.listdir(path)):
            full = os.path.join(path, name)
            is_file = os.path.isfile(full)
            try:
                size = os.path.getsize(full) if is_file else _dir_size(full)
            except OSError:
                size = 0
            total += size
            entries.append({
                "name": name,
                "kind": "file" if is_file else "directory",
                "bytes": size,
                "human": human_bytes(size),
            })

    free_bytes: int | None = None
    volume_bytes: int | None = None
    try:
        import shutil
        du = shutil.disk_usage(path if os.path.isdir(path) else ".")
        free_bytes, volume_bytes = du.free, du.total
    except OSError:
        pass

    entries.sort(key=lambda e: e["bytes"], reverse=True)
    return {
        "path": path,
        "exists": os.path.isdir(path),
        "totalBytes": total,
        "totalHuman": human_bytes(total),
        "entries": entries,
        "volumeFreeBytes": free_bytes,
        "volumeTotalBytes": volume_bytes,
        "volumeFreeHuman": human_bytes(free_bytes),
        "volumeTotalHuman": human_bytes(volume_bytes),
    }


def embedding_info(collection: str | None = None) -> dict:
    """Report the vector dimension actually stored, not the one configured."""
    try:
        page = col_get(collection=collection, limit=1, include=["embeddings"])
        vec = page["embeddings"][0] if page["embeddings"] else None
        return {
            "model": settings.embedding_model,
            "dimensions": len(vec) if vec else None,
            "space": "cosine",
        }
    except Exception as exc:  # diagnostic path - never fail the stats call
        return {"model": settings.embedding_model, "dimensions": None, "error": str(exc)}


# --- Cross-app migration ------------------------------------------------------

def move_notebooks(
    notebook_ids: list[str],
    source_collection: str,
    target_collection: str,
    batch_size: int = 500,
) -> dict:
    """
    Move every chunk of the given notebooks from one app's collection to
    another, preserving ids, documents, metadata and — crucially — the stored
    embeddings, so nothing has to be re-embedded.

    This exists because both apps ingested into one shared collection before
    tenants were introduced; it is the one-time path that separates them.
    Copy-then-delete per batch: a crash mid-run leaves duplicates, which a
    re-run resolves, rather than a hole.
    """
    if source_collection == target_collection:
        raise ValueError("source and target collections are the same")
    if not notebook_ids:
        return {"moved": 0, "notebooks": {}}

    from app.db.chroma import _build_where, get_collection

    src = get_collection(source_collection)
    dst = get_collection(target_collection)
    moved_per_notebook: dict[str, int] = {}
    total = 0

    for notebook_id in notebook_ids:
        where = _build_where([notebook_id])
        moved = 0
        while True:
            page = col_get(
                collection=source_collection,
                where=where,
                limit=batch_size,
                include=["documents", "metadatas", "embeddings"],
            )
            ids = page["ids"]
            if not ids:
                break
            dst.add(
                ids=ids,
                documents=page["documents"],
                metadatas=page["metadatas"],
                embeddings=page["embeddings"],
            )
            src.delete(ids=ids)
            moved += len(ids)
            if len(ids) < batch_size:
                break
        moved_per_notebook[notebook_id] = moved
        total += moved
        log_step(
            logger, "ADMIN_MOVE",
            f"notebook={notebook_id} moved={moved} "
            f"{source_collection} -> {target_collection}",
        )

    invalidate_snapshot()
    return {"moved": total, "notebooks": moved_per_notebook}
