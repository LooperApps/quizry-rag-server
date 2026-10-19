import logging
import os
import tempfile
import threading

import chromadb

from app.config import settings
from app.log_utils import get_trace_id, log_step

logger = logging.getLogger(__name__)

_client: chromadb.PersistentClient | None = None
_lock = threading.Lock()

# Collections are per app (see app/apps.py); within one, notebooks are isolated
# by the `notebookId` metadata field. This name is the single-tenant default —
# the collection the server wrote before tenants existed, kept so an
# unconfigured deployment still finds its data.
COLLECTION_NAME = "notebook_docs"

# get_or_create_collection is a round trip; the handles are cheap and stable,
# so keep one per name rather than paying for it on every chunk read.
_collections: dict[str, chromadb.Collection] = {}
_collections_lock = threading.Lock()


def _resolve_chroma_path() -> str:
    """
    Return a writable path for ChromaDB storage.
    Tries the configured path first; falls back to /tmp/chroma if not writable
    (e.g. Render persistent disk not yet mounted).
    """
    path = settings.chroma_path
    try:
        os.makedirs(path, exist_ok=True)
        # Quick write-access probe
        probe = os.path.join(path, ".write_probe")
        with open(probe, "w") as f:
            f.write("ok")
        os.remove(probe)
        logger.info(f"ChromaDB path is writable: {path}")
        return path
    except OSError as e:
        fallback = os.path.join(tempfile.gettempdir(), "chroma")
        logger.warning(
            f"ChromaDB path '{path}' is not writable ({e}). "
            f"Falling back to {fallback}. "
            f"Data will NOT persist across restarts. "
            f"Fix: attach a persistent disk at '{path}' in the Render dashboard."
        )
        os.makedirs(fallback, exist_ok=True)
        return fallback


def get_client() -> chromadb.PersistentClient:
    """Return the singleton ChromaDB persistent client."""
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                path = _resolve_chroma_path()
                logger.info(f"Initializing ChromaDB at path: {path}")
                _client = chromadb.PersistentClient(path=path)
    return _client


# Vector size of collections created before EMBEDDING_DIMENSIONS existed:
# gemini-embedding-001's full output. Those collections carry no
# `embedding_dimensions` metadata, so this is what an absent value means.
LEGACY_DIMENSIONS = 3072


def _create_collection(name: str) -> chromadb.Collection:
    # An existing collection is opened as-is, never re-created with new
    # metadata: its stored vectors fix its dimension, whatever is configured.
    try:
        return get_client().get_collection(name=name)
    except Exception:
        pass
    # New collections record the vector size they are built with, so queries
    # and ingests against them embed at that size (see dimensions_for).
    meta = {"embedding_dimensions": settings.embedding_dimensions}
    # chromadb 1.x supports both the legacy metadata dict and the new
    # configuration API. Try the new API first; fall back to legacy.
    try:
        return get_client().get_or_create_collection(
            name=name,
            configuration={"hnsw:space": "cosine"},
            metadata=meta,
        )
    except TypeError:
        # Older chromadb versions use metadata= instead of configuration=
        return get_client().get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine", **meta},
        )


def dimensions_for(name: str | None = None) -> int:
    """The embedding size a collection's vectors were built with."""
    meta = get_collection(name).metadata or {}
    try:
        return int(meta.get("embedding_dimensions") or LEGACY_DIMENSIONS)
    except (TypeError, ValueError):
        return LEGACY_DIMENSIONS


def get_collection(name: str | None = None) -> chromadb.Collection:
    """Return one app's collection, creating it if necessary."""
    name = name or COLLECTION_NAME
    cached = _collections.get(name)
    if cached is not None:
        return cached
    with _collections_lock:
        cached = _collections.get(name)
        if cached is None:
            cached = _create_collection(name)
            _collections[name] = cached
        return cached


def count_for_notebooks(notebook_ids: list[str], collection: str | None = None) -> int:
    """
    Return the number of stored chunks for the given notebook IDs.
    Uses collection.get() with include=[] (IDs only) for efficiency.
    """
    trace = get_trace_id()
    col = get_collection(collection)
    where = _build_where(notebook_ids)
    try:
        result = col.get(where=where, include=[])
        count = len(result["ids"])
        log_step(
            logger, "CHROMA_COUNT",
            f"collection={col.name} notebook_ids={notebook_ids!r} where={where!r} count={count}",
        )
        return count
    except Exception as e:
        log_step(
            logger, "CHROMA_COUNT",
            f"collection={col.name} notebook_ids={notebook_ids!r} ERROR: {e}",
        )
        return 0


def _build_where(notebook_ids: list[str]) -> dict:
    """Build a ChromaDB where-filter for one or more notebook IDs."""
    if len(notebook_ids) == 1:
        return {"notebookId": {"$eq": notebook_ids[0]}}
    return {"$or": [{"notebookId": {"$eq": nid}} for nid in notebook_ids]}


def reset_collection(collection: str | None = None) -> int:
    """
    Drop and recreate one app's collection, wiping all its documents.
    Returns the doc count before reset.
    """
    name = collection or COLLECTION_NAME
    client = get_client()
    try:
        old_count = get_collection(name).count()
    except Exception:
        old_count = 0
    client.delete_collection(name)
    with _collections_lock:
        _collections.pop(name, None)
    get_collection(name)  # recreate
    logger.info(f"[chroma] Collection '{name}' reset. Deleted {old_count} documents.")
    return old_count
