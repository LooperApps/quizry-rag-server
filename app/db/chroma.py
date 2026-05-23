import logging
import os
import threading

import chromadb

from app.config import settings

logger = logging.getLogger(__name__)

_client: chromadb.PersistentClient | None = None
_lock = threading.Lock()

# Single collection name — notebooks are isolated by `notebookId` metadata field.
COLLECTION_NAME = "notebook_docs"


def get_client() -> chromadb.PersistentClient:
    """Return the singleton ChromaDB persistent client."""
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                # Ensure the directory exists before ChromaDB tries to open it
                os.makedirs(settings.chroma_path, exist_ok=True)
                logger.info(f"Initializing ChromaDB at path: {settings.chroma_path}")
                _client = chromadb.PersistentClient(
                    path=settings.chroma_path,
                )
    return _client


def get_collection() -> chromadb.Collection:
    """Return the single shared collection, creating it if necessary."""
    # chromadb 1.x supports both the legacy metadata dict and the new
    # configuration API. Try the new API first; fall back to legacy.
    try:
        return get_client().get_or_create_collection(
            name=COLLECTION_NAME,
            configuration={"hnsw:space": "cosine"},
        )
    except TypeError:
        # Older chromadb versions use metadata= instead of configuration=
        return get_client().get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )


def count_for_notebooks(notebook_ids: list[str]) -> int:
    """
    Return the number of stored chunks for the given notebook IDs.
    Uses collection.get() with include=[] (IDs only) for efficiency.
    """
    col = get_collection()
    where = _build_where(notebook_ids)
    try:
        result = col.get(where=where, include=[])
        return len(result["ids"])
    except Exception as e:
        logger.warning(f"count_for_notebooks error: {e}")
        return 0


def _build_where(notebook_ids: list[str]) -> dict:
    """Build a ChromaDB where-filter for one or more notebook IDs."""
    if len(notebook_ids) == 1:
        return {"notebookId": {"$eq": notebook_ids[0]}}
    return {"$or": [{"notebookId": {"$eq": nid}} for nid in notebook_ids]}
