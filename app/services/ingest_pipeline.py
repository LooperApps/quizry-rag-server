"""
Ingestion pipeline orchestrator.

Coordinates: download → extract text → chunk → embed → store in ChromaDB.
Also provides helpers for deleting chunks (notebook-level and source-level).
"""

import asyncio
import logging
import uuid

from app.db.chroma import get_collection, _build_where
from app.services.chunker import chunk_document
from app.services.embedder import embed_texts
from app.services.extractor import download_bytes, extract_text

logger = logging.getLogger(__name__)


async def run_ingest(
    notebook_id: str,
    source_id: str,
    storage_url: str,
    file_name: str,
) -> int:
    """
    Full ingestion pipeline for one source file.

    Steps:
      1. Download from Firebase Storage URL
      2. Extract plain text
      3. Chunk via LLM (runs in thread pool to avoid blocking the event loop)
      4. Delete any existing chunks for this source (idempotent re-ingest)
      5. Embed all chunks
      6. Store in ChromaDB

    Returns the number of chunks stored.
    Raises ValueError / Exception on failure — the router converts these to HTTP 500.
    """
    logger.info(
        f"[pipeline] START notebook={notebook_id} source={source_id} file={file_name}"
    )

    # 1. Download
    data = await download_bytes(storage_url)
    logger.info(f"[pipeline] Downloaded {len(data):,} bytes for source={source_id}")

    # 2. Extract text
    text = extract_text(data, file_name)
    if not text.strip():
        raise ValueError(
            f"No extractable text found in '{file_name}'. "
            "Check that the file is not image-only or encrypted."
        )
    logger.info(f"[pipeline] Extracted {len(text):,} chars for source={source_id}")

    # 3. Chunk (CPU + network bound → run in thread pool)
    loop = asyncio.get_running_loop()
    chunks: list[dict] = await loop.run_in_executor(
        None,
        lambda: chunk_document(text, file_name, notebook_id, source_id),
    )
    logger.info(f"[pipeline] Created {len(chunks)} chunks for source={source_id}")

    if not chunks:
        raise ValueError(f"Chunking produced 0 chunks for source={source_id}")

    # 4. Delete existing chunks for this source (supports re-ingest)
    _delete_source_chunks(notebook_id, source_id)

    # 5. Embed (network bound → run in thread pool)
    texts = [c["content"] for c in chunks]
    embeddings: list[list[float]] = await loop.run_in_executor(
        None,
        lambda: embed_texts(texts),
    )

    # 6. Store
    collection = get_collection()
    # IDs include a random suffix to avoid collisions on concurrent re-ingest
    ids = [f"{source_id}_{i}_{uuid.uuid4().hex[:6]}" for i in range(len(chunks))]
    metadatas = [c["metadata"] for c in chunks]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )
    logger.info(
        f"[pipeline] DONE notebook={notebook_id} source={source_id} "
        f"chunks={len(chunks)} total_in_collection={collection.count()}"
    )
    return len(chunks)


def _delete_source_chunks(notebook_id: str, source_id: str) -> None:
    """
    Delete all ChromaDB documents belonging to one specific source.
    Safe to call even if no documents exist yet.
    """
    collection = get_collection()
    try:
        collection.delete(
            where={
                "$and": [
                    {"notebookId": {"$eq": notebook_id}},
                    {"sourceId": {"$eq": source_id}},
                ]
            }
        )
        logger.debug(f"[pipeline] Deleted existing chunks for source={source_id}")
    except Exception as exc:
        # Non-fatal: log and continue
        logger.warning(f"[pipeline] Could not delete old chunks for source={source_id}: {exc}")


def delete_notebook_chunks(notebook_id: str) -> None:
    """
    Delete ALL ChromaDB documents belonging to a notebook.
    Called when a notebook is deleted in Firestore.
    """
    collection = get_collection()
    try:
        where = _build_where([notebook_id])
        collection.delete(where=where)
        logger.info(f"[pipeline] Deleted all chunks for notebook={notebook_id}")
    except Exception as exc:
        logger.warning(f"[pipeline] Could not delete chunks for notebook={notebook_id}: {exc}")
