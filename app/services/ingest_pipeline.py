"""
Ingestion pipeline orchestrator.

Coordinates: download → extract text → chunk → embed → store in ChromaDB.
Also provides helpers for deleting chunks (notebook-level and source-level).
"""

import asyncio
import logging
import uuid

from app.config import settings
from app.db.chroma import get_collection, _build_where
from app.log_utils import get_trace_id, log_step, log_step_data
from app.services.chunker import chunk_document
from app.services.embedder import embed_texts
from app.services.extractor import download_bytes, extract_text

logger = logging.getLogger(__name__)
LINE = "=" * 80


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
    trace = get_trace_id()
    logger.info(
        f"\n{LINE}\n"
        f"[INGEST-PIPELINE] trace={trace} START\n"
        f"notebook_id={notebook_id!r}\n"
        f"source_id={source_id!r}\n"
        f"file_name={file_name!r}\n"
        f"storage_url={storage_url!r}\n"
        f"{LINE}"
    )

    # 1. Download
    log_step(logger, "STEP1_DOWNLOAD", f"Downloading from storage_url={storage_url!r}")
    data = await download_bytes(storage_url)
    log_step(
        logger, "STEP1_DOWNLOAD",
        f"Downloaded {len(data):,} bytes for source={source_id}",
    )

    # 2. Extract text
    log_step(logger, "STEP2_EXTRACT", f"Extracting text from file={file_name}")
    text = extract_text(data, file_name)
    if not text.strip():
        raise ValueError(
            f"No extractable text found in '{file_name}'. "
            "Check that the file is not image-only or encrypted."
        )
    log_step(
        logger, "STEP2_EXTRACT",
        f"Extracted {len(text):,} chars for source={source_id}",
    )
    # Log a preview of the extracted text
    log_step_data(logger, "EXTRACTED_TEXT", f"source={source_id}", text[:3000])

    # 3. Chunk (CPU + network bound → run in thread pool)
    log_step(logger, "STEP3_CHUNK", f"Chunking {len(text)} chars for source={source_id}")
    loop = asyncio.get_running_loop()
    chunks: list[dict] = await loop.run_in_executor(
        None,
        lambda: chunk_document(text, file_name, notebook_id, source_id),
    )
    log_step(
        logger, "STEP3_CHUNK",
        f"Created {len(chunks)} chunks for source={source_id}",
    )

    if not chunks:
        raise ValueError(f"Chunking produced 0 chunks for source={source_id}")

    # Log chunk previews
    for i, ch in enumerate(chunks):
        logger.debug(
            f"[INGEST-PIPELINE] trace={trace} chunk[{i}] "
            f"len={len(ch['content'])} preview={ch['content'][:200]!r}"
        )

    # 4. Delete existing chunks for this source (supports re-ingest)
    log_step(logger, "STEP4_CLEANUP", f"Deleting old chunks for source={source_id}")
    _delete_source_chunks(notebook_id, source_id)

    # 5. Embed (network bound → run in thread pool)
    texts = [c["content"] for c in chunks]
    log_step(
        logger, "STEP5_EMBED",
        f"Embedding {len(texts)} chunks model={settings.embedding_model}",
    )
    embeddings: list[list[float]] = await loop.run_in_executor(
        None,
        lambda: embed_texts(texts),
    )
    log_step(
        logger, "STEP5_EMBED",
        f"Got {len(embeddings)} embeddings, each dim={len(embeddings[0]) if embeddings else 0}",
    )

    # 6. Store in ChromaDB
    collection = get_collection()
    # IDs include a random suffix to avoid collisions on concurrent re-ingest
    ids = [f"{source_id}_{i}_{uuid.uuid4().hex[:6]}" for i in range(len(chunks))]
    metadatas = [c["metadata"] for c in chunks]

    log_step(
        logger, "STEP6_STORE",
        f"Adding {len(ids)} chunks to ChromaDB collection",
    )

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )

    total_in_collection = collection.count()
    logger.info(
        f"\n{LINE}\n"
        f"[INGEST-PIPELINE] trace={trace} DONE\n"
        f"notebook_id={notebook_id!r}\n"
        f"source_id={source_id!r}\n"
        f"file_name={file_name!r}\n"
        f"chunks_stored={len(chunks)}\n"
        f"total_in_collection={total_in_collection}\n"
        f"{LINE}"
    )
    return len(chunks)


def _delete_source_chunks(notebook_id: str, source_id: str) -> None:
    """
    Delete all ChromaDB documents belonging to one specific source.
    Safe to call even if no documents exist yet.
    """
    trace = get_trace_id()
    collection = get_collection()
    where_filter = {
        "$and": [
            {"notebookId": {"$eq": notebook_id}},
            {"sourceId": {"$eq": source_id}},
        ]
    }
    try:
        # Count how many will be deleted
        matching = collection.get(where=where_filter, include=[])
        count = len(matching["ids"])
        log_step(
            logger, "DELETE_SOURCE",
            f"Deleting {count} chunks for notebook={notebook_id} source={source_id}",
        )
        collection.delete(where=where_filter)
        log_step(
            logger, "DELETE_SOURCE",
            f"Deleted {count} chunks for source={source_id}",
        )
    except Exception as exc:
        # Non-fatal: log and continue
        log_step(
            logger, "DELETE_SOURCE",
            f"Could not delete old chunks for source={source_id}: {exc}",
        )


def delete_notebook_chunks(notebook_id: str) -> None:
    """
    Delete ALL ChromaDB documents belonging to a notebook.
    Called when a notebook is deleted in Firestore.
    """
    trace = get_trace_id()
    collection = get_collection()
    try:
        where = _build_where([notebook_id])
        # Count how many will be deleted
        matching = collection.get(where=where, include=[])
        count = len(matching["ids"])
        log_step(
            logger, "DELETE_NOTEBOOK",
            f"Deleting {count} chunks for notebook={notebook_id}",
        )
        collection.delete(where=where)
        log_step(
            logger, "DELETE_NOTEBOOK",
            f"Deleted {count} chunks for notebook={notebook_id}",
        )
    except Exception as exc:
        log_step(
            logger, "DELETE_NOTEBOOK",
            f"Could not delete chunks for notebook={notebook_id}: {exc}",
        )
