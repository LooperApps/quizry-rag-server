"""
Ingestion pipeline orchestrator.

Coordinates: download → extract text → chunk → embed → store in ChromaDB.
Also provides helpers for deleting chunks (notebook-level and source-level).

Two storage paths, chosen by the target collection's flags (app/db/chroma.py):
  legacy   whole-document text, default chunker, delete-then-add with random
           id suffixes — exactly the behaviour unflagged collections always had
  pages    page-wise extraction (DeepSeek for junk/scanned pages and figures),
           pages_v1 chunker, deterministic ids "{sourceId}:{chunkIndex:05d}"
           upserted, then the source's leftover ids deleted (store_pages) —
           also the whole of POST /ingest/text
Everything blocking runs in the default executor, never on the event loop.
"""

import asyncio
import logging
import uuid

from app.config import settings
from app.db.chroma import (
    COLLECTION_NAME,
    WRITE_BATCH,
    _build_where,
    collection_flags,
    dimensions_for,
    flags_from_metadata,
    get_collection,
    writing,
)
from app.log_utils import get_trace_id, log_step, log_step_data, redact_url
from app.services.chunker import chunk_document, chunk_pages
from app.services.embedder import embed_documents, embed_texts
from app.services.extractor import download_bytes, extract_pages, extract_text
from app.services.text_quality import normalize_light

logger = logging.getLogger(__name__)
LINE = "=" * 80


class SourceConflictError(ValueError):
    """The sourceId's chunks belong to another notebook (HTTP 409)."""


def chunk_id(source_id: str, index: int) -> str:
    return f"{source_id}:{index:05d}"


async def run_ingest(
    notebook_id: str,
    source_id: str,
    storage_url: str,
    file_name: str,
    collection: str | None = None,
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
        f"collection={collection!r}\n"
        f"notebook_id={notebook_id!r}\n"
        f"source_id={source_id!r}\n"
        f"file_name={file_name!r}\n"
        f"storage_url={redact_url(storage_url)!r}\n"
        f"{LINE}"
    )

    # 1. Download
    log_step(logger, "STEP1_DOWNLOAD", f"Downloading from storage_url={redact_url(storage_url)!r}")
    data = await download_bytes(storage_url)
    log_step(
        logger, "STEP1_DOWNLOAD",
        f"Downloaded {len(data):,} bytes for source={source_id}",
    )
    # Registered as one write from here on, so a promotion cannot swap the
    # collection between reading its flags and storing the chunks.
    with writing(collection or COLLECTION_NAME):
        return await _ingest_bytes(data, notebook_id, source_id, file_name, collection)


async def _ingest_bytes(
    data: bytes,
    notebook_id: str,
    source_id: str,
    file_name: str,
    collection: str | None,
) -> int:
    """Steps 2-6 of run_ingest for downloaded bytes."""
    trace = get_trace_id()
    loop = asyncio.get_running_loop()

    flags = await loop.run_in_executor(None, collection_flags, collection)
    if flags.pages:
        # Flagged collection: page-wise extraction (+ DeepSeek), pages_v1 chunks.
        log_step(logger, "STEP2_EXTRACT", f"Page-wise extraction of file={file_name} vision={flags.pdf_vision}")
        extraction = await loop.run_in_executor(
            None,
            lambda: extract_pages(data, file_name, vision_enabled=flags.pdf_vision),
        )
        pages = [(p.page, p.text) for p in extraction.pages if p.text.strip()]
        if not pages:
            raise ValueError(
                f"No extractable text found in '{file_name}'. "
                "Check that the file is not image-only or encrypted."
            )
        log_step(
            logger, "STEP2_EXTRACT",
            f"source={source_id} pages={extraction.page_count} with_text={len(pages)} "
            f"method={extraction.method} vision_pages={extraction.vision_pages} "
            f"truncated={extraction.truncated}",
        )
        return await loop.run_in_executor(
            None,
            lambda: store_pages(notebook_id, source_id, file_name, pages, collection),
        )

    # 2. Extract text
    log_step(logger, "STEP2_EXTRACT", f"Extracting text from file={file_name}")
    text = await loop.run_in_executor(None, extract_text, data, file_name)
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
    await loop.run_in_executor(None, _delete_source_chunks, notebook_id, source_id, collection)

    # 5. Embed (network bound → run in thread pool)
    texts = [c["content"] for c in chunks]
    log_step(
        logger, "STEP5_EMBED",
        f"Embedding {len(texts)} chunks model={settings.embedding_model}",
    )
    embeddings: list[list[float]] = await loop.run_in_executor(
        None,
        lambda: embed_texts(texts, dimensions_for(collection)),
    )
    log_step(
        logger, "STEP5_EMBED",
        f"Got {len(embeddings)} embeddings, each dim={len(embeddings[0]) if embeddings else 0}",
    )

    # 6. Store in ChromaDB
    # IDs include a random suffix to avoid collisions on concurrent re-ingest
    ids = [f"{source_id}_{i}_{uuid.uuid4().hex[:6]}" for i in range(len(chunks))]
    metadatas = [c["metadata"] for c in chunks]

    log_step(
        logger, "STEP6_STORE",
        f"Adding {len(ids)} chunks to ChromaDB collection",
    )

    def _store() -> tuple[str, int]:
        col = get_collection(collection)
        # Batched: one add of more than Chroma's max batch (~5.4k) fails.
        for start in range(0, len(ids), WRITE_BATCH):
            end = start + WRITE_BATCH
            col.add(
                ids=ids[start:end],
                embeddings=embeddings[start:end],
                documents=texts[start:end],
                metadatas=metadatas[start:end],
            )
        return col.name, col.count()

    col_name, total_in_collection = await loop.run_in_executor(None, _store)
    logger.info(
        f"\n{LINE}\n"
        f"[INGEST-PIPELINE] trace={trace} DONE\n"
        f"collection={col_name}\n"
        f"notebook_id={notebook_id!r}\n"
        f"source_id={source_id!r}\n"
        f"file_name={file_name!r}\n"
        f"chunks_stored={len(chunks)}\n"
        f"total_in_collection={total_in_collection}\n"
        f"{LINE}"
    )
    return len(chunks)


def store_pages(
    notebook_id: str,
    source_id: str,
    file_name: str,
    pages: list[tuple[int, str]],
    collection: str | None = None,
) -> int:
    """
    Chunk (pages_v1), embed (collection flags) and store one source's pages,
    idempotently: chunk ids are "{sourceId}:{chunkIndex:05d}", written with
    upsert, and only afterwards are the source's ids that the new version no
    longer has deleted — a re-ingest never passes through zero chunks, and a
    retry or a concurrent duplicate converges on the same set.

    Refuses (SourceConflictError) when this sourceId already has chunks under
    another notebook: the ids would collide and the upsert would move them.
    Synchronous; call it from a worker thread. Returns the chunk count.
    """
    trace = get_trace_id()
    chunks = chunk_pages(pages, file_name, notebook_id, source_id)
    if not chunks:
        raise ValueError(f"No text to index for source={source_id}")
    texts = [c["content"] for c in chunks]
    ids = [chunk_id(source_id, i) for i in range(len(chunks))]

    with writing(collection or COLLECTION_NAME):
        col = get_collection(collection)
        flags = flags_from_metadata(col.metadata)

        existing = col.get(where={"sourceId": {"$eq": source_id}}, include=["metadatas"])
        existing_ids = list(existing.get("ids") or [])
        owners = {
            str((meta or {}).get("notebookId") or "")
            for meta in (existing.get("metadatas") or [])
        } - {"", notebook_id}
        if owners:
            raise SourceConflictError(
                f"sourceId {source_id!r} already has chunks under notebook(s) {sorted(owners)!r} "
                f"in {col.name}; refusing to overwrite them"
            )

        log_step(
            logger, "STORE_PAGES",
            f"collection={col.name} source={source_id} pages={len(pages)} chunks={len(chunks)} "
            f"existing={len(existing_ids)} normalized={flags.normalized} task_types={flags.task_types}",
        )
        vectors = embed_documents(texts, flags)

        metadatas = [c["metadata"] for c in chunks]
        for start in range(0, len(ids), WRITE_BATCH):
            end = start + WRITE_BATCH
            col.upsert(
                ids=ids[start:end],
                embeddings=vectors[start:end],
                documents=texts[start:end],
                metadatas=metadatas[start:end],
            )
        keep = set(ids)
        stale = [cid for cid in existing_ids if cid not in keep]
        for start in range(0, len(stale), WRITE_BATCH):
            col.delete(ids=stale[start:start + WRITE_BATCH])
        name = col.name

    logger.info(
        f"\n{LINE}\n"
        f"[INGEST-PIPELINE] trace={trace} STORED (pages_v1)\n"
        f"collection={name}\n"
        f"notebook_id={notebook_id!r}\n"
        f"source_id={source_id!r}\n"
        f"file_name={file_name!r}\n"
        f"chunks_stored={len(chunks)} stale_deleted={len(stale)}\n"
        f"{LINE}"
    )
    return len(chunks)


def normalize_pages(pages: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """
    POST /ingest/text input, lightly normalised (control/invisible characters
    dropped, space runs collapsed, newlines kept), in page order, empty pages
    dropped. Repeated page numbers are merged in their given order.
    """
    merged: dict[int, list[str]] = {}
    for page, text in pages:
        cleaned = normalize_light(text or "")
        if cleaned:
            merged.setdefault(int(page), []).append(cleaned)
    return [(page, "\n\n".join(parts)) for page, parts in sorted(merged.items())]


async def ingest_text(
    notebook_id: str,
    source_id: str,
    file_name: str,
    pages: list[tuple[int, str]],
    collection: str | None = None,
) -> tuple[int, int]:
    """
    POST /ingest/text: store already-extracted page text (pages_v1 chunks,
    deterministic upsert, the collection's embedding flags). Runs in a worker
    thread. Returns (chunkCount, pageCount); ValueError when there is no text.
    """
    cleaned = normalize_pages(pages)
    if not cleaned:
        raise ValueError(f"No text to index for source={source_id}: every page is empty")
    log_step(
        logger, "INGEST_TEXT",
        f"collection={collection!r} notebook={notebook_id!r} source={source_id!r} file={file_name!r} "
        f"pages={len(pages)} non_empty={len(cleaned)} chars={sum(len(t) for _, t in cleaned):,}",
    )
    loop = asyncio.get_running_loop()
    count = await loop.run_in_executor(
        None,
        lambda: store_pages(notebook_id, source_id, file_name, cleaned, collection),
    )
    return count, len(pages)


def _delete_source_chunks(
    notebook_id: str, source_id: str, collection: str | None = None
) -> None:
    """
    Delete all ChromaDB documents belonging to one specific source.
    Safe to call even if no documents exist yet.
    """
    trace = get_trace_id()
    col = get_collection(collection)
    where_filter = {
        "$and": [
            {"notebookId": {"$eq": notebook_id}},
            {"sourceId": {"$eq": source_id}},
        ]
    }
    try:
        # Count how many will be deleted
        matching = col.get(where=where_filter, include=[])
        count = len(matching["ids"])
        log_step(
            logger, "DELETE_SOURCE",
            f"Deleting {count} chunks for notebook={notebook_id} source={source_id}",
        )
        col.delete(where=where_filter)
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


def delete_notebook_chunks(notebook_id: str, collection: str | None = None) -> None:
    """
    Delete ALL ChromaDB documents belonging to a notebook.
    Called when a notebook is deleted in Firestore.
    """
    trace = get_trace_id()
    col = get_collection(collection)
    try:
        where = _build_where([notebook_id])
        # Count how many will be deleted
        matching = col.get(where=where, include=[])
        count = len(matching["ids"])
        log_step(
            logger, "DELETE_NOTEBOOK",
            f"Deleting {count} chunks for notebook={notebook_id}",
        )
        col.delete(where=where)
        log_step(
            logger, "DELETE_NOTEBOOK",
            f"Deleted {count} chunks for notebook={notebook_id}",
        )
    except Exception as exc:
        log_step(
            logger, "DELETE_NOTEBOOK",
            f"Could not delete chunks for notebook={notebook_id}: {exc}",
        )
