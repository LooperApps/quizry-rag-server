"""
Re-embed one collection into another at a different vector size.

Used to move a collection to EMBEDDING_DIMENSIONS (e.g. 3072 -> 768) without
re-ingesting any source file: chunk text and metadata are read from the old
collection, embedded again, and written to the new one under the same ids.
The old collection keeps serving until RAG_APPS is pointed at the new name.

Idempotent: chunks whose id is already in the target are skipped, so running
it again after the switch picks up anything ingested into the old collection
while the first pass ran. Runs in a background thread; one job at a time.
"""

import logging
import threading
import time
from datetime import datetime, timezone

from app.db.chroma import dimensions_for, get_collection
from app.services.admin_index import invalidate_snapshot
from app.services.embedder import embed_texts

logger = logging.getLogger(__name__)

PAGE_SIZE = 200
# Breathing room between pages so live retrieve/ingest calls are not starved.
PAUSE_SECONDS = 0.2

_lock = threading.Lock()
_job: dict | None = None


def status() -> dict | None:
    with _lock:
        return dict(_job) if _job else None


def start(source: str, target: str) -> dict:
    global _job
    if source == target:
        raise ValueError("source and target collections are the same")
    with _lock:
        if _job and _job["state"] == "running":
            raise RuntimeError(f"a re-embed job is already running ({_job['source']} -> {_job['target']})")
        src_count = get_collection(source).count()
        _job = {
            "source": source,
            "target": target,
            "targetDimensions": dimensions_for(target),
            "state": "running",
            "sourceChunks": src_count,
            "scanned": 0,
            "embedded": 0,
            "skipped": 0,
            "error": None,
            "startedAt": datetime.now(timezone.utc).isoformat(),
            "finishedAt": None,
        }
        threading.Thread(target=_run, args=(source, target), daemon=True).start()
        return dict(_job)


def _update(**fields) -> None:
    with _lock:
        if _job:
            _job.update(fields)


def _run(source: str, target: str) -> None:
    src = get_collection(source)
    dst = get_collection(target)
    dims = dimensions_for(target)
    logger.info(f"[REEMBED] start {source} -> {target} dims={dims} chunks={src.count()}")
    offset = scanned = embedded = skipped = 0
    try:
        while True:
            page = src.get(limit=PAGE_SIZE, offset=offset, include=["documents", "metadatas"])
            ids = list(page.get("ids") or [])
            if not ids:
                break
            offset += len(ids)
            docs = list(page.get("documents") or [])
            metas = list(page.get("metadatas") or [])

            present = set(dst.get(ids=ids, include=[])["ids"])
            keep = [i for i, cid in enumerate(ids) if cid not in present and docs[i]]
            skipped += len(ids) - len(keep)
            if keep:
                texts = [docs[i] for i in keep]
                vectors = embed_texts(texts, dims)
                dst.add(
                    ids=[ids[i] for i in keep],
                    documents=texts,
                    metadatas=[metas[i] for i in keep],
                    embeddings=vectors,
                )
                embedded += len(keep)
            scanned += len(ids)
            _update(scanned=scanned, embedded=embedded, skipped=skipped)
            if scanned % 5000 < PAGE_SIZE:
                logger.info(f"[REEMBED] {source} -> {target}: scanned={scanned} embedded={embedded}")
            time.sleep(PAUSE_SECONDS)
        _update(state="done", finishedAt=datetime.now(timezone.utc).isoformat())
        logger.info(f"[REEMBED] done {source} -> {target}: embedded={embedded} skipped={skipped}")
    except Exception as exc:  # report, never crash the server
        logger.exception(f"[REEMBED] failed {source} -> {target}")
        _update(state="error", error=str(exc)[:500], finishedAt=datetime.now(timezone.utc).isoformat())
    finally:
        invalidate_snapshot(target)
