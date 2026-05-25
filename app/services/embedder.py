"""
Google AI text embedding service with batching and retry.
"""

import logging
import threading

import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings

logger = logging.getLogger(__name__)

EMBED_BATCH_SIZE = 100

_configured = False
_lock = threading.Lock()


def _ensure_configured() -> None:
    global _configured
    if not _configured:
        with _lock:
            if not _configured:
                genai.configure(api_key=settings.gemini_api_key)
                _configured = True


@retry(
    wait=wait_exponential(multiplier=1, min=5, max=60),
    stop=stop_after_attempt(3),
    reraise=True,
)
def _embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed a single batch of texts (≤ EMBED_BATCH_SIZE)."""
    _ensure_configured()
    result = genai.embed_content(
        model=settings.embedding_model,
        content=texts,
        task_type="retrieval_document",
    )
    return result["embedding"]


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of texts, batching as needed.
    Returns embeddings in the same order as the input list.
    """
    if not texts:
        return []

    all_embeddings: list[list[float]] = []
    total_batches = (len(texts) - 1) // EMBED_BATCH_SIZE + 1

    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        batch_num = i // EMBED_BATCH_SIZE + 1
        logger.debug(f"[embedder] Embedding batch {batch_num}/{total_batches} ({len(batch)} texts)")
        all_embeddings.extend(_embed_batch(batch))

    logger.info(f"[embedder] Embedded {len(texts)} texts in {total_batches} batch(es)")
    return all_embeddings


def embed_query(text: str) -> list[float]:
    """Embed a single query string."""
    _ensure_configured()
    result = genai.embed_content(
        model=settings.embedding_model,
        content=text,
        task_type="retrieval_query",
    )
    return result["embedding"]
