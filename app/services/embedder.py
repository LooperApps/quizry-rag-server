"""
Google AI text embedding service with batching and retry.
"""

import logging
import threading

from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings

logger = logging.getLogger(__name__)

EMBED_BATCH_SIZE = 100

_client: genai.Client | None = None
_lock = threading.Lock()


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                _client = genai.Client(api_key=settings.gemini_api_key)
                try:
                    models = [m.name for m in _client.models.list()]
                    logger.info(f"[embedder] Available models: {models}")
                except Exception as e:
                    logger.warning(f"[embedder] Could not list models: {e}")
    return _client


@retry(
    wait=wait_exponential(multiplier=1, min=5, max=60),
    stop=stop_after_attempt(3),
    reraise=True,
)
def _embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed a single batch of texts (≤ EMBED_BATCH_SIZE)."""
    result = _get_client().models.embed_content(
        model=settings.embedding_model,
        contents=texts,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
    )
    return [e.values for e in result.embeddings]


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
    result = _get_client().models.embed_content(
        model=settings.embedding_model,
        contents=text,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    return result.embeddings[0].values
