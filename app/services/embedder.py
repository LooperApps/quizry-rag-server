"""
Text embedding service using litellm.
GEMINI_API_KEY env var is read automatically by litellm for Google AI embeddings.
"""

import logging

from litellm import embedding as litellm_embedding
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings

logger = logging.getLogger(__name__)

EMBED_BATCH_SIZE = 100


@retry(
    wait=wait_exponential(multiplier=1, min=5, max=60),
    stop=stop_after_attempt(3),
    reraise=True,
)
def _embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed a single batch of texts (≤ EMBED_BATCH_SIZE)."""
    response = litellm_embedding(
        model=settings.embedding_model,
        input=texts,
    )
    return [item["embedding"] for item in response.data]


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
    return _embed_batch([text])[0]
