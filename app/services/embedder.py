"""
Text embedding service using litellm.
GEMINI_API_KEY env var is read automatically by litellm for Google AI embeddings.
"""

import logging
import time

from litellm import embedding as litellm_embedding
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings
from app.log_utils import get_trace_id, log_step, log_step_data

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
    trace = get_trace_id()
    if not texts:
        log_step(logger, "EMBEDDER", "No texts to embed — returning empty")
        return []

    all_embeddings: list[list[float]] = []
    total_batches = (len(texts) - 1) // EMBED_BATCH_SIZE + 1

    log_step(
        logger, "EMBEDDER",
        f"embedding {len(texts)} texts in {total_batches} batch(es) "
        f"model={settings.embedding_model}",
    )

    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        batch_num = i // EMBED_BATCH_SIZE + 1
        log_step(
            logger, "EMBEDDER",
            f"batch {batch_num}/{total_batches} ({len(batch)} texts)",
        )
        # Log text previews for each batch
        for j, t in enumerate(batch):
            logger.debug(
                f"[EMBEDDER] trace={trace} batch[{batch_num}] text[{j}] "
                f"preview={t[:150]!r}"
            )
        all_embeddings.extend(_embed_batch(batch))

    log_step(
        logger, "EMBEDDER",
        f"done — {len(texts)} texts → {len(all_embeddings)} embeddings",
    )
    return all_embeddings


def embed_query(text: str) -> list[float]:
    """Embed a single query string."""
    trace = get_trace_id()
    t0 = time.perf_counter()
    log_step(
        logger, "EMBEDDER",
        f"embedding query model={settings.embedding_model} text={text!r}",
    )
    result = _embed_batch([text])[0]
    elapsed = int((time.perf_counter() - t0) * 1000)
    log_step(
        logger, "EMBEDDER",
        f"query done elapsed={elapsed}ms dims={len(result)}",
    )
    return result
