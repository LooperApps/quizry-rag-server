"""
Text embedding service using litellm.
GEMINI_API_KEY env var is read automatically by litellm for Google AI embeddings.

Collections created with behaviour flags (app/db/chroma.py) get two extras;
legacy collections get neither, so their vectors stay exactly as built:
  normalized  every vector is L2-normalised (gemini-embedding-001 only
              normalises its full 3072-dim output)
  task_types  documents embed as RETRIEVAL_DOCUMENT, queries as RETRIEVAL_QUERY
"""

import logging
import time

import numpy as np
from litellm import embedding as litellm_embedding
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings
from app.db.chroma import LEGACY_DIMENSIONS, CollectionFlags
from app.log_utils import get_trace_id, log_step, log_step_data

logger = logging.getLogger(__name__)

EMBED_BATCH_SIZE = 100

TASK_DOCUMENT = "RETRIEVAL_DOCUMENT"
TASK_QUERY = "RETRIEVAL_QUERY"


def l2_normalize(vectors: list[list[float]]) -> list[list[float]]:
    """Scale each vector to unit length (zero vectors are left as they are)."""
    if not vectors:
        return []
    arr = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return (arr / np.where(norms == 0, 1.0, norms)).tolist()


@retry(
    wait=wait_exponential(multiplier=1, min=5, max=60),
    stop=stop_after_attempt(3),
    reraise=True,
)
def _embed_batch(
    texts: list[str],
    dimensions: int | None = None,
    task_type: str | None = None,
) -> list[list[float]]:
    """Embed a single batch of texts (≤ EMBED_BATCH_SIZE)."""
    response = litellm_embedding(
        model=settings.embedding_model,
        input=texts,
        # Omitted at the model's full size so legacy collections get exactly
        # the vectors they were built with.
        **({"dimensions": dimensions} if dimensions and dimensions != LEGACY_DIMENSIONS else {}),
        # Only flagged collections ask for a task type (litellm -> taskType).
        **({"task_type": task_type} if task_type else {}),
    )
    return [item["embedding"] for item in response.data]


def embed_documents(texts: list[str], flags: CollectionFlags) -> list[list[float]]:
    """Embed chunks for storage in a collection, honouring its flags."""
    vectors = embed_texts(texts, flags.dimensions, task_type=TASK_DOCUMENT if flags.task_types else None)
    return l2_normalize(vectors) if flags.normalized else vectors


def embed_queries(texts: list[str], flags: CollectionFlags) -> list[list[float]]:
    """Embed search queries for a collection (one batch call), honouring its flags."""
    vectors = embed_texts(texts, flags.dimensions, task_type=TASK_QUERY if flags.task_types else None)
    return l2_normalize(vectors) if flags.normalized else vectors


def embed_texts(
    texts: list[str],
    dimensions: int | None = None,
    task_type: str | None = None,
) -> list[list[float]]:
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
        all_embeddings.extend(_embed_batch(batch, dimensions, task_type))

    log_step(
        logger, "EMBEDDER",
        f"done — {len(texts)} texts → {len(all_embeddings)} embeddings",
    )
    return all_embeddings


def embed_query(
    text: str,
    dimensions: int | None = None,
    flags: CollectionFlags | None = None,
) -> list[float]:
    """Embed a single query string (with the collection's flags, when given)."""
    trace = get_trace_id()
    t0 = time.perf_counter()
    log_step(
        logger, "EMBEDDER",
        f"embedding query model={settings.embedding_model} text={text!r}",
    )
    if flags is not None and (flags.task_types or flags.normalized):
        result = embed_queries([text], flags)[0]
    else:
        result = _embed_batch([text], dimensions)[0]
    elapsed = int((time.perf_counter() - t0) * 1000)
    log_step(
        logger, "EMBEDDER",
        f"query done elapsed={elapsed}ms dims={len(result)}",
    )
    return result
