"""
LLM-based document chunker.

Ports the chunking logic from week5/pro_implementation/ingest.py.
Uses a ThreadPoolExecutor (instead of multiprocessing) for cross-platform
compatibility and to avoid issues with uvicorn's worker processes.
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from litellm import completion
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings

logger = logging.getLogger(__name__)

# Target average characters per chunk (used to estimate chunk count for the prompt)
AVERAGE_CHUNK_SIZE = 150

# Max characters sent to the LLM in a single chunking call.
# Larger texts are split into overlapping sections first.
MAX_CHARS_PER_CALL = 12_000

# Overlap in characters between adjacent sections passed to the LLM
SECTION_OVERLAP_CHARS = 500


# ─── Pydantic schema for structured LLM output ────────────────────────────────

class Chunk(BaseModel):
    headline: str = Field(
        description="A brief heading (a few words) most likely to be surfaced in a query"
    )
    summary: str = Field(
        description="2-4 sentences summarising the chunk to answer common questions"
    )
    original_text: str = Field(
        description="The original text of this chunk, exactly as it appears in the document"
    )


class Chunks(BaseModel):
    chunks: list[Chunk]


# ─── Core LLM call ────────────────────────────────────────────────────────────

@retry(
    wait=wait_exponential(multiplier=1, min=10, max=120),
    stop=stop_after_attempt(3),
    reraise=True,
)
def _call_llm_chunk(text: str, source: str, notebook_id: str, source_id: str) -> list[dict]:
    """
    Send one section of text to the LLM and return a list of chunk dicts.
    Retries up to 3 times with exponential back-off for rate-limit / transient errors.
    """
    how_many = max(1, len(text) // AVERAGE_CHUNK_SIZE)

    prompt = (
        f"You split a document into overlapping chunks for a Knowledge Base used by an "
        f"educational app. Students will query the Knowledge Base to get help studying.\n\n"
        f"Document source: {source}\n"
        f"Target approximately {how_many} chunks (adjust as needed). "
        f"Use ~25% overlap between consecutive chunks so context is preserved across boundaries.\n\n"
        f"For each chunk provide:\n"
        f"  headline  – a brief heading (a few words)\n"
        f"  summary   – 2-4 sentences summarising the chunk\n"
        f"  original_text – the exact original text (no changes)\n\n"
        f"Together the chunks must cover the ENTIRE text with no gaps.\n\n"
        f"DOCUMENT:\n{text}\n\n"
        f'Return a JSON object with this exact structure (raw JSON only, no markdown):\n'
        f'{{"chunks": [{{"headline": "...", "summary": "...", "original_text": "..."}}]}}'
    )

    response = completion(
        model=settings.litellm_model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )

    raw = Chunks.model_validate_json(response.choices[0].message.content)
    results = []
    for chunk in raw.chunks:
        content = f"{chunk.headline}\n\n{chunk.summary}\n\n{chunk.original_text}"
        results.append(
            {
                "content": content,
                "metadata": {
                    "notebookId": notebook_id,
                    "sourceId": source_id,
                    "source": source,
                    "headline": chunk.headline,
                },
            }
        )
    return results


# ─── Section splitter ─────────────────────────────────────────────────────────

def _split_into_sections(text: str, max_chars: int, overlap: int) -> list[str]:
    """
    Split `text` into sections of at most `max_chars` characters,
    with `overlap` characters of context carried over from the previous section.
    Splits preferably on paragraph boundaries.
    """
    if len(text) <= max_chars:
        return [text]

    paragraphs = text.split("\n\n")
    sections: list[str] = []
    current_parts: list[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para) + 2  # +2 for the "\n\n" we stripped
        if current_len + para_len > max_chars and current_parts:
            sections.append("\n\n".join(current_parts))
            # Keep the last ~`overlap` characters as context for the next section
            overlap_text = "\n\n".join(current_parts)[-overlap:]
            current_parts = [overlap_text, para]
            current_len = len(overlap_text) + para_len
        else:
            current_parts.append(para)
            current_len += para_len

    if current_parts:
        sections.append("\n\n".join(current_parts))

    return sections


# ─── Public API ───────────────────────────────────────────────────────────────

def chunk_document(
    text: str,
    source: str,
    notebook_id: str,
    source_id: str,
) -> list[dict]:
    """
    Chunk a full document text into overlapping, semantically meaningful pieces.

    For short texts (≤ MAX_CHARS_PER_CALL), a single LLM call is made.
    For larger texts, the document is split into overlapping sections and
    processed in parallel using a ThreadPoolExecutor.

    Returns a list of dicts: {"content": str, "metadata": dict}
    """
    sections = _split_into_sections(text, MAX_CHARS_PER_CALL, SECTION_OVERLAP_CHARS)

    if len(sections) == 1:
        return _call_llm_chunk(sections[0], source, notebook_id, source_id)

    logger.info(
        f"[chunker] Splitting {len(text):,} chars into {len(sections)} sections "
        f"for source={source_id}"
    )

    workers = min(settings.chunk_workers, len(sections))
    all_chunks: list[dict] = []
    errors: list[Exception] = []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_idx = {
            pool.submit(_call_llm_chunk, section, source, notebook_id, source_id): i
            for i, section in enumerate(sections)
        }
        # Collect in submission order for deterministic output
        ordered = [None] * len(sections)
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            exc = future.exception()
            if exc:
                logger.error(f"[chunker] Section {idx} failed: {exc}")
                errors.append(exc)
            else:
                ordered[idx] = future.result()

    if errors and all(r is None for r in ordered):
        raise errors[0]  # All sections failed — surface the first error

    for result in ordered:
        if result:
            all_chunks.extend(result)

    logger.info(f"[chunker] Produced {len(all_chunks)} chunks from {len(sections)} sections")
    return all_chunks
