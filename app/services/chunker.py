"""
Document chunker with two modes:
  default – fast recursive text splitter (no LLM, zero cost, like OpenAI/Google internals)
  smart   – LLM-based semantic splitting with plain-text delimiter output
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from litellm import completion
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings

logger = logging.getLogger(__name__)

# Target average characters per chunk (used to estimate chunk count for the prompt)
AVERAGE_CHUNK_SIZE = 150

# Max characters sent to the LLM in a single chunking call (smart mode).
MAX_CHARS_PER_CALL = 12_000

# Overlap in characters between adjacent sections (smart mode)
SECTION_OVERLAP_CHARS = 500

# Default mode: target chunk size and overlap in characters
DEFAULT_CHUNK_SIZE = 800
DEFAULT_OVERLAP = 200

# Separators tried in order for the default recursive splitter
_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


# ─── Default mode: recursive text splitter (no LLM) ─────────────────────────────────

def _recursive_split(text: str, separators: list[str], chunk_size: int) -> list[str]:
    """Recursively split text using the first separator that produces chunks ≤ chunk_size."""
    if len(text) <= chunk_size or not separators:
        return [text]
    sep, rest = separators[0], separators[1:]
    parts = text.split(sep)
    chunks: list[str] = []
    current = ""
    for part in parts:
        candidate = (current + sep + part) if current else part
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            # part itself may be too long — recurse
            if len(part) > chunk_size:
                chunks.extend(_recursive_split(part, rest, chunk_size))
                current = ""
            else:
                current = part
    if current:
        chunks.append(current)
    return chunks


def _chunk_default(
    text: str, source: str, notebook_id: str, source_id: str
) -> list[dict]:
    """Fast recursive splitter with overlap — no LLM, zero cost."""
    base_chunks = _recursive_split(text, _SEPARATORS, DEFAULT_CHUNK_SIZE)
    result: list[dict] = []
    for i, chunk in enumerate(base_chunks):
        # Add overlap from the previous chunk's tail
        if i > 0:
            tail = base_chunks[i - 1][-DEFAULT_OVERLAP:]
            content = tail + chunk
        else:
            content = chunk
        result.append({
            "content": content,
            "metadata": {"notebookId": notebook_id, "sourceId": source_id, "source": source},
        })
    return result


# ─── Pre-split mode: one chunk per <<<ENTRY>>> marker ───────────────────────────

_ENTRY_DELIMITER = "<<<ENTRY>>>"


def _chunk_pre_split(
    text: str, source: str, notebook_id: str, source_id: str
) -> list[dict]:
    """Split on <<<ENTRY>>> markers — one chunk per structural entry (e.g. each JSON object)."""
    entries = [e.strip() for e in text.split(_ENTRY_DELIMITER) if e.strip()]
    logger.info(f"[chunker] pre-split mode: {len(entries)} entries for source={source_id}")
    return [
        {
            "content": entry,
            "metadata": {"notebookId": notebook_id, "sourceId": source_id, "source": source},
        }
        for entry in entries
    ]


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
        f"Together the chunks must cover the ENTIRE text with no gaps.\n\n"
        f"DOCUMENT:\n{text}\n\n"
        f"Return the chunks as plain text. Separate each chunk with exactly:\n"
        f"<<<CHUNK>>>\n"
        f"Output nothing else — no numbering, no labels, no markdown."
    )

    response = completion(
        model=settings.litellm_model,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.choices[0].message.content or ""
    parts = [p.strip() for p in raw.split("<<<CHUNK>>>") if p.strip()]
    return [
        {
            "content": part,
            "metadata": {
                "notebookId": notebook_id,
                "sourceId": source_id,
                "source": source,
            },
        }
        for part in parts
    ]


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
    mode: str | None = None,
) -> list[dict]:
    """
    Chunk a document. mode="default" (fast, no LLM) or mode="smart" (LLM-based).
    Defaults to settings.chunk_mode if not specified.
    Returns a list of dicts: {"content": str, "metadata": dict}
    """
    effective_mode = mode or settings.chunk_mode

    if _ENTRY_DELIMITER in text:
        logger.info(f"[chunker] pre-split mode for source={source_id}")
        return _chunk_pre_split(text, source, notebook_id, source_id)

    if effective_mode == "default":
        logger.info(f"[chunker] default mode for source={source_id}")
        return _chunk_default(text, source, notebook_id, source_id)

    # smart mode — LLM-based
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
