"""
Document chunker with two modes:
  default – fast recursive text splitter (no LLM, zero cost, like OpenAI/Google internals)
  smart   – LLM-based semantic splitting with plain-text delimiter output

plus the page-aware chunker (`chunk_pages`, "pages_v1") used by collections
that carry `chunker: pages_v1` in their metadata.
"""

import logging
import math
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from litellm import completion
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings
from app.log_utils import get_trace_id, log_prompt, log_step

logger = logging.getLogger(__name__)

# Target average characters per chunk (used to estimate chunk count for the prompt)
AVERAGE_CHUNK_SIZE = 150

# Max characters sent to the LLM in a single chunking call (smart mode).
MAX_CHARS_PER_CALL = 12_000

# Overlap in characters between adjacent sections (smart mode)
SECTION_OVERLAP_CHARS = 500

# Default mode: target chunk size and overlap in characters.
# Hebrew is ~1 char/token, so 800 tokens ≈ 800 chars, 400 tokens ≈ 400 chars.
DEFAULT_CHUNK_SIZE = 800
DEFAULT_OVERLAP = 400

# Separators tried in order for the default recursive splitter
_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


# ─── Default mode: recursive text splitter (no LLM) ─────────────────────────────────

def _recursive_split(text: str, separators: list[str], chunk_size: int) -> list[str]:
    """Recursively split text using the first separator that produces chunks ≤ chunk_size."""
    if len(text) <= chunk_size or not separators:
        return [text]
    sep, rest = separators[0], separators[1:]
    if sep == "":
        # Last resort: no separator left (e.g. a PDF whose text extracted
        # without spaces). str.split("") raises, so cut fixed-size pieces.
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
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
    trace = get_trace_id()
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
    log_step(
        logger, "CHUNKER_DEFAULT",
        f"source={source_id} text_len={len(text)} produced={len(result)} chunks "
        f"chunk_size={DEFAULT_CHUNK_SIZE} overlap={DEFAULT_OVERLAP}",
    )
    return result


# ─── Page-aware chunker (pages_v1) ────────────────────────────────────────────
#
# Chunks follow pages: a page longer than the target is split into near-equal
# pieces at the best boundary (paragraph > line > sentence > word) near each
# ideal cut, short neighbouring pages are packed together up to the target,
# and every chunk records the page span of its own text. Each chunk after the
# first starts with up to PAGES_OVERLAP_CHARS of the previous chunk's text,
# cut at a word boundary, so content is always a contiguous run of the source
# (pages joined by a blank line). Sizes are in characters.

PAGES_TARGET_CHARS = 1400
PAGES_OVERLAP_CHARS = 150
PAGES_MIN_CHARS = 200
PAGES_MAX_CHARS = 2000
# How far from the ideal cut a boundary may be, as a share of the target.
_PAGES_CUT_SLACK = 0.25
_SENTENCE_END = re.compile(r"[.!?؟۔](?=\s)")


@dataclass
class _Piece:
    page: int
    text: str
    sep: str  # what separated it from the previous piece in the source


def _cut_near(text: str, start: int, ideal: int) -> tuple[int, int]:
    """(end of this piece, start of the next) for a cut close to `ideal`."""
    slack = int(PAGES_TARGET_CHARS * _PAGES_CUT_SLACK)
    lo = max(start + PAGES_MIN_CHARS, ideal - slack)
    hi = min(start + PAGES_TARGET_CHARS, ideal + slack, len(text))

    window = text[lo:hi]

    def nearest(offsets: list[int]) -> int | None:
        return lo + min(offsets, key=lambda o: abs(lo + o - ideal)) if offsets else None

    for sep in ("\n\n", "\n"):
        found = nearest([m.start() for m in re.finditer(re.escape(sep), window)])
        if found is not None:
            return found, found + len(sep)
    found = nearest([m.end() for m in _SENTENCE_END.finditer(window)])
    if found is None:
        found = nearest([i for i, ch in enumerate(window) if ch == " "])
    if found is None:
        return ideal, ideal  # no boundary at all: hard cut, nothing consumed
    nxt = found
    while nxt < len(text) and text[nxt].isspace():
        nxt += 1
    return found, nxt


def _split_page(page: int, text: str, sep: str) -> list[_Piece]:
    """Split one page's text into near-equal pieces of at most the target size."""
    pieces: list[_Piece] = []
    start = 0
    while len(text) - start > PAGES_TARGET_CHARS:
        parts_left = math.ceil((len(text) - start) / PAGES_TARGET_CHARS)
        ideal = start + (len(text) - start) // parts_left
        end, nxt = _cut_near(text, start, ideal)
        pieces.append(_Piece(page, text[start:end].rstrip(), sep))
        gap = text[end:nxt]
        sep = "\n\n" if "\n\n" in gap else ("\n" if "\n" in gap else (" " if gap else ""))
        start = nxt
    pieces.append(_Piece(page, text[start:], sep))
    return [p for p in pieces if p.text.strip()]


def _overlap_tail(text: str) -> str:
    """The last PAGES_OVERLAP_CHARS of `text`, starting at a word boundary."""
    if len(text) <= PAGES_OVERLAP_CHARS:
        return text
    tail = text[-PAGES_OVERLAP_CHARS:]
    if text[-PAGES_OVERLAP_CHARS - 1].isspace():
        return tail.lstrip()
    for i, ch in enumerate(tail):
        if ch.isspace():
            return tail[i:].lstrip()
    return tail  # one long token: keep it whole


def chunk_pages(
    pages: list[tuple[int, str]],
    source: str,
    notebook_id: str,
    source_id: str,
) -> list[dict]:
    """
    Page-aware chunking (pages_v1). `pages` is [(page number, text)] with text
    already normalised; empty pages are skipped. Returns
    [{"content", "metadata": {notebookId, sourceId, source, chunkIndex,
    pageStart, pageEnd}}] where the page span covers the chunk's own text
    (not the overlap borrowed from the previous chunk).
    """
    pieces: list[_Piece] = []
    for page, text in pages:
        text = (text or "").strip()
        if text:
            pieces.extend(_split_page(int(page), text, "\n\n" if pieces else ""))

    groups: list[list[_Piece]] = []
    current: list[_Piece] = []
    size = 0
    for piece in pieces:
        if current:
            joined = size + len(piece.sep) + len(piece.text)
            if joined <= PAGES_TARGET_CHARS or size < PAGES_MIN_CHARS:
                current.append(piece)
                size = joined
                continue
            groups.append(current)
        current, size = [piece], len(piece.text)
    if current:
        if groups and size < PAGES_MIN_CHARS:
            groups[-1].extend(current)  # a short tail joins the previous chunk
        else:
            groups.append(current)

    result: list[dict] = []
    previous = ""
    for index, group in enumerate(groups):
        base = group[0].text + "".join(p.sep + p.text for p in group[1:])
        content = base
        if previous:
            tail = _overlap_tail(previous)
            if len(tail) + len(group[0].sep) + len(base) <= PAGES_MAX_CHARS:
                content = tail + (group[0].sep or "") + base
        result.append({
            "content": content,
            "metadata": {
                "notebookId": notebook_id,
                "sourceId": source_id,
                "source": source,
                "chunkIndex": index,
                "pageStart": group[0].page,
                "pageEnd": group[-1].page,
            },
        })
        previous = base

    log_step(
        logger, "CHUNKER_PAGES",
        f"source={source_id} pages={len(pages)} produced={len(result)} chunks "
        f"target={PAGES_TARGET_CHARS} overlap={PAGES_OVERLAP_CHARS}",
    )
    return result


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
    trace = get_trace_id()
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

    log_step(
        logger, "CHUNKER_LLM",
        f"Calling LLM for chunking source={source_id} "
        f"text_len={len(text)} target_chunks={how_many} model={settings.litellm_model}",
    )

    response = completion(
        model=settings.litellm_model,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.choices[0].message.content or ""

    log_prompt(
        logger, "_call_llm_chunk",
        prompt=prompt,
        response=raw,
        model=settings.litellm_model,
        source_id=source_id,
    )

    parts = [p.strip() for p in raw.split("<<<CHUNK>>>") if p.strip()]
    result = [
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
    log_step(
        logger, "CHUNKER_LLM",
        f"Produced {len(result)} chunks from LLM for source={source_id}",
    )
    return result


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
    trace = get_trace_id()
    effective_mode = mode or settings.chunk_mode

    log_step(
        logger, "CHUNKER",
        f"mode={effective_mode} source={source_id} text_len={len(text)} "
        f"notebook={notebook_id}",
    )

    if effective_mode == "default":
        result = _chunk_default(text, source, notebook_id, source_id)
        log_step(
            logger, "CHUNKER",
            f"mode=default done — {len(result)} chunks for source={source_id}",
        )
        return result

    # smart mode — LLM-based
    sections = _split_into_sections(text, MAX_CHARS_PER_CALL, SECTION_OVERLAP_CHARS)
    log_step(
        logger, "CHUNKER",
        f"mode=smart text_len={len(text)} split_into={len(sections)} sections "
        f"max_chars_per_call={MAX_CHARS_PER_CALL} overlap={SECTION_OVERLAP_CHARS}",
    )

    if len(sections) == 1:
        result = _call_llm_chunk(sections[0], source, notebook_id, source_id)
        log_step(
            logger, "CHUNKER",
            f"mode=smart done — {len(result)} chunks for source={source_id}",
        )
        return result

    log_step(
        logger, "CHUNKER",
        f"Processing {len(sections)} sections with {settings.chunk_workers} workers",
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
        log_step(logger, "CHUNKER", f"All {len(sections)} sections FAILED — raising first error")
        raise errors[0]  # All sections failed — surface the first error

    for result in ordered:
        if result:
            all_chunks.extend(result)

    log_step(
        logger, "CHUNKER",
        f"mode=smart done — {len(all_chunks)} chunks from {len(sections)} sections "
        f"for source={source_id}, errors={len(errors)}",
    )
    return all_chunks
