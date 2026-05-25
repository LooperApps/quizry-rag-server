"""
File download and text extraction service.

Supported formats (matching index.js inferMimeType):
  pdf, txt, md, html/htm, csv, docx, pptx
  png, jpg, jpeg  → OCR via OpenAI Vision (gpt-4o-mini)
"""

import io
import base64
import logging
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)


async def download_bytes(url: str) -> bytes:
    """
    Download a file from a Firebase Storage download URL.
    The URL is a standard HTTPS URL (alt=media&token=...).
    """
    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.content


def extract_text(data: bytes, file_name: str) -> str:
    """
    Extract plain text from a file given its raw bytes and name.
    Returns an empty string for unsupported/binary-only formats.
    """
    ext = Path(file_name).suffix.lower().lstrip(".")

    extractors = {
        "pdf": _extract_pdf,
        "txt": _extract_plain,
        "md": _extract_plain,
        "csv": _extract_plain,
        "html": _extract_html,
        "htm": _extract_html,
        "docx": _extract_docx,
        "pptx": _extract_pptx,
        "json": _extract_plain,
        "jpg": _extract_image,
        "jpeg": _extract_image,
        "png": _extract_image,
        "webp": _extract_image,
    }

    extractor = extractors.get(ext)
    if extractor is None:
        logger.warning(f"Unsupported file extension '{ext}' for {file_name}, skipping text extraction")
        return ""

    try:
        text = extractor(data)
        logger.info(f"Extracted {len(text):,} chars from {file_name} ({ext})")
        return text
    except Exception as e:
        logger.error(f"Text extraction failed for {file_name}: {e}", exc_info=True)
        raise ValueError(f"Could not extract text from {file_name}: {e}") from e


# ─── Format-specific extractors ───────────────────────────────────────────────

def _extract_plain(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


def _extract_json_UNUSED(data: bytes) -> str:
    """
    Parse JSON and convert to clean readable text.
    If the JSON is an array of {major, university, requirements} objects
    (university requirements format), each entry becomes a labelled text block
    separated by double newlines — so the chunker never splits mid-entry.
    Falls back to raw UTF-8 text for any other JSON shape.
    """
    import json
    raw = data.decode("utf-8", errors="replace").lstrip("\ufeff")  # strip BOM
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("[extractor] _extract_json: not valid JSON, returning raw text")
        return raw  # not valid JSON — treat as plain text

    logger.info(f"[extractor] _extract_json: type={type(obj).__name__}, "
                f"len={len(obj) if isinstance(obj, (list, dict)) else 'n/a'}")

    # Resolve top-level dict: collect ALL values that are lists of university entries
    entries = obj
    if isinstance(obj, dict):
        all_entries: list = []
        for v in obj.values():
            if (isinstance(v, list) and v and isinstance(v[0], dict)
                    and "major" in v[0] and "university" in v[0] and "requirements" in v[0]):
                all_entries.extend(v)
        if all_entries:
            entries = all_entries
            logger.info(f"[extractor] _extract_json: merged {len(obj)} dict keys → {len(entries)} total entries")
        else:
            logger.warning(f"[extractor] _extract_json: dict has no matching list values, returning raw. keys={list(obj.keys())[:5]}")
            return raw

    # Detect university requirements format: list of {major, university, requirements}
    if (
        isinstance(entries, list)
        and entries
        and isinstance(entries[0], dict)
        and "major" in entries[0]
        and "university" in entries[0]
        and "requirements" in entries[0]
    ):
        blocks = []
        for entry in entries:
            major = entry.get("major", "").strip()
            university = entry.get("university", "").strip()
            reqs = entry.get("requirements", [])
            if isinstance(reqs, list):
                req_lines = "\n".join(f"- {r}" for r in reqs if r)
            else:
                req_lines = str(reqs)
            block = f"אוניברסיטה: {university}\nתחום: {major}\nתנאי קבלה:\n{req_lines}"
            blocks.append(block)
        logger.info(f"[extractor] _extract_json: produced {len(blocks)} university blocks")
        return "\n\n".join(blocks)

    # Fallback: raw
    logger.warning(f"[extractor] _extract_json: unknown JSON shape, returning raw. first_keys={list(entries[0].keys()) if isinstance(entries, list) and entries and isinstance(entries[0], dict) else 'n/a'}")
    return raw


def _extract_html(data: bytes) -> str:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(data, "html.parser")
    # Remove script and style tags
    for tag in soup(["script", "style"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


def _extract_pdf(data: bytes) -> str:
    """
    Try pypdf first (fast, handles most PDFs).
    Fall back to pdfminer for scanned/complex layouts.
    """
    text = _pdf_via_pypdf(data)
    if text.strip():
        return text
    logger.info("pypdf returned empty text, falling back to pdfminer")
    return _pdf_via_pdfminer(data)


def _pdf_via_pypdf(data: bytes) -> str:
    import pypdf
    reader = pypdf.PdfReader(io.BytesIO(data))
    pages = []
    for page in reader.pages:
        t = page.extract_text()
        if t:
            pages.append(t)
    return "\n\n".join(pages)


def _pdf_via_pdfminer(data: bytes) -> str:
    from pdfminer.high_level import extract_text_to_fp
    from pdfminer.layout import LAParams
    out = io.StringIO()
    extract_text_to_fp(io.BytesIO(data), out, laparams=LAParams(), output_type="text")
    return out.getvalue()


def _extract_docx(data: bytes) -> str:
    import docx  # python-docx
    doc = docx.Document(io.BytesIO(data))
    parts = []
    for para in doc.paragraphs:
        if para.text.strip():
            parts.append(para.text)
    # Also extract tables
    for table in doc.tables:
        for row in table.rows:
            row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
            if row_text:
                parts.append(row_text)
    return "\n\n".join(parts)


def _extract_pptx(data: bytes) -> str:
    from pptx import Presentation
    prs = Presentation(io.BytesIO(data))
    slides = []
    for i, slide in enumerate(prs.slides, 1):
        texts = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    t = para.text.strip()
                    if t:
                        texts.append(t)
        if texts:
            slides.append(f"[Slide {i}]\n" + "\n".join(texts))
    return "\n\n".join(slides)


def _extract_image(data: bytes) -> str:
    """
    OCR an image using Google Gemini via litellm (gemini-2.0-flash).
    Uses GEMINI_API_KEY env var — avoids OpenAI restricted-key scope issues.
    """
    from litellm import completion

    b64 = base64.b64encode(data).decode()
    # Detect mime type from magic bytes
    if data[:3] == b"\xff\xd8\xff":
        mime = "image/jpeg"
    elif data[:8] == b"\x89PNG\r\n\x1a\n":
        mime = "image/png"
    elif data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        mime = "image/webp"
    else:
        mime = "image/jpeg"  # safe fallback

    response = completion(
        model="gemini/gemini-2.5-flash",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Extract ALL text visible in this image exactly as written. "
                            "Preserve headings, bullet points, tables, and formatting. "
                            "If there is no text, describe the image content in detail instead."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    },
                ],
            }
        ],
        max_tokens=4096,
    )
    text = response.choices[0].message.content or ""
    logger.info(f"[OCR] Extracted {len(text):,} chars from image")
    return text
