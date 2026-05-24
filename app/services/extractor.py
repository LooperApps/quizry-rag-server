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
    OCR an image using OpenAI Vision (gpt-4o-mini).
    Returns the extracted text, or raises ValueError if OCR yields nothing.
    """
    from openai import OpenAI
    from app.config import settings

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

    client = OpenAI(api_key=settings.openai_api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
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
                        "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "high"},
                    },
                ],
            }
        ],
        max_tokens=4096,
    )
    text = response.choices[0].message.content or ""
    logger.info(f"[OCR] Extracted {len(text):,} chars from image")
    return text
