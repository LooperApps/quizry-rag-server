"""
File download and text extraction service.

Supported formats (matching index.js inferMimeType):
  pdf, txt, md, html/htm, csv, docx, pptx, json
  png, jpg, jpeg, webp, gif (+ bmp, tif/tiff, heic/heif, converted first)
      → OCR via DeepSeek vision (app/services/vision.py), for every collection

Two entry points:
  extract_text   whole-document text, exactly as before (pypdf, pdfminer only
                 when pypdf finds nothing). Collections without flags use it.
  extract_pages  page-wise text for flagged collections and POST /extract:
                 each PDF page's text layer is graded (app/services/text_quality),
                 junk/scanned pages are OCR'd and pages with large figures get
                 figure descriptions, both via DeepSeek within a page budget.
"""

import contextvars
import io
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import httpx

from app.config import settings
from app.log_utils import get_trace_id, log_step, log_step_data, redact_url
from app.services import vision
from app.services.text_quality import TextQuality, assess_text, clean_text

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = frozenset({"jpg", "jpeg", "png", "webp", "gif", "bmp", "tif", "tiff", "heic", "heif"})


async def download_bytes(url: str) -> bytes:
    """
    Download a file from a Firebase Storage download URL.
    The URL is a standard HTTPS URL (alt=media&token=...).
    """
    trace = get_trace_id()
    log_step(logger, "DOWNLOAD", f"Downloading from URL (length hidden)")
    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
        response = await client.get(url)
        if not response.is_success:
            # raise_for_status()'s message embeds the full URL, download token included.
            raise RuntimeError(
                f"Download failed: HTTP {response.status_code} for {redact_url(url)}"
            )
        content = response.content
        log_step(
            logger, "DOWNLOAD",
            f"Downloaded {len(content):,} bytes (status={response.status_code})",
        )
        return content


def extract_text(data: bytes, file_name: str) -> str:
    """
    Extract plain text from a file given its raw bytes and name.
    Returns an empty string for unsupported/binary-only formats.
    """
    trace = get_trace_id()
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
        "json": _extract_json,
        **{image_ext: _extract_image for image_ext in IMAGE_EXTENSIONS},
    }

    extractor = extractors.get(ext)
    if extractor is None:
        log_step(
            logger, "EXTRACT",
            f"Unsupported extension '{ext}' for {file_name} — skipping",
        )
        return ""

    try:
        log_step(logger, "EXTRACT", f"Extracting text from {file_name} (ext={ext})")
        text = extractor(data)
        log_step(
            logger, "EXTRACT",
            f"Extracted {len(text):,} chars from {file_name} ({ext})",
        )
        return text
    except Exception as e:
        log_step(
            logger, "EXTRACT",
            f"FAILED for {file_name}: {e}",
        )
        logger.error(f"Text extraction failed for {file_name}: {e}", exc_info=True)
        raise ValueError(f"Could not extract text from {file_name}: {e}") from e


# ─── Format-specific extractors ───────────────────────────────────────────────

def _extract_plain(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


def _collect_strings(obj) -> list[str]:
    """Recursively collect all string leaf values from any JSON object."""
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, list):
        return [s for item in obj for s in _collect_strings(item)]
    if isinstance(obj, dict):
        return [s for v in obj.values() for s in _collect_strings(v)]
    return []


def _university_entries(obj) -> list[dict]:
    """Collect {major, university, requirements} entries from a list or dict-of-lists."""
    def is_entry_list(v) -> bool:
        return (isinstance(v, list) and v and isinstance(v[0], dict)
                and "major" in v[0] and "university" in v[0] and "requirements" in v[0])

    if is_entry_list(obj):
        return obj
    if isinstance(obj, dict):
        entries: list[dict] = []
        for v in obj.values():
            if is_entry_list(v):
                entries.extend(v)
        return entries
    return []


def _extract_json(data: bytes) -> str:
    """
    Extract clean text from a JSON file.

    University-requirements files ({major, university, requirements} entries)
    become labelled text blocks separated by double newlines, so the chunker
    never splits mid-entry and retrieval stays precise. Any other JSON shape
    falls back to recursively collecting all string values (no syntax noise).
    """
    import json
    raw = data.decode("utf-8", errors="replace").lstrip("﻿")
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("[extractor] _extract_json: not valid JSON, returning raw text")
        return raw

    entries = _university_entries(obj)
    if entries:
        blocks = []
        for entry in entries:
            major = str(entry.get("major", "")).strip()
            university = str(entry.get("university", "")).strip()
            reqs = entry.get("requirements", [])
            if isinstance(reqs, list):
                req_lines = "\n".join(f"- {r}" for r in reqs if r)
            else:
                req_lines = str(reqs)
            blocks.append(f"אוניברסיטה: {university}\nתחום: {major}\nתנאי קבלה:\n{req_lines}")
        logger.info(f"[extractor] _extract_json: produced {len(blocks)} university blocks")
        return "\n\n".join(blocks)

    strings = _collect_strings(obj)
    logger.info(f"[extractor] _extract_json: collected {len(strings)} string values from JSON")
    return "\n".join(s for s in strings if s.strip())


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
    OCR an image with DeepSeek vision (the canonical OCR prompt; figures come
    back as [وصف صورة] lines). Images never go to Gemini.
    """
    log_step(logger, "OCR", f"Calling DeepSeek OCR data_size={len(data)} bytes")
    text = vision.ocr_image(data, label="image").text
    log_step(logger, "OCR", f"Extracted {len(text):,} chars from image")
    return text


# ─── Page-wise extraction (flagged collections, POST /extract) ────────────────

# pdfium is not thread-safe, not even across separate documents: every pdfium
# call in this process happens under this lock. DeepSeek calls run outside it.
_PDFIUM_LOCK = threading.Lock()

# An image must cover this share of the page to count as a figure.
FIGURE_MIN_AREA = 0.08
# The same image (pixel size + stream length) on this many pages is decoration.
REPEATED_IMAGE_PAGES = 3
# Luminance std-dev (0-255) below which an image is a faint background tint.
# Measured: background tiles 2-15, real figures 23-59.
FIGURE_MIN_CONTRAST = 18.0
# Images larger than this are not decoded for the contrast check (cost/RAM);
# at that size they are scans or photos, not tints.
CONTRAST_MAX_PIXELS = 4_000_000
# Vector paths on a page with no text: outlined text or a drawing worth OCR.
VISUAL_MIN_PATHS = 50
RENDER_DPI = 150
RENDER_MAX_SIDE = 2000
RENDER_JPEG_QUALITY = 80


@dataclass
class ExtractedPage:
    page: int
    text: str
    method: str = "text"   # text | ocr | text+vision
    images: int = 0        # figures detected on the page


@dataclass
class PageExtraction:
    pages: list[ExtractedPage]
    page_count: int
    vision_pages: int = 0  # pages whose text came from (or was extended by) DeepSeek
    truncated: bool = False  # some page needed vision but the page budget ran out

    @property
    def method(self) -> str:
        """Summary for the source doc: text, ocr or mixed."""
        methods = {p.method for p in self.pages if p.text}
        if methods <= {"text"}:
            return "text"
        return "ocr" if methods == {"ocr"} else "mixed"


@dataclass
class _PageWork:
    page: int
    quality: TextQuality
    figures: int = 0
    visual: bool = False
    result: vision.VisionResult | None = None
    kind: str = ""  # ocr | describe, when a vision call was planned


def extract_pages(
    data: bytes,
    file_name: str,
    *,
    vision_enabled: bool = True,
    max_vision_pages: int | None = None,
) -> PageExtraction:
    """
    Extract text page by page. PDFs are graded and repaired with DeepSeek
    (when `vision_enabled`, up to `max_vision_pages` calls); images are OCR'd;
    every other supported format becomes a single page. Text is cleaned with
    text_quality.clean_text. Raises ValueError for unsupported or unreadable
    files and vision.VisionError when OCR was the only way to get text and it
    failed.
    """
    ext = Path(file_name).suffix.lower().lstrip(".")
    budget = settings.vision_max_pages if max_vision_pages is None else max_vision_pages
    budget = max(0, min(budget, settings.vision_max_pages))

    if ext == "pdf":
        try:
            return _extract_pdf_pages(data, file_name, vision_enabled, budget)
        except (ValueError, vision.VisionError):
            raise
        except Exception as exc:
            logger.error(f"Page extraction failed for {file_name}: {exc}", exc_info=True)
            raise ValueError(f"Could not extract text from {file_name}: {exc}") from exc

    if ext in IMAGE_EXTENSIONS:
        result = vision.ocr_image(data, label=file_name)
        return PageExtraction(
            pages=[ExtractedPage(page=1, text=clean_text(result.text), method="ocr")],
            page_count=1,
            vision_pages=1,
        )

    text = extract_text(data, file_name)
    if not text and ext not in _TEXT_EXTENSIONS:
        raise ValueError(f"Unsupported file type '.{ext}' for {file_name}")
    return PageExtraction(pages=[ExtractedPage(page=1, text=clean_text(text))], page_count=1)


_TEXT_EXTENSIONS = frozenset({"txt", "md", "csv", "html", "htm", "docx", "pptx", "json"})


def _extract_pdf_pages(data: bytes, file_name: str, vision_enabled: bool, budget: int) -> PageExtraction:
    """
    pdfium's text layer per page (measured on the Palestine books: far fewer
    U+FFFD and no split lam-alef compared with pypdf, and it stays fast and
    small on 100+ MB files), graded, then repaired with DeepSeek. pypdf's text
    is the fallback only when pdfium cannot open the file — then there is no
    rendering, so no vision either.
    """
    import pypdfium2 as pdfium

    try:
        with _PDFIUM_LOCK:
            doc = pdfium.PdfDocument(data)
    except Exception as exc:
        texts = _pdf_texts(data)
        if texts is None:
            raise ValueError(f"Could not open PDF {file_name}: {exc}") from exc
        log_step(logger, "EXTRACT_PAGES", f"pdfium cannot open {file_name} ({exc}); pypdf text only")
        pages = []
        for i, raw in enumerate(texts):
            quality = assess_text(raw)
            pages.append(ExtractedPage(page=i + 1, text=quality.text if quality.usable else ""))
        return PageExtraction(pages=pages, page_count=len(texts))

    try:
        with _PDFIUM_LOCK:
            page_count = len(doc)
            texts = [_pdfium_text(doc, i) for i in range(page_count)]
            layouts = _pdf_layouts(doc) if vision_enabled else [(0, False)] * page_count

        work = [
            _PageWork(page=i + 1, quality=assess_text(texts[i] if i < len(texts) else ""),
                      figures=layouts[i][0], visual=layouts[i][1])
            for i in range(page_count)
        ]
        truncated = False
        if vision_enabled:
            truncated = _plan_vision(work, budget)
            _run_vision(doc, work, file_name)
    finally:
        with _PDFIUM_LOCK:
            doc.close()

    pages: list[ExtractedPage] = []
    vision_pages = failed = 0
    for w in work:
        q = w.quality
        text, method = (q.text if q.usable else ""), "text"
        if w.kind and w.result is None:
            failed += 1
        elif w.kind == "ocr":
            text, method = clean_text(w.result.text), "ocr"
            vision_pages += 1
        elif w.kind == "describe":
            vision_pages += 1
            figures = _figure_lines(w.result.text)
            if figures:
                text, method = f"{text}\n\n{figures}".strip(), "text+vision"
        pages.append(ExtractedPage(page=w.page, text=text, method=method, images=w.figures))

    statuses: dict[str, int] = {}
    for w in work:
        statuses[w.quality.status] = statuses.get(w.quality.status, 0) + 1
    planned = sum(1 for w in work if w.kind)
    log_step(
        logger, "EXTRACT_PAGES",
        f"{file_name}: pages={page_count} text_layer={statuses} vision_planned={planned} "
        f"vision_ok={vision_pages} vision_failed={failed} truncated={truncated} budget={budget}",
    )
    if planned and not vision_pages and not any(p.text for p in pages):
        raise vision.VisionError(
            f"DeepSeek OCR failed for all {planned} page(s) of {file_name}, "
            "and the PDF has no usable text layer"
        )
    return PageExtraction(pages=pages, page_count=page_count, vision_pages=vision_pages, truncated=truncated)


def _pdf_texts(data: bytes) -> list[str] | None:
    """pypdf text per page ("" for a page that fails); None if pypdf cannot open the file."""
    import pypdf

    try:
        reader = pypdf.PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            reader.decrypt("")
        pages = reader.pages
        count = len(pages)
    except Exception as exc:
        log_step(logger, "EXTRACT_PAGES", f"pypdf cannot open the PDF either ({exc})")
        return None
    texts = []
    for i in range(count):
        try:
            texts.append(pages[i].extract_text() or "")
        except Exception as exc:
            log_step(logger, "EXTRACT_PAGES", f"pypdf failed on page {i + 1}: {exc}")
            texts.append("")
    return texts


def _pdfium_text(doc, index: int) -> str:
    """One page's text layer ("" when the page cannot be read). Caller holds the lock."""
    try:
        page = doc[index]
    except Exception as exc:
        log_step(logger, "EXTRACT_PAGES", f"pdfium cannot load page {index + 1}: {exc}")
        return ""
    try:
        textpage = page.get_textpage()
        try:
            getter = getattr(textpage, "get_text_bounded", None) or textpage.get_text_range
            return getter() or ""
        finally:
            textpage.close()
    except Exception as exc:
        log_step(logger, "EXTRACT_PAGES", f"pdfium text failed on page {index + 1}: {exc}")
        return ""
    finally:
        page.close()


# pypdfium2 4.x and 5.x spell a few helpers differently.
def _obj_bounds(obj) -> tuple[float, float, float, float]:
    return (getattr(obj, "get_bounds", None) or obj.get_pos)()


def _image_px(obj) -> tuple[int, int]:
    return (getattr(obj, "get_px_size", None) or obj.get_size)()


def _pdf_layouts(doc) -> list[tuple[int, bool]]:
    """
    (figures, visual) per page. Caller holds _PDFIUM_LOCK.

    A figure is an image covering >= FIGURE_MIN_AREA of the page that is not
    repeated on REPEATED_IMAGE_PAGES pages and is not a faint tint. A page is
    "visual" — worth OCR even without a text layer — when it has a figure or
    enough vector paths to be outlined text or a drawing.
    """
    per_page: list[tuple[list[tuple], int]] = []
    key_pages: dict[tuple, int] = {}
    contrast: dict[tuple, float] = {}
    for i in range(len(doc)):
        page = doc[i]
        try:
            images, paths = _page_large_images(page, contrast)
        except Exception as exc:  # a malformed page must not sink the document
            log_step(logger, "EXTRACT_PAGES", f"layout scan failed on page {i + 1}: {exc}")
            images, paths = [], 0
        finally:
            page.close()
        per_page.append((images, paths))
        for key in {key for key, _ in images}:
            key_pages[key] = key_pages.get(key, 0) + 1

    layouts = []
    for images, paths in per_page:
        figures = sum(
            1 for key, _ in images
            if key_pages[key] < REPEATED_IMAGE_PAGES and contrast.get(key, 255.0) >= FIGURE_MIN_CONTRAST
        )
        layouts.append((figures, figures > 0 or paths >= VISUAL_MIN_PATHS))
    return layouts


def _page_large_images(page, contrast: dict[tuple, float]) -> tuple[list[tuple], int]:
    """Large images on one page as (key, area share), plus the vector-path count."""
    import pypdfium2.raw as pdfium_c

    left, bottom, right, top = page.get_bbox()
    page_area = max(1.0, (right - left) * (top - bottom))
    images: list[tuple] = []
    paths = 0

    def walk(form, matrices, level):
        nonlocal paths
        objects = page.get_objects(max_depth=1, form=form) if form is not None else page.get_objects(max_depth=1)
        for obj in objects:
            kind = obj.type
            if kind == pdfium_c.FPDF_PAGEOBJ_PATH:
                paths += 1
            elif kind == pdfium_c.FPDF_PAGEOBJ_FORM and level < 6:
                walk(obj, [obj.get_matrix()] + matrices, level + 1)
            elif kind == pdfium_c.FPDF_PAGEOBJ_IMAGE:
                x0, y0, x1, y1 = _obj_bounds(obj)
                corners = [(x0, y0), (x1, y0), (x0, y1), (x1, y1)]
                # Bounds of objects inside Form XObjects are in form space.
                for m in matrices:
                    corners = [(m.a * x + m.c * y + m.e, m.b * x + m.d * y + m.f) for x, y in corners]
                xs = [x for x, _ in corners]
                ys = [y for _, y in corners]
                w = max(0.0, min(right, max(xs)) - max(left, min(xs)))
                h = max(0.0, min(top, max(ys)) - max(bottom, min(ys)))
                area = w * h / page_area
                if area < FIGURE_MIN_AREA:
                    continue
                try:
                    px = tuple(_image_px(obj))
                except Exception:
                    px = (0, 0)
                key = (px, pdfium_c.FPDFImageObj_GetImageDataRaw(obj.raw, None, 0))
                if key not in contrast:
                    contrast[key] = _image_contrast(obj, px)
                images.append((key, area))

    walk(None, [], 0)
    return images, paths


def _image_contrast(obj, px: tuple[int, int]) -> float:
    """Luminance std-dev of an image object, 255 when it is too big or undecodable."""
    if not px[0] or px[0] * px[1] > CONTRAST_MAX_PIXELS:
        return 255.0
    from PIL import ImageStat

    try:
        bitmap = obj.get_bitmap(render=False)
        try:
            img = bitmap.to_pil().convert("L")
        finally:
            bitmap.close()
        img.thumbnail((128, 128))
        return float(ImageStat.Stat(img).stddev[0])
    except Exception:
        return 255.0


def _plan_vision(work: list[_PageWork], budget: int) -> bool:
    """
    Choose which pages get a DeepSeek call, in priority order: unreadable or
    text-less pages that show something (OCR), damaged pages (OCR), then
    readable pages with figures (descriptions). Returns True when the budget
    left some of them out.
    """
    tiers = [
        ("ocr", [w for w in work if w.quality.status == "junk"
                 or (w.quality.status == "empty" and w.visual)]),
        ("ocr", [w for w in work if w.quality.status == "degraded"]),
        ("describe", [w for w in work if w.quality.status == "ok" and w.figures]),
    ]
    truncated = False
    for kind, pages in tiers:
        for w in pages:
            if budget <= 0:
                truncated = True
                break
            w.kind = kind
            budget -= 1
    return truncated


def _run_vision(doc, work: list[_PageWork], file_name: str) -> None:
    """
    Run the planned pages' DeepSeek calls in parallel. Each worker renders its
    own page (one page at a time under the pdfium lock) right before its call,
    so at most VISION_CONCURRENCY page images exist at once.
    """
    planned = [w for w in work if w.kind]
    if not planned:
        return

    def run(w: _PageWork) -> vision.VisionResult:
        with _PDFIUM_LOCK:
            jpeg = _render_page_jpeg(doc, w.page - 1)
        call = vision.ocr_image if w.kind == "ocr" else vision.describe_image
        return call(jpeg, f"{file_name} p{w.page} {w.kind}")

    with ThreadPoolExecutor(max_workers=max(1, settings.vision_concurrency)) as pool:
        # Each call runs in a copy of this context, so its log lines keep the trace id.
        futures = {pool.submit(contextvars.copy_context().run, run, w): w for w in planned}
        for future in as_completed(futures):
            w = futures[future]
            try:
                w.result = future.result()
            except Exception as exc:
                log_step(logger, "EXTRACT_PAGES", f"vision failed on page {w.page} ({w.kind}): {exc}")


def _render_page_jpeg(doc, index: int) -> bytes:
    """Render one page to JPEG at RENDER_DPI (longest side <= RENDER_MAX_SIDE). Caller holds the lock."""
    page = doc[index]
    try:
        width, height = page.get_size()
        scale = min(RENDER_DPI / 72, RENDER_MAX_SIDE / max(width, height, 1.0))
        bitmap = page.render(scale=scale)
        try:
            out = io.BytesIO()
            bitmap.to_pil().convert("RGB").save(out, format="JPEG", quality=RENDER_JPEG_QUALITY)
            return out.getvalue()
        finally:
            bitmap.close()
    finally:
        page.close()


def _figure_lines(reply: str) -> str:
    """The [وصف صورة] lines of a describe reply ("" when it found no figures)."""
    if vision.is_no_figures(reply):
        return ""
    reply = clean_text(reply)
    lines = [line for line in reply.split("\n") if line.startswith(vision.FIGURE_PREFIX)]
    if lines:
        return "\n".join(lines)
    return f"{vision.FIGURE_PREFIX} {reply}" if reply else ""
