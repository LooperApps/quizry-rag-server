"""
DeepSeek vision: OCR of images and PDF pages, and figure descriptions.

Called over plain HTTP, not litellm: litellm's deepseek provider flattens
message content to a string and silently drops the image parts, so the model
would "read" a page it never saw. Gemini never receives an image.

Every call sends `thinking: {type: disabled}` — with reasoning on, a page burns
1-2.5k hidden tokens and can come back empty when max_tokens is reached.
Calls are limited process-wide to VISION_CONCURRENCY, retried on 429/5xx/
timeouts, and their token usage is logged and totalled (see usage_totals).
"""

import base64
import io
import logging
import threading
import time
from dataclasses import dataclass

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.config import settings
from app.log_utils import log_step

logger = logging.getLogger(__name__)

# Canonical prompts — the same wording is used by the Cloud Functions and the
# library pipeline (design contract §1). Do not reword one copy only.
OCR_PROMPT = (
    "أنت نظام OCR دقيق لكتب مدرسية. انسخ كل النص الظاهر في صورة الصفحة حرفياً وبترتيب القراءة "
    "الطبيعي. حافظ على العناوين والقوائم والجداول بصيغة Markdown. لا تترجم ولا تلخص ولا تضف شرحاً. "
    "إذا احتوت الصفحة على رسوم أو صور أو مخططات تعليمية فأضف لكل منها سطراً يبدأ بـ [وصف صورة] "
    "يصف محتواها التعليمي باختصار. تجاهل الزخارف والشعارات."
)
DESCRIBE_PROMPT = (
    "صف باختصار كل رسم أو صورة أو مخطط أو جدول مصوَّر ذي محتوى تعليمي في صورة الصفحة هذه، "
    "كل واحد في سطر يبدأ بـ [وصف صورة]. اذكر ما يظهر فيه من بيانات وتسميات وأرقام. لا تنسخ نص "
    "الصفحة ولا تصف الزخارف أو الشعارات. إن لم توجد رسوم تعليمية فأجب بكلمة: لا يوجد"
)
FIGURE_PREFIX = "[وصف صورة]"
NO_FIGURES = "لا يوجد"

# What DeepSeek accepts as an image_url data URI; anything else is converted.
SUPPORTED_MIME = frozenset({"image/jpeg", "image/png", "image/gif", "image/webp"})
MAX_SIDE_PX = 8000
# Well under DeepSeek's 32 MiB per image / 48 MiB per body once base64-encoded.
MAX_IMAGE_BYTES = 20 * 1024 * 1024
JPEG_QUALITY = 85

# ISO-BMFF brands of HEIC/HEIF/AVIF files (bytes 8-12, after "ftyp").
_HEIF_BRANDS = {b"heic", b"heix", b"hevc", b"hevx", b"heim", b"heis", b"mif1", b"msf1", b"avif"}

# USD per 1M tokens as (input, output), off-peak and peak — for the usage log.
_PRICE_OFFPEAK = (0.15, 0.60)
_PRICE_PEAK = (0.30, 1.20)

_semaphore = threading.BoundedSemaphore(max(1, settings.vision_concurrency))
_client: httpx.Client | None = None
_client_lock = threading.Lock()
_usage_lock = threading.Lock()
_usage = {"calls": 0, "failures": 0, "promptTokens": 0, "completionTokens": 0}


class VisionError(RuntimeError):
    """A vision call failed for good (bad request, auth, retries exhausted)."""


class UnsupportedImageError(ValueError):
    """The bytes are not an image this server can send (ValueError -> HTTP 422)."""


class _Retryable(VisionError):
    """429 or 5xx: worth another attempt."""


@dataclass
class VisionResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    elapsed_ms: int = 0
    finish_reason: str | None = None


# ─── Image preparation ────────────────────────────────────────────────────────

def sniff_mime(data: bytes) -> str | None:
    """Identify an image from its magic bytes; None when unrecognised."""
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[:2] == b"BM":
        return "image/bmp"
    if data[:4] in (b"II*\x00", b"MM\x00*"):
        return "image/tiff"
    if data[4:8] == b"ftyp" and data[8:12] in _HEIF_BRANDS:
        return "image/heic"
    return None


_heif_registered = False


def _open_image(data: bytes):
    global _heif_registered
    from PIL import Image

    if not _heif_registered:
        _heif_registered = True
        try:
            # Optional plugin: HEIC/HEIF photos (iPhone) only decode with it installed.
            import pillow_heif  # type: ignore[import-not-found]

            pillow_heif.register_heif_opener()
        except ImportError:
            pass
    return Image.open(io.BytesIO(data))


def _to_jpeg(img, max_side: int) -> bytes:
    from PIL import Image, ImageOps

    try:
        img.seek(0)  # first frame of multi-frame TIFF/GIF
    except EOFError:
        pass
    img = ImageOps.exif_transpose(img) or img  # EXIF orientation is dropped on save
    if max(img.size) > max_side:
        img = img.copy()
        img.thumbnail((max_side, max_side))
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        flat = Image.new("RGB", rgba.size, (255, 255, 255))
        flat.paste(rgba, mask=rgba.getchannel("A"))
        img = flat
    elif img.mode != "RGB":
        img = img.convert("RGB")
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=JPEG_QUALITY)
    return out.getvalue()


def prepare_image(data: bytes) -> tuple[bytes, str]:
    """
    Return (bytes, mime) in a form DeepSeek accepts: JPEG/PNG/GIF/WebP passed
    through untouched when small enough; anything else (HEIC, BMP, TIFF, ...),
    oversized, or larger than MAX_SIDE_PX re-encoded as JPEG with Pillow.
    """
    mime = sniff_mime(data)
    if mime == "image/heic":
        try:
            img = _open_image(data)
        except Exception as exc:
            raise UnsupportedImageError(
                "HEIC/HEIF images must be converted to JPEG before upload"
            ) from exc
    else:
        try:
            img = _open_image(data)
        except Exception as exc:
            if mime in SUPPORTED_MIME and len(data) <= MAX_IMAGE_BYTES:
                return data, mime  # Pillow cannot parse it; let the API decide
            raise UnsupportedImageError(f"Not a readable image ({mime or 'unknown format'})") from exc

    if mime in SUPPORTED_MIME and len(data) <= MAX_IMAGE_BYTES and max(img.size) <= MAX_SIDE_PX:
        return data, mime

    max_side = MAX_SIDE_PX
    jpeg = _to_jpeg(img, max_side)
    while len(jpeg) > MAX_IMAGE_BYTES and max_side > 1024:
        max_side = int(max_side * 0.75)
        jpeg = _to_jpeg(img, max_side)
    log_step(
        logger, "VISION",
        f"converted {mime or 'unknown'} {img.size[0]}x{img.size[1]} {len(data):,}B "
        f"-> image/jpeg {len(jpeg):,}B",
    )
    return jpeg, "image/jpeg"


# ─── HTTP ─────────────────────────────────────────────────────────────────────

def _http() -> httpx.Client:
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = httpx.Client(
                    base_url=settings.deepseek_api_base.rstrip("/"),
                    timeout=httpx.Timeout(settings.vision_timeout_s, connect=10.0),
                )
    return _client


@retry(
    retry=retry_if_exception_type((_Retryable, httpx.TransportError)),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    stop=stop_after_attempt(4),
    reraise=True,
)
def _post(body: dict) -> dict:
    response = _http().post(
        "/chat/completions",
        json=body,
        headers={"Authorization": f"Bearer {settings.deepseek_api_key}"},
    )
    if response.status_code == 429 or response.status_code >= 500:
        raise _Retryable(f"DeepSeek HTTP {response.status_code}: {response.text[:300]}")
    if response.status_code != 200:
        raise VisionError(f"DeepSeek HTTP {response.status_code}: {response.text[:300]}")
    return response.json()


def _record(prompt_tokens: int = 0, completion_tokens: int = 0, failed: bool = False) -> None:
    with _usage_lock:
        _usage["calls"] += 1
        _usage["failures"] += int(failed)
        _usage["promptTokens"] += prompt_tokens
        _usage["completionTokens"] += completion_tokens


def _cost(prompt_tokens: int, completion_tokens: int, prices: tuple[float, float]) -> float:
    return (prompt_tokens * prices[0] + completion_tokens * prices[1]) / 1_000_000


def usage_totals() -> dict:
    """Process-lifetime vision usage, with the cost at off-peak and peak prices."""
    with _usage_lock:
        totals = dict(_usage)
    totals["usdOffPeak"] = round(_cost(totals["promptTokens"], totals["completionTokens"], _PRICE_OFFPEAK), 4)
    totals["usdPeak"] = round(_cost(totals["promptTokens"], totals["completionTokens"], _PRICE_PEAK), 4)
    totals["model"] = settings.vision_model
    return totals


# ─── Public API ───────────────────────────────────────────────────────────────

def analyze_image(
    data: bytes,
    prompt: str,
    *,
    max_tokens: int | None = None,
    label: str = "",
) -> VisionResult:
    """
    Send one image plus a text prompt to DeepSeek and return the reply.

    Raises UnsupportedImageError for bytes that are not a usable image and
    VisionError when the API refuses or keeps failing after retries.
    """
    payload, mime = prepare_image(data)
    body = {
        "model": settings.vision_model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{base64.b64encode(payload).decode()}"},
                },
            ],
        }],
        "max_tokens": max_tokens or settings.vision_max_tokens,
        "temperature": 0,
        "thinking": {"type": "disabled"},
    }
    with _semaphore:
        t0 = time.perf_counter()
        try:
            raw = _post(body)
        except Exception as exc:
            _record(failed=True)
            log_step(logger, "VISION", f"FAILED {label} model={settings.vision_model}: {exc}")
            if isinstance(exc, VisionError):
                raise
            raise VisionError(f"DeepSeek vision call failed: {exc}") from exc
        elapsed = int((time.perf_counter() - t0) * 1000)

    try:
        choice = raw["choices"][0]
        text = (choice.get("message") or {}).get("content") or ""
        finish = choice.get("finish_reason")
    except (KeyError, IndexError, TypeError) as exc:
        _record(failed=True)
        raise VisionError(f"Unexpected DeepSeek response shape: {str(raw)[:300]}") from exc
    usage = raw.get("usage") or {}
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    _record(prompt_tokens, completion_tokens)

    log_step(
        logger, "VISION",
        f"{label} model={raw.get('model') or settings.vision_model} mime={mime} "
        f"bytes={len(payload):,} elapsed={elapsed}ms finish={finish} "
        f"prompt_tokens={prompt_tokens} completion_tokens={completion_tokens} "
        f"usd_offpeak={_cost(prompt_tokens, completion_tokens, _PRICE_OFFPEAK):.5f} "
        f"usd_peak={_cost(prompt_tokens, completion_tokens, _PRICE_PEAK):.5f} chars={len(text)}",
    )
    if finish == "length":
        logger.warning(f"[VISION] {label} reply cut at max_tokens={body['max_tokens']}")
    return VisionResult(
        text=text.strip(),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        elapsed_ms=elapsed,
        finish_reason=finish,
    )


def ocr_image(data: bytes, label: str = "") -> VisionResult:
    """Transcribe an image/page (with [وصف صورة] lines for its figures)."""
    return analyze_image(data, OCR_PROMPT, label=label or "ocr")


def describe_image(data: bytes, label: str = "") -> VisionResult:
    """Describe only the figures of a page whose text is already known."""
    return analyze_image(data, DESCRIBE_PROMPT, label=label or "describe")


def is_no_figures(text: str) -> bool:
    """True for the describe prompt's "nothing here" answer (or an empty one)."""
    core = text.strip().strip(".،!:").strip()
    return not core or core == NO_FIGURES
