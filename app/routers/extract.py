"""
Extraction router.

POST /extract — page-wise text of one file (PDF text layer graded and repaired
with DeepSeek OCR, figure descriptions; images OCR'd; office/html/text files as
one page). Nothing is indexed. Used by Cloud Functions, e.g. for flashcards
from an uploaded file.
"""

import base64
import binascii
import logging
import time

from fastapi import APIRouter, Depends, HTTPException, status
from starlette.concurrency import run_in_threadpool

from app.apps import App
from app.auth import resolve_app
from app.log_utils import get_trace_id, log_step, redact_url
from app.models.extract import ExtractedPageOut, ExtractRequest, ExtractResponse, MAX_BASE64_BYTES
from app.services import vision
from app.services.extractor import download_bytes, extract_pages

router = APIRouter(prefix="/extract", tags=["extract"])
logger = logging.getLogger(__name__)

# Files fetched from storageUrl: beyond this the per-page work no longer fits
# one synchronous request on the instance.
MAX_DOWNLOAD_BYTES = 100 * 1024 * 1024


@router.post(
    "",
    response_model=ExtractResponse,
    summary="Extract page text from a file (no indexing)",
)
async def extract(
    req: ExtractRequest,
    rag_app: App = Depends(resolve_app),
) -> ExtractResponse:
    trace = get_trace_id()
    t_start = time.perf_counter()

    if req.fileBase64:
        try:
            data = base64.b64decode(req.fileBase64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise HTTPException(status_code=422, detail=f"fileBase64 is not valid base64: {exc}") from exc
        if len(data) > MAX_BASE64_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"fileBase64 exceeds {MAX_BASE64_BYTES // (1024 * 1024)} MB decoded",
            )
        origin = "fileBase64"
    else:
        try:
            data = await download_bytes(req.storageUrl or "")
        except Exception as exc:
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
        if len(data) > MAX_DOWNLOAD_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File is {len(data):,} bytes; /extract accepts up to {MAX_DOWNLOAD_BYTES:,}",
            )
        origin = redact_url(req.storageUrl or "")

    log_step(
        logger, "EXTRACT",
        f"trace={trace} app={rag_app.id} file={req.fileName!r} from={origin} bytes={len(data):,} "
        f"maxVisionPages={req.maxVisionPages}",
    )
    try:
        result = await run_in_threadpool(
            extract_pages, data, req.fileName, vision_enabled=True, max_vision_pages=req.maxVisionPages,
        )
    except vision.VisionError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

    log_step(
        logger, "EXTRACT",
        f"trace={trace} file={req.fileName!r} pages={result.page_count} method={result.method} "
        f"visionPages={result.vision_pages} truncated={result.truncated} "
        f"elapsed={int((time.perf_counter() - t_start) * 1000)}ms",
    )
    return ExtractResponse(
        pageCount=result.page_count,
        pages=[ExtractedPageOut(page=p.page, text=p.text, method=p.method) for p in result.pages],
        visionPages=result.vision_pages,
        truncated=result.truncated,
    )
