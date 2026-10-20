import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from app.apps import all_apps, log_registry
from app.config import settings
from app.db.chroma import get_collection
from app.log_utils import get_trace_id, redact_body, set_trace_id
from app.routers import admin, extract, health, ingest, retrieve

# Expose API keys so litellm can pick them up automatically from the environment.
# litellm reads DEEPSEEK_API_KEY and GEMINI_API_KEY by convention.
os.environ.setdefault("DEEPSEEK_API_KEY", settings.deepseek_api_key)
os.environ.setdefault("GEMINI_API_KEY", settings.gemini_api_key)

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

LINE = "=" * 80


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Quizry RAG server...")
    log_registry()
    # Warm up ChromaDB, creating each app's collection if it does not exist yet.
    # Doing it here rather than on first request means a brand new app is
    # queryable (and visibly empty) from boot instead of 404-ing into existence.
    for rag_app in all_apps():
        col = get_collection(rag_app.collection)
        logger.info(
            f"ChromaDB ready — app={rag_app.id} collection={rag_app.collection} "
            f"documents={col.count()}"
        )
    yield
    logger.info("Shutting down Quizry RAG server")


app = FastAPI(
    title="Quizry RAG Server",
    version="1.0.0",
    description="Retrieval-augmented generation service for Quizry notebook AI features",
    lifespan=lifespan,
    docs_url="/docs" if settings.enable_docs else None,
    redoc_url="/redoc" if settings.enable_docs else None,
    openapi_url="/openapi.json" if settings.enable_docs else None,
)

# Cloud Functions call from a server, but allow all origins during development.
# Tighten `allow_origins` to the specific Cloud Functions service URL in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)


# ─── Request/Response body logging middleware ─────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    trace_id = set_trace_id()
    t_start = time.perf_counter()

    method = request.method
    path = request.url.path

    # Read and log the request body — redacted (download tokens, base64
    # payloads, whole-book page text) and truncated.
    body_bytes = await request.body()
    body_log = redact_body(path, body_bytes.decode("utf-8", errors="replace"))
    qs = request.url.query
    full_path = f"{path}?{qs}" if qs else path

    logger.info(
        f"\n{LINE}\n"
        f"[REQUEST] trace={trace_id} {method} {full_path}\n"
        f"--- BODY ---\n{body_log}\n"
        f"{LINE}"
    )

    # Capture the response body
    response: Response = await call_next(request)

    # Try to read response body
    resp_body = ""
    try:
        if hasattr(response, "body"):
            resp_body = response.body.decode("utf-8", errors="replace")
        elif callable(getattr(response, "streaming", None)):
            pass  # streaming responses are handled separately
    except Exception:
        resp_body = "<unreadable>"

    resp_log = redact_body("", resp_body)

    elapsed_ms = int((time.perf_counter() - t_start) * 1000)
    logger.info(
        f"\n{LINE}\n"
        f"[RESPONSE] trace={trace_id} {method} {full_path} "
        f"status={response.status_code} elapsed={elapsed_ms}ms\n"
        f"--- BODY ---\n{resp_log}\n"
        f"{LINE}"
    )

    return response


app.include_router(health.router)
app.include_router(ingest.router)
app.include_router(retrieve.router)
app.include_router(extract.router)
if settings.enable_admin:
    app.include_router(admin.router)
