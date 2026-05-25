import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.db.chroma import get_collection
from app.routers import health, ingest, retrieve

# Expose API keys so litellm can pick them up automatically from the environment.
# litellm reads DEEPSEEK_API_KEY and GEMINI_API_KEY by convention.
os.environ.setdefault("DEEPSEEK_API_KEY", settings.deepseek_api_key)
os.environ.setdefault("GEMINI_API_KEY", settings.gemini_api_key)

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Quizry RAG server...")
    # Warm up ChromaDB (creates the collection if it doesn't exist)
    col = get_collection()
    logger.info(f"ChromaDB initialized — collection has {col.count()} documents")
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
    allow_methods=["POST", "GET", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)

app.include_router(health.router)
app.include_router(ingest.router)
app.include_router(retrieve.router)
