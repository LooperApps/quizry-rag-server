from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Required
    gemini_api_key: str
    rag_api_key: str  # Shared secret — set in Render dashboard + Firebase Functions config

    # ChromaDB storage path (Render persistent disk mount point)
    chroma_path: str = "/data/chroma"

    # Model identifiers
    gemini_model: str = "gemini-2.5-flash"              # used for reranking (google.generativeai SDK)
    litellm_model: str = "gemini/gemini-2.5-flash"      # used for chunking (litellm)
    embedding_model: str = "models/text-embedding-004"  # Google AI embedding model

    # Retrieval tuning
    retrieval_k: int = 20   # how many docs to pull per query vector
    final_k: int = 10       # how many to return after reranking

    # Chunking concurrency (ThreadPoolExecutor workers)
    chunk_workers: int = 3

    # Logging
    log_level: str = "INFO"

    # FastAPI docs (disable in production, enable for debug)
    enable_docs: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


settings = Settings()
