from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Required — set in Render dashboard
    rag_api_key: str      # shared auth secret for /retrieve endpoint
    deepseek_api_key: str # forwarded to litellm as DEEPSEEK_API_KEY
    gemini_api_key: str   # forwarded to litellm as GEMINI_API_KEY (embeddings + image OCR)

    # ChromaDB storage path (Render persistent disk mount point)
    chroma_path: str = "/data/chroma"

    # Model identifiers (litellm model strings)
    litellm_model: str = "deepseek/deepseek-chat"      # used for query rewriting (smart chunking)
    embedding_model: str = "gemini/gemini-embedding-001" # Google AI embedding model via litellm

    # Retrieval tuning
    retrieval_k: int = 20   # how many docs to pull per query vector
    final_k: int = 10       # how many to return after reranking

    # Chunking concurrency (ThreadPoolExecutor workers)
    chunk_workers: int = 3

    # Chunking mode: "default" = fast recursive splitter (no LLM, like OpenAI/Google)
    #                "smart"   = LLM-based semantic splitting
    chunk_mode: str = "default"

    # Admin API (consumed by the quizry-db-manager Node server).
    # admin_api_key falls back to rag_api_key when empty — see app/auth.py.
    admin_api_key: str = ""
    enable_admin: bool = True

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
