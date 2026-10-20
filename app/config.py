from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Required — set in Render dashboard
    rag_api_key: str      # shared auth secret for /retrieve endpoint
    deepseek_api_key: str # forwarded to litellm as DEEPSEEK_API_KEY; also used for vision
    gemini_api_key: str   # forwarded to litellm as GEMINI_API_KEY (embeddings only)

    # ChromaDB storage path (Render persistent disk mount point)
    chroma_path: str = "/data/chroma"

    # Multi-app (tenant) routing — see app/apps.py for the full format.
    # Empty means single-tenant: one app on the `notebook_docs` collection
    # authenticated with rag_api_key, i.e. exactly the pre-tenant behaviour.
    #   rag_apps      "quiz_main:notebook_docs,daftar_ai:daftar_notebook_docs"
    #   rag_app_keys  "daftar_ai:<key>"  (the first app falls back to rag_api_key)
    rag_apps: str = ""
    rag_app_keys: str = ""

    # Model identifiers (litellm model strings)
    litellm_model: str = "deepseek/deepseek-chat"      # used for query rewriting (smart chunking)
    embedding_model: str = "gemini/gemini-embedding-001" # Google AI embedding model via litellm
    # Vector size for NEW collections (gemini-embedding-001 supports 768/1536/3072).
    # Existing collections keep the size they were built with — see
    # app/db/chroma.py:dimensions_for. 768 is a quarter of the memory of 3072.
    embedding_dimensions: int = 768

    # Retrieval tuning
    retrieval_k: int = 20   # how many docs to pull per query vector
    final_k: int = 10       # how many to return after reranking

    # Chunking concurrency (ThreadPoolExecutor workers)
    chunk_workers: int = 3

    # Chunking mode: "default" = fast recursive splitter (no LLM, like OpenAI/Google)
    #                "smart"   = LLM-based semantic splitting
    # Only collections without a `chunker` flag use it; see app/db/chroma.py.
    chunk_mode: str = "default"

    # DeepSeek vision (image OCR, scanned / junk-font PDF pages, figure
    # descriptions). Called over plain HTTP — litellm's deepseek provider drops
    # image parts. Every value has a working default; the env vars
    # (DEEPSEEK_API_BASE, VISION_MODEL, ...) only exist to tune them.
    deepseek_api_base: str = "https://api.deepseek.com"
    vision_model: str = "deepseek-flash"
    vision_max_tokens: int = 3000
    vision_max_pages: int = 60      # DeepSeek page calls per ingested PDF (OCR + figures)
    vision_concurrency: int = 4     # concurrent DeepSeek vision calls, process-wide
    vision_timeout_s: float = 120

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
