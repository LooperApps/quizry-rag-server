"""
Test environment: dummy keys, a throwaway Chroma directory, no network.

Set before any `app.*` import — app/config.py reads the environment at import
time and refuses to start without the required keys.
"""

import os
import tempfile

_CHROMA_DIR = tempfile.mkdtemp(prefix="rag-tests-chroma-")

os.environ.setdefault("RAG_API_KEY", "test-quiz-key")
os.environ.setdefault("DEEPSEEK_API_KEY", "test-deepseek-key")
os.environ.setdefault("GEMINI_API_KEY", "test-gemini-key")
os.environ.setdefault("RAG_APPS", "quiz_main:notebook_docs_768,daftar_ai:daftar_notebook_docs_768")
os.environ.setdefault("RAG_APP_KEYS", "daftar_ai:test-daftar-key")
os.environ.setdefault("ADMIN_API_KEY", "test-admin-key")
os.environ.setdefault("EMBEDDING_DIMENSIONS", "768")
os.environ["CHROMA_PATH"] = _CHROMA_DIR
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
os.environ.setdefault("LOG_LEVEL", "WARNING")
