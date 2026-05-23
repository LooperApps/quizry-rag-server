"""
Tests for the retrieval endpoint and pipeline.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.services.retriever import ChunkResult

client = TestClient(app)
AUTH = {"Authorization": f"Bearer {settings.rag_api_key}"}

MOCK_CHUNKS = [
    ChunkResult(content="Photosynthesis converts light to energy.", source="bio.pdf", sourceId="s1"),
    ChunkResult(content="Chlorophyll absorbs red and blue light.", source="bio.pdf", sourceId="s1"),
]


# ─── /retrieve POST ───────────────────────────────────────────────────────────

class TestRetrieveEndpoint:
    def test_missing_auth_returns_401(self):
        resp = client.post("/retrieve", json={"notebookId": "nb1", "question": "test"})
        assert resp.status_code == 401

    def test_missing_question_returns_422(self):
        resp = client.post("/retrieve", headers=AUTH, json={"notebookId": "nb1"})
        assert resp.status_code == 422

    def test_missing_notebookid_returns_422(self):
        resp = client.post("/retrieve", headers=AUTH, json={"question": "What is DNA?"})
        assert resp.status_code == 422

    @patch("app.routers.retrieve.fetch_context")
    def test_successful_retrieval_single_notebook(self, mock_fetch):
        mock_fetch.return_value = (MOCK_CHUNKS, "photosynthesis process plants")

        resp = client.post(
            "/retrieve",
            headers=AUTH,
            json={"notebookId": "nb1", "question": "How do plants make food?", "k": 5},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["chunks"]) == 2
        assert body["chunks"][0]["content"] == MOCK_CHUNKS[0].content
        assert body["chunks"][0]["source"] == "bio.pdf"
        assert body["rewrittenQuery"] == "photosynthesis process plants"
        # Verify notebook_ids is resolved correctly
        call_kwargs = mock_fetch.call_args
        assert call_kwargs.kwargs["notebook_ids"] == ["nb1"]
        assert call_kwargs.kwargs["k"] == 5

    @patch("app.routers.retrieve.fetch_context")
    def test_successful_retrieval_multi_notebook(self, mock_fetch):
        mock_fetch.return_value = (MOCK_CHUNKS, "")

        resp = client.post(
            "/retrieve",
            headers=AUTH,
            json={
                "notebookIds": ["nb1", "nb2"],
                "question": "Compare photosynthesis and respiration",
            },
        )
        assert resp.status_code == 200
        call_kwargs = mock_fetch.call_args
        assert set(call_kwargs.kwargs["notebook_ids"]) == {"nb1", "nb2"}

    @patch("app.routers.retrieve.fetch_context")
    def test_empty_collection_returns_empty_chunks(self, mock_fetch):
        mock_fetch.return_value = ([], "")
        resp = client.post(
            "/retrieve",
            headers=AUTH,
            json={"notebookId": "nb_empty", "question": "What is energy?"},
        )
        assert resp.status_code == 200
        assert resp.json()["chunks"] == []

    @patch("app.routers.retrieve.fetch_context")
    def test_service_error_returns_500(self, mock_fetch):
        mock_fetch.side_effect = RuntimeError("ChromaDB unavailable")
        resp = client.post(
            "/retrieve",
            headers=AUTH,
            json={"notebookId": "nb1", "question": "test"},
        )
        assert resp.status_code == 500

    def test_k_clamped_to_max_20(self):
        resp = client.post(
            "/retrieve",
            headers=AUTH,
            json={"notebookId": "nb1", "question": "test", "k": 999},
        )
        assert resp.status_code == 422

    def test_notebookid_and_notebookids_deduplication(self):
        """notebookIds with duplicates should be deduplicated."""
        with patch("app.routers.retrieve.fetch_context") as mock_fetch:
            mock_fetch.return_value = ([], "")
            resp = client.post(
                "/retrieve",
                headers=AUTH,
                json={"notebookIds": ["nb1", "nb1", "nb2"], "question": "test"},
            )
            assert resp.status_code == 200
            call_kwargs = mock_fetch.call_args
            assert len(call_kwargs.kwargs["notebook_ids"]) == 2


# ─── /health GET ──────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_no_auth_required(self):
        with patch("app.routers.health.get_collection") as mock_col:
            mock_instance = MagicMock()
            mock_instance.count.return_value = 100
            mock_col.return_value = mock_instance
            resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["totalDocs"] == 100

    def test_health_chromadb_error_returns_degraded(self):
        with patch("app.routers.health.get_collection") as mock_col:
            mock_col.side_effect = RuntimeError("disk full")
            resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "degraded"


# ─── Retriever unit tests ─────────────────────────────────────────────────────

class TestRetriever:
    @patch("app.services.retriever.count_for_notebooks", return_value=0)
    def test_empty_notebook_returns_empty(self, _):
        from app.services.retriever import fetch_context
        chunks, query = fetch_context(["nb_empty"], "test question")
        assert chunks == []
        assert query == "test question"

    @patch("app.services.retriever.count_for_notebooks", return_value=5)
    @patch("app.services.retriever.rewrite_query", return_value="rewritten query")
    @patch("app.services.retriever.embed_query", return_value=[0.1] * 1536)
    @patch("app.services.retriever._query_chroma", return_value=MOCK_CHUNKS)
    @patch("app.services.retriever._rerank", return_value=MOCK_CHUNKS)
    def test_full_pipeline(self, mock_rerank, mock_query, mock_embed, mock_rewrite, mock_count):
        from app.services.retriever import fetch_context
        chunks, rewritten = fetch_context(["nb1"], "How do plants make food?", k=2)
        assert len(chunks) == 2
        assert rewritten == "rewritten query"
        mock_rewrite.assert_called_once()
        assert mock_query.call_count == 2  # once for original, once for rewritten
        mock_rerank.assert_called_once()

    @patch("app.services.retriever.count_for_notebooks", return_value=5)
    @patch("app.services.retriever.rewrite_query", side_effect=RuntimeError("LLM error"))
    @patch("app.services.retriever.embed_query", return_value=[0.1] * 1536)
    @patch("app.services.retriever._query_chroma", return_value=MOCK_CHUNKS)
    @patch("app.services.retriever._rerank", return_value=MOCK_CHUNKS)
    def test_rewrite_failure_falls_back(self, mock_rerank, mock_query, mock_embed, mock_rewrite, mock_count):
        from app.services.retriever import fetch_context
        chunks, rewritten = fetch_context(["nb1"], "original question")
        # Rewrite failed — only one query (original), rewritten == original
        assert rewritten == "original question"
        assert mock_query.call_count == 1
