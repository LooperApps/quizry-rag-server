"""
Tests for the ingestion pipeline.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app

client = TestClient(app)
AUTH = {"Authorization": f"Bearer {settings.rag_api_key}"}


# ─── /ingest POST ─────────────────────────────────────────────────────────────

class TestIngestEndpoint:
    def test_missing_auth_returns_401(self):
        resp = client.post("/ingest", json={
            "notebookId": "nb1", "sourceId": "s1",
            "storageUrl": "https://example.com/file.pdf",
            "fileName": "notes.pdf",
        })
        assert resp.status_code == 401

    def test_invalid_token_returns_401(self):
        resp = client.post(
            "/ingest",
            headers={"Authorization": "Bearer wrong-key"},
            json={
                "notebookId": "nb1", "sourceId": "s1",
                "storageUrl": "https://example.com/file.pdf",
                "fileName": "notes.pdf",
            },
        )
        assert resp.status_code == 401

    def test_empty_notebookId_returns_422(self):
        resp = client.post(
            "/ingest",
            headers=AUTH,
            json={
                "notebookId": "", "sourceId": "s1",
                "storageUrl": "https://example.com/file.pdf",
                "fileName": "notes.pdf",
            },
        )
        assert resp.status_code == 422

    @patch("app.routers.ingest.run_ingest", new_callable=AsyncMock)
    def test_successful_ingest(self, mock_run):
        mock_run.return_value = 42
        resp = client.post(
            "/ingest",
            headers=AUTH,
            json={
                "notebookId": "nb1", "sourceId": "src1",
                "storageUrl": "https://firebasestorage.googleapis.com/v0/b/test/o/file.pdf?alt=media",
                "fileName": "lecture.pdf",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["chunkCount"] == 42
        assert body["sourceId"] == "src1"
        mock_run.assert_awaited_once()

    @patch("app.routers.ingest.run_ingest", new_callable=AsyncMock)
    def test_value_error_returns_422(self, mock_run):
        mock_run.side_effect = ValueError("No extractable text found")
        resp = client.post(
            "/ingest",
            headers=AUTH,
            json={
                "notebookId": "nb1", "sourceId": "src1",
                "storageUrl": "https://example.com/image.png",
                "fileName": "image.png",
            },
        )
        assert resp.status_code == 422
        assert "No extractable text" in resp.json()["detail"]

    @patch("app.routers.ingest.run_ingest", new_callable=AsyncMock)
    def test_unexpected_error_returns_500(self, mock_run):
        mock_run.side_effect = RuntimeError("Storage unreachable")
        resp = client.post(
            "/ingest",
            headers=AUTH,
            json={
                "notebookId": "nb1", "sourceId": "src1",
                "storageUrl": "https://example.com/file.pdf",
                "fileName": "file.pdf",
            },
        )
        assert resp.status_code == 500


# ─── DELETE endpoints ─────────────────────────────────────────────────────────

class TestDeleteEndpoints:
    @patch("app.routers.ingest.delete_notebook_chunks")
    def test_delete_notebook(self, mock_delete):
        resp = client.delete("/ingest/notebooks/nb123", headers=AUTH)
        assert resp.status_code == 200
        assert resp.json()["notebookId"] == "nb123"
        mock_delete.assert_called_once_with("nb123")

    @patch("app.routers.ingest._delete_source_chunks")
    def test_delete_source(self, mock_delete):
        resp = client.delete("/ingest/notebooks/nb123/sources/src456", headers=AUTH)
        assert resp.status_code == 200
        body = resp.json()
        assert body["notebookId"] == "nb123"
        assert body["sourceId"] == "src456"
        mock_delete.assert_called_once_with("nb123", "src456")


# ─── Extractor unit tests ─────────────────────────────────────────────────────

class TestExtractor:
    def test_plain_text_extraction(self):
        from app.services.extractor import extract_text
        data = "Hello world\n\nThis is a test.".encode("utf-8")
        result = extract_text(data, "notes.txt")
        assert "Hello world" in result
        assert "test" in result

    def test_markdown_extraction(self):
        from app.services.extractor import extract_text
        data = "# Title\n\nSome **content** here.".encode("utf-8")
        result = extract_text(data, "notes.md")
        assert "Title" in result

    def test_html_extraction(self):
        from app.services.extractor import extract_text
        data = b"<html><body><p>Hello</p><script>alert(1)</script></body></html>"
        result = extract_text(data, "page.html")
        assert "Hello" in result
        assert "alert" not in result  # script tag removed

    def test_unsupported_format_returns_empty(self):
        from app.services.extractor import extract_text
        result = extract_text(b"\x89PNG\r\n", "image.png")
        assert result == ""
