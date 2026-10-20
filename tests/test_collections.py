"""
Integration tests against a real (temporary) Chroma directory: collection
flags, the where-builder, POST /ingest/text, multi-query /retrieve, staging
collection access control and the admin collection routes (list, promote,
delete, vacuum). Embeddings are faked: no network.
"""

import math
import re
import uuid
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.apps import CollectionAccessError, get_app, resolve_collection
from app.db.chroma import (
    _build_where,
    collection_flags,
    collection_space,
    get_client,
    get_collection,
)
from app.main import app

client = TestClient(app)
QUIZ = {"Authorization": "Bearer test-quiz-key"}
DAFTAR = {"Authorization": "Bearer test-daftar-key"}
ADMIN = {"Authorization": "Bearer test-admin-key"}

QUIZ_LIVE = "notebook_docs_768"
DAFTAR_LIVE = "daftar_notebook_docs_768"

# A tiny vocabulary so the fake embedding is meaningful: dimension i counts
# occurrences of word i, scaled by 3 so vectors are clearly NOT unit length.
VOCAB = ["photosynthesis", "chlorophyll", "mitochondria", "fractions", "algebra", "history", "volcano"]


def fake_embed_texts(texts, dimensions=None, task_type=None):
    fake_embed_texts.calls.append({"n": len(texts), "dimensions": dimensions, "task_type": task_type})
    dims = dimensions or 768
    out = []
    for text in texts:
        vec = np.full(dims, 0.01)
        lowered = text.lower()
        for i, word in enumerate(VOCAB):
            vec[i] += 3.0 * len(re.findall(word, lowered))
        out.append(vec.tolist())
    return out


fake_embed_texts.calls = []


@pytest.fixture(autouse=True)
def _fake_embeddings():
    fake_embed_texts.calls.clear()
    with patch("app.services.embedder.embed_texts", side_effect=fake_embed_texts), \
         patch("app.services.embedder._embed_batch",
               side_effect=lambda texts, dimensions=None, task_type=None: fake_embed_texts(texts, dimensions, task_type)), \
         patch("app.services.retriever.rewrite_query", side_effect=lambda q, h=None: q):
        yield


def _uid(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def _page_text(word: str, n: int = 120) -> str:
    return " ".join(f"{word} sentence number {i}." for i in range(n))


def _ingest_text(headers, notebook_id, source_id, pages, collection=None):
    body = {"notebookId": notebook_id, "sourceId": source_id, "fileName": f"{source_id}.pdf",
            "pages": [{"page": p, "text": t} for p, t in pages]}
    if collection:
        body["collection"] = collection
    return client.post("/ingest/text", headers=headers, json=body)


# ─── where-builder ────────────────────────────────────────────────────────────

class TestBuildWhere:
    def test_single_notebook_is_eq(self):
        assert _build_where(["nb1"]) == {"notebookId": {"$eq": "nb1"}}

    def test_several_notebooks_use_in(self):
        assert _build_where(["a", "b"]) == {"notebookId": {"$in": ["a", "b"]}}

    def test_sources_only_is_single_branch(self):
        assert _build_where([], ["s1"]) == {"sourceId": {"$eq": "s1"}}
        assert _build_where([], ["s1", "s2"]) == {"sourceId": {"$in": ["s1", "s2"]}}

    def test_both_lists_or(self):
        assert _build_where(["a"], ["s1", "s2"]) == {
            "$or": [{"notebookId": {"$eq": "a"}}, {"sourceId": {"$in": ["s1", "s2"]}}]
        }

    def test_nothing_raises(self):
        with pytest.raises(ValueError):
            _build_where([], [])

    def test_or_filter_is_accepted_by_chroma(self):
        col = get_collection(_uid("where_probe").replace("-", "_"))
        col.add(ids=["x1", "x2", "x3"], embeddings=[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
                metadatas=[{"notebookId": "a", "sourceId": "s1"}, {"notebookId": "b", "sourceId": "s2"},
                           {"notebookId": "c", "sourceId": "s3"}])
        got = col.get(where=_build_where(["a"], ["s2"]), include=[])["ids"]
        assert sorted(got) == ["x1", "x2"]


# ─── collection flags ─────────────────────────────────────────────────────────

class TestCollectionFlags:
    def test_new_collection_is_cosine_with_flags(self):
        name = _uid("new_flags").replace("-", "_")
        col = get_collection(name)
        assert collection_space(col) == "cosine"
        flags = collection_flags(name)
        assert flags.normalized and flags.task_types and flags.pdf_vision and flags.pages
        assert flags.dimensions == 768

    def test_legacy_collection_keeps_flags_off(self):
        name = _uid("legacy").replace("-", "_")
        get_client().create_collection(name=name, metadata={"embedding_dimensions": 768})
        flags = collection_flags(name)
        assert not (flags.normalized or flags.task_types or flags.pdf_vision or flags.pages)
        assert flags.dimensions == 768


# ─── staging resolution ───────────────────────────────────────────────────────

class TestResolveCollection:
    def test_live_and_staging(self):
        daftar = get_app("daftar_ai")
        assert resolve_collection(daftar, None) == DAFTAR_LIVE
        assert resolve_collection(daftar, DAFTAR_LIVE) == DAFTAR_LIVE
        assert resolve_collection(daftar, DAFTAR_LIVE + "__next") == DAFTAR_LIVE + "__next"

    @pytest.mark.parametrize("name", [QUIZ_LIVE, QUIZ_LIVE + "__next", "other", "daftar_notebook_docs_768x",
                                      "daftar_notebook_docs_768__../x", "a" * 80])
    def test_rejected(self, name):
        with pytest.raises(CollectionAccessError):
            resolve_collection(get_app("daftar_ai"), name)


# ─── POST /ingest/text ────────────────────────────────────────────────────────

class TestIngestText:
    def test_store_upsert_and_stale_delete(self):
        staging = DAFTAR_LIVE + "__t_ingest"
        nb, src = _uid("nb"), _uid("src")
        pages = [(p, _page_text("photosynthesis" if p % 2 else "algebra")) for p in range(1, 7)]
        resp = _ingest_text(DAFTAR, nb, src, pages, staging)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == "ok" and body["collection"] == staging and body["pageCount"] == 6
        n = body["chunkCount"]
        assert n >= 3

        col = get_collection(staging)
        got = col.get(where={"sourceId": src}, include=["metadatas", "embeddings"])
        assert sorted(got["ids"]) == [f"{src}:{i:05d}" for i in range(n)]
        for meta in got["metadatas"]:
            assert meta["notebookId"] == nb and meta["source"] == f"{src}.pdf"
            assert 1 <= meta["pageStart"] <= meta["pageEnd"] <= 6
        # flagged collection: unit vectors, RETRIEVAL_DOCUMENT task type
        for vec in got["embeddings"]:
            assert math.isclose(float(np.linalg.norm(vec)), 1.0, rel_tol=1e-4)
        assert any(c["task_type"] == "RETRIEVAL_DOCUMENT" for c in fake_embed_texts.calls)

        # Re-ingest the same source with fewer pages: same ids reused, leftovers gone.
        resp = _ingest_text(DAFTAR, nb, src, pages[:2], staging)
        assert resp.status_code == 200
        m = resp.json()["chunkCount"]
        assert m < n
        ids = sorted(col.get(where={"sourceId": src}, include=[])["ids"])
        assert ids == [f"{src}:{i:05d}" for i in range(m)]

    def test_source_of_another_notebook_conflicts(self):
        staging = DAFTAR_LIVE + "__t_conflict"
        src = _uid("src")
        assert _ingest_text(DAFTAR, "nbA", src, [(1, _page_text("history"))], staging).status_code == 200
        resp = _ingest_text(DAFTAR, "nbB", src, [(1, _page_text("history"))], staging)
        assert resp.status_code == 409

    def test_empty_text_422(self):
        resp = _ingest_text(DAFTAR, "nb", "src", [(1, "   "), (2, "\n")])
        assert resp.status_code == 422
        resp = _ingest_text(DAFTAR, "nb", "src", [(1, "\x00\x01")])
        assert resp.status_code == 422

    def test_other_apps_collection_forbidden(self):
        for name in (QUIZ_LIVE, QUIZ_LIVE + "__next"):
            resp = _ingest_text(DAFTAR, "nb", "src", [(1, "text " * 50)], name)
            assert resp.status_code == 403, name

    def test_bad_page_number_422(self):
        resp = client.post("/ingest/text", headers=DAFTAR, json={
            "notebookId": "nb", "sourceId": "s", "fileName": "f.pdf", "pages": [{"page": 0, "text": "x"}],
        })
        assert resp.status_code == 422

    def test_requires_app_key(self):
        resp = client.post("/ingest/text", json={"notebookId": "nb", "sourceId": "s", "fileName": "f",
                                                  "pages": [{"page": 1, "text": "x"}]})
        assert resp.status_code == 401


# ─── /retrieve: multi-query, sourceIds, staging ───────────────────────────────

@pytest.fixture(scope="module")
def corpus():
    """Two notebooks in a staging collection: the user's own and a library one."""
    staging = DAFTAR_LIVE + "__t_retrieve"
    nb_own, nb_lib = _uid("own"), _uid("lib")
    src_own, src_lib = _uid("s_own"), _uid("s_lib")
    with patch("app.services.embedder.embed_texts", side_effect=fake_embed_texts):
        r1 = _ingest_text(DAFTAR, nb_own, src_own,
                          [(1, _page_text("photosynthesis")), (2, _page_text("chlorophyll"))], staging)
        r2 = _ingest_text(DAFTAR, nb_lib, src_lib,
                          [(1, _page_text("fractions")), (2, _page_text("volcano"))], staging)
    assert r1.status_code == 200 and r2.status_code == 200
    return {"staging": staging, "nb_own": nb_own, "nb_lib": nb_lib, "src_own": src_own, "src_lib": src_lib}


class TestRetrieveMulti:
    def test_multi_query_round_robin_and_fields(self, corpus):
        resp = client.post("/retrieve", headers=DAFTAR, json={
            "notebookId": corpus["nb_own"], "sourceIds": [corpus["src_lib"]],
            "queries": ["photosynthesis", "volcano"], "k": 4, "collection": corpus["staging"],
        })
        assert resp.status_code == 200, resp.text
        body = resp.json()
        chunks = body["chunks"]
        assert body["rewrittenQuery"] is None
        assert len(chunks) == 4
        # round-robin: best of query 1, best of query 2, ...
        assert "photosynthesis" in chunks[0]["content"]
        assert "volcano" in chunks[1]["content"]
        assert {c["sourceId"] for c in chunks} == {corpus["src_own"], corpus["src_lib"]}
        for c in chunks:
            assert c["notebookId"] in (corpus["nb_own"], corpus["nb_lib"])
            assert isinstance(c["pageStart"], int) and isinstance(c["pageEnd"], int)
            assert isinstance(c["distance"], float)
        # one embedding batch for all queries, as RETRIEVAL_QUERY
        query_calls = [c for c in fake_embed_texts.calls if c["task_type"] == "RETRIEVAL_QUERY"]
        assert len(query_calls) == 1 and query_calls[0]["n"] == 2

    def test_dedupe_keeps_best_distance(self, corpus):
        resp = client.post("/retrieve", headers=DAFTAR, json={
            "sourceIds": [corpus["src_own"]], "queries": ["photosynthesis", "photosynthesis chlorophyll"],
            "k": 40, "collection": corpus["staging"],
        })
        assert resp.status_code == 200
        ids = [(c["content"][:40], c["pageStart"]) for c in resp.json()["chunks"]]
        assert len(ids) == len(set(ids))

    def test_source_ids_only_excludes_other_notebooks(self, corpus):
        resp = client.post("/retrieve", headers=DAFTAR, json={
            "sourceIds": [corpus["src_lib"]], "question": "fractions", "rewrite": False,
            "collection": corpus["staging"],
        })
        assert resp.status_code == 200
        chunks = resp.json()["chunks"]
        assert chunks and all(c["sourceId"] == corpus["src_lib"] for c in chunks)

    def test_single_question_mode_has_new_fields(self, corpus):
        resp = client.post("/retrieve", headers=DAFTAR, json={
            "notebookIds": [corpus["nb_own"], corpus["nb_lib"]], "question": "volcano", "k": 2,
            "collection": corpus["staging"],
        })
        assert resp.status_code == 200
        first = resp.json()["chunks"][0]
        assert "volcano" in first["content"] and first["distance"] is not None

    def test_staging_of_other_app_forbidden_and_missing_404(self, corpus):
        resp = client.post("/retrieve", headers=QUIZ, json={
            "notebookId": "x", "question": "q", "collection": corpus["staging"]})
        assert resp.status_code == 403
        resp = client.post("/retrieve", headers=DAFTAR, json={
            "notebookId": "x", "question": "q", "collection": DAFTAR_LIVE + "__does_not_exist"})
        assert resp.status_code == 404


class TestRetrieveValidation:
    @pytest.mark.parametrize("body", [
        {"notebookId": "nb", "queries": ["a", "b"]},
        {"sourceIds": ["s"], "question": "q"},
        {"notebookId": "nb", "question": "q", "k": 40},
        {"notebookId": "nb", "question": "q", "queries": []},
    ])
    def test_accepted(self, body):
        with patch("app.routers.retrieve.fetch_context", return_value=([], "q")), \
             patch("app.routers.retrieve.fetch_multi", return_value=[]):
            assert client.post("/retrieve", headers=QUIZ, json=body).status_code == 200

    @pytest.mark.parametrize("body", [
        {"notebookId": "nb"},
        {"question": "q"},
        {"notebookId": "nb", "question": "q", "k": 41},
        {"notebookId": "nb", "queries": [str(i) for i in range(13)]},
        {"notebookId": "nb", "queries": ["  "]},
        {"notebookIds": ["  "], "question": "q"},
    ])
    def test_rejected(self, body):
        assert client.post("/retrieve", headers=QUIZ, json=body).status_code == 422

    def test_question_and_queries_merge(self):
        with patch("app.routers.retrieve.fetch_multi", return_value=[]) as multi:
            client.post("/retrieve", headers=QUIZ, json={
                "notebookId": "nb", "question": "main", "queries": ["t1", "main", "t2"]})
        assert multi.call_args.kwargs["queries"] == ["main", "t1", "t2"]

    def test_both_id_fields_are_deduplicated(self):
        with patch("app.routers.retrieve.fetch_context", return_value=([], "q")) as fetch:
            client.post("/retrieve", headers=QUIZ, json={
                "notebookId": "nb0", "notebookIds": ["nb1", "nb1", " nb2 "], "question": "q"})
        assert fetch.call_args.kwargs["notebook_ids"] == ["nb1", "nb2"]


# ─── legacy collection behaviour ──────────────────────────────────────────────

class TestLegacyCollection:
    def test_quiz_main_ingest_unchanged(self):
        """An unflagged collection: default chunker, random-suffix ids, raw vectors, no task type."""
        name = _uid("legacy_ingest").replace("-", "_")
        get_client().create_collection(name=name, metadata={"embedding_dimensions": 768})
        import asyncio

        from app.services import ingest_pipeline

        text = _page_text("mitochondria", 200)
        with patch.object(ingest_pipeline, "download_bytes", return_value=text.encode()), \
             patch.object(ingest_pipeline, "embed_texts", side_effect=fake_embed_texts):
            n = asyncio.run(ingest_pipeline.run_ingest("nbL", "srcL", "https://x/y?token=t", "notes.txt", name))
        got = get_collection(name).get(include=["metadatas", "embeddings"])
        assert len(got["ids"]) == n > 1
        assert all(re.fullmatch(r"srcL_\d+_[0-9a-f]{6}", i) for i in got["ids"])
        assert all(set(m) == {"notebookId", "sourceId", "source"} for m in got["metadatas"])
        assert all(not math.isclose(float(np.linalg.norm(v)), 1.0, rel_tol=1e-3) for v in got["embeddings"])
        assert all(c["task_type"] is None for c in fake_embed_texts.calls)


# ─── admin: collections, promote, delete, vacuum ──────────────────────────────

class TestAdminCollections:
    def test_list_reports_flags_space_and_apps(self):
        get_collection(QUIZ_LIVE)
        resp = client.get("/admin/collections", headers=ADMIN)
        assert resp.status_code == 200
        body = resp.json()
        by_name = {c["name"]: c for c in body["collections"]}
        assert by_name[QUIZ_LIVE]["servesApp"] == "quiz_main"
        assert by_name[QUIZ_LIVE]["space"] == "cosine"
        assert {"segments", "diskBytes", "metadata", "normalized", "count"} <= set(by_name[QUIZ_LIVE])
        assert body["sqliteBytes"] > 0
        assert isinstance(body["orphanSegmentDirs"], list)

    def test_admin_requires_admin_key(self):
        assert client.get("/admin/collections", headers=DAFTAR).status_code == 401

    def test_delete_refuses_live_and_needs_confirm(self):
        get_collection(QUIZ_LIVE)
        assert client.delete(f"/admin/collections/{QUIZ_LIVE}?confirm={QUIZ_LIVE}", headers=ADMIN).status_code == 409
        name = _uid("dropme").replace("-", "_")
        get_collection(name).add(ids=["a"], embeddings=[[0.1] * 768])
        assert client.delete(f"/admin/collections/{name}?confirm=nope", headers=ADMIN).status_code == 400
        resp = client.delete(f"/admin/collections/{name}?confirm={name}", headers=ADMIN)
        assert resp.status_code == 200, resp.text
        assert resp.json()["count"] == 1
        assert client.delete(f"/admin/collections/{name}?confirm={name}", headers=ADMIN).status_code == 404

    def test_promote_swaps_staging_in(self):
        staging = DAFTAR_LIVE + "__next"
        nb, src = _uid("nbP"), _uid("srcP")
        # the live collection exists (legacy, unflagged) with one chunk
        live_before = get_collection(DAFTAR_LIVE)
        live_before.add(ids=["legacy-1"], embeddings=[[0.2] * 768], documents=["old"],
                        metadatas=[{"notebookId": "old-nb", "sourceId": "old-src", "source": "o"}])
        old_count = live_before.count()
        assert _ingest_text(DAFTAR, nb, src, [(1, _page_text("history"))], staging).status_code == 200

        assert client.post("/admin/apps/daftar_ai/promote", headers=ADMIN,
                           json={"staging": QUIZ_LIVE + "__next"}).status_code == 400
        assert client.post("/admin/apps/daftar_ai/promote", headers=ADMIN,
                           json={"staging": DAFTAR_LIVE + "__missing"}).status_code == 404

        resp = client.post("/admin/apps/daftar_ai/promote", headers=ADMIN, json={"staging": staging})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["live"] == DAFTAR_LIVE
        assert re.fullmatch(DAFTAR_LIVE + r"__old_\d{12}(_\d+)?", body["old"])
        assert body["counts"][body["old"]] == old_count
        assert body["space"] == "cosine" and body["flags"]["normalized"]

        # the live name now serves the promoted data, through fresh handles
        assert collection_flags(DAFTAR_LIVE).pages
        resp = client.post("/retrieve", headers=DAFTAR, json={"notebookId": nb, "queries": ["history"]})
        assert resp.status_code == 200 and resp.json()["chunks"]
        names = [c["name"] for c in client.get("/admin/collections", headers=ADMIN).json()["collections"]]
        assert staging not in names and body["old"] in names
        # the kept old collection is readable through ?collection=
        resp = client.get(f"/admin/chunks?app=daftar_ai&collection={body['old']}", headers=ADMIN)
        assert resp.status_code == 200 and resp.json()["total"] == old_count

    def test_vacuum(self):
        name = _uid("vac").replace("-", "_")
        col = get_collection(name)
        col.add(ids=[f"v{i}" for i in range(300)], embeddings=np.random.default_rng(0).normal(size=(300, 768)).tolist(),
                documents=["filler text " * 50] * 300)
        col.delete(ids=[f"v{i}" for i in range(300)])
        resp = client.post("/admin/vacuum", headers=ADMIN)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["quickCheck"] == "ok"
        assert body["afterBytes"] <= body["beforeBytes"]
        # Chroma keeps working on the same client afterwards
        get_collection(QUIZ_LIVE).count()
        col.add(ids=["after"], embeddings=[[0.3] * 768])
        assert col.count() == 1

    def test_stats_reports_space_process_versions(self):
        resp = client.get("/admin/stats?app=quiz_main", headers=ADMIN)
        assert resp.status_code == 200
        body = resp.json()
        assert body["embedding"]["space"] in ("cosine", "l2")
        assert "rssBytes" in body["process"]
        assert body["versions"]["chromadb"]

    def test_write_fence_blocks_during_maintenance(self):
        from app.db.chroma import MaintenanceError, maintenance, writing

        with maintenance("test") as in_flight:
            assert isinstance(in_flight, dict)
            with pytest.raises(MaintenanceError):
                with writing("x"):
                    pass
            resp = _ingest_text(DAFTAR, "nb", "s", [(1, "text " * 100)], DAFTAR_LIVE + "__fence")
            assert resp.status_code == 503
            assert client.get("/health").json()["chromadb"] == "maintenance"
        with writing("x"):
            pass
