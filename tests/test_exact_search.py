"""Exact search inside small filters (filtered HNSW can return nothing)."""

import numpy as np
import chromadb

from app.services import retriever


def _collection(tmp_path, space: str):
    client = chromadb.PersistentClient(path=str(tmp_path / space))
    return client.get_or_create_collection(name=f"exact_{space}", configuration={"hnsw": {"space": space}})


def _fill(col, rng):
    vecs = rng.normal(size=(300, 16)).astype(np.float32)
    col.add(
        ids=[f"c{i}" for i in range(300)],
        embeddings=vecs.tolist(),
        documents=[f"doc {i}" for i in range(300)],
        metadatas=[{"notebookId": "small" if i < 12 else "big"} for i in range(300)],
    )
    return vecs


def test_exact_query_matches_brute_force_cosine(tmp_path):
    rng = np.random.default_rng(1)
    col = _collection(tmp_path, "cosine")
    vecs = _fill(col, rng)
    q = rng.normal(size=16).astype(np.float32)
    res = retriever._exact_query(col, {"notebookId": {"$eq": "small"}}, [q.tolist()], 5)
    small = vecs[:12]
    cos = 1 - (small / np.linalg.norm(small, axis=1, keepdims=True)) @ (q / np.linalg.norm(q))
    expected = [f"c{i}" for i in np.argsort(cos)[:5]]
    assert res["ids"][0] == expected
    assert all(m["notebookId"] == "small" for m in res["metadatas"][0])
    assert res["distances"][0] == sorted(res["distances"][0])


def test_exact_query_l2_and_empty_filter(tmp_path):
    rng = np.random.default_rng(2)
    col = _collection(tmp_path, "l2")
    vecs = _fill(col, rng)
    q = rng.normal(size=16).astype(np.float32)
    res = retriever._exact_query(col, {"notebookId": {"$eq": "small"}}, [q.tolist(), q.tolist()], 3)
    d2 = ((vecs[:12] - q) ** 2).sum(1)
    assert res["ids"][0] == [f"c{i}" for i in np.argsort(d2)[:3]]
    assert len(res["ids"]) == 2
    empty = retriever._exact_query(col, {"notebookId": {"$eq": "none"}}, [q.tolist()], 3)
    assert empty["ids"] == [[]]


def test_search_uses_exact_for_small_scopes(tmp_path, monkeypatch):
    rng = np.random.default_rng(3)
    col = _collection(tmp_path, "cosine")
    _fill(col, rng)
    called = {}

    def fake_query(**kwargs):
        called["hnsw"] = True
        return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

    monkeypatch.setattr(col, "query", fake_query, raising=False)
    q = rng.normal(size=16).tolist()
    res = retriever._search(col, {"notebookId": {"$eq": "small"}}, [q], 4, 12)
    assert len(res["ids"][0]) == 4 and "hnsw" not in called
    # Large scope: HNSW answers short (as filtered search does) -> exact fallback.
    monkeypatch.setattr(retriever.settings, "exact_search_max", 5)
    res = retriever._search(col, {"notebookId": {"$eq": "small"}}, [q], 4, 12)
    assert called.get("hnsw") and len(res["ids"][0]) == 4
