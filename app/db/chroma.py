import logging
import os
import sqlite3
import tempfile
import threading
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import chromadb

from app.config import settings
from app.log_utils import get_trace_id, log_step

logger = logging.getLogger(__name__)

_client: chromadb.PersistentClient | None = None
_client_path: str | None = None
_lock = threading.Lock()

# Collections are per app (see app/apps.py); within one, notebooks are isolated
# by the `notebookId` metadata field. This name is the single-tenant default —
# the collection the server wrote before tenants existed, kept so an
# unconfigured deployment still finds its data.
COLLECTION_NAME = "notebook_docs"

# get_or_create_collection is a round trip; the handles are cheap and stable,
# so keep one per name rather than paying for it on every chunk read.
_collections: dict[str, chromadb.Collection] = {}
_collections_lock = threading.Lock()


def _resolve_chroma_path() -> str:
    """
    Return a writable path for ChromaDB storage.
    Tries the configured path first; falls back to /tmp/chroma if not writable
    (e.g. Render persistent disk not yet mounted).
    """
    path = settings.chroma_path
    try:
        os.makedirs(path, exist_ok=True)
        # Quick write-access probe
        probe = os.path.join(path, ".write_probe")
        with open(probe, "w") as f:
            f.write("ok")
        os.remove(probe)
        logger.info(f"ChromaDB path is writable: {path}")
        return path
    except OSError as e:
        fallback = os.path.join(tempfile.gettempdir(), "chroma")
        logger.warning(
            f"ChromaDB path '{path}' is not writable ({e}). "
            f"Falling back to {fallback}. "
            f"Data will NOT persist across restarts. "
            f"Fix: attach a persistent disk at '{path}' in the Render dashboard."
        )
        os.makedirs(fallback, exist_ok=True)
        return fallback


def get_client() -> chromadb.PersistentClient:
    """Return the singleton ChromaDB persistent client."""
    global _client, _client_path
    if _client is None:
        with _lock:
            if _client is None:
                path = _resolve_chroma_path()
                logger.info(f"Initializing ChromaDB at path: {path}")
                _client = chromadb.PersistentClient(path=path)
                _client_path = path
    return _client


def chroma_dir() -> str:
    """The directory the client actually opened (may be the /tmp fallback)."""
    get_client()
    return _client_path or settings.chroma_path


# Vector size of collections created before EMBEDDING_DIMENSIONS existed:
# gemini-embedding-001's full output. Those collections carry no
# `embedding_dimensions` metadata, so this is what an absent value means.
LEGACY_DIMENSIONS = 3072

# ─── Per-collection behaviour flags ───────────────────────────────────────────
# Stored in the metadata of every collection created from now on (design
# contract §4). A collection without them — quiz_main's notebook_docs_768,
# daftar's current index — keeps the original behaviour exactly: default
# chunker, raw vectors, no task types, text-only PDFs.
PAGES_CHUNKER = "pages_v1"
# Upper bound for one Chroma add/upsert/delete call (Chroma's own is ~5.4k).
WRITE_BATCH = 2000


@dataclass(frozen=True)
class CollectionFlags:
    dimensions: int = LEGACY_DIMENSIONS
    normalized: bool = False   # L2-normalise every stored and query vector
    task_types: bool = False   # RETRIEVAL_DOCUMENT / RETRIEVAL_QUERY embeddings
    chunker: str | None = None  # "pages_v1" -> page-aware chunker, deterministic ids
    pdf_vision: bool = False   # PDF pages with junk text / figures go through DeepSeek

    @property
    def pages(self) -> bool:
        return self.chunker == PAGES_CHUNKER


def new_collection_metadata() -> dict:
    return {
        "embedding_dimensions": settings.embedding_dimensions,
        "normalized": True,
        "chunker": PAGES_CHUNKER,
        "pdf_vision": True,
        "task_types": True,
    }


def flags_from_metadata(meta: dict | None) -> CollectionFlags:
    meta = meta or {}
    try:
        dims = int(meta.get("embedding_dimensions") or LEGACY_DIMENSIONS)
    except (TypeError, ValueError):
        dims = LEGACY_DIMENSIONS
    chunker = meta.get("chunker")
    return CollectionFlags(
        dimensions=dims,
        normalized=meta.get("normalized") is True,
        task_types=meta.get("task_types") is True,
        chunker=chunker if isinstance(chunker, str) and chunker else None,
        pdf_vision=meta.get("pdf_vision") is True,
    )


def collection_flags(name: str | None = None) -> CollectionFlags:
    """The behaviour flags of a collection (all off for legacy collections)."""
    return flags_from_metadata(get_collection(name).metadata)


def collection_space(col: chromadb.Collection) -> str | None:
    """The distance space the index really uses (l2 / cosine / ip)."""
    cfg = getattr(col, "configuration_json", None)
    if isinstance(cfg, dict):
        for key in ("hnsw", "spann", "hnsw_configuration"):
            section = cfg.get(key)
            if isinstance(section, dict) and section.get("space"):
                return str(section["space"])
    space = (col.metadata or {}).get("hnsw:space")  # chromadb < 1.0
    return str(space) if space else None


def _create_collection(name: str) -> chromadb.Collection:
    # An existing collection is opened as-is, never re-created with new
    # metadata: its stored vectors fix its dimension, whatever is configured.
    try:
        return get_client().get_collection(name=name)
    except Exception:
        pass
    # New collections record the vector size they are built with (see
    # dimensions_for) and the behaviour flags above. The space goes in
    # `configuration={"hnsw": ...}`: chromadb 1.x silently ignores the old
    # `{"hnsw:space": ...}` configuration key, which is why every collection
    # created before this change is L2.
    meta = new_collection_metadata()
    try:
        col = get_client().get_or_create_collection(
            name=name,
            configuration={"hnsw": {"space": "cosine"}},
            metadata=meta,
        )
    except Exception as exc:
        # chromadb < 1.0 has no dict configuration; it reads the metadata key.
        logger.warning(f"[chroma] configuration= rejected ({exc}); using legacy hnsw:space metadata")
        col = get_client().get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine", **meta},
        )
    space = collection_space(col)
    if space != "cosine":
        logger.error(f"[chroma] collection {name!r} was created with space={space!r}, expected cosine")
    logger.info(f"[chroma] Created collection {name!r} space={space} metadata={col.metadata}")
    return col


def dimensions_for(name: str | None = None) -> int:
    """The embedding size a collection's vectors were built with."""
    return collection_flags(name).dimensions


def get_collection(name: str | None = None, create: bool = True) -> chromadb.Collection:
    """
    Return one collection, creating it if necessary. With create=False a
    missing collection raises chromadb's NotFoundError (or ValueError on old
    versions) instead — for reads of staging names that must not conjure an
    empty collection into existence.
    """
    name = name or COLLECTION_NAME
    cached = _collections.get(name)
    if cached is not None:
        return cached
    with _collections_lock:
        cached = _collections.get(name)
        if cached is None:
            cached = _create_collection(name) if create else get_client().get_collection(name=name)
            _collections[name] = cached
        return cached


def collection_exists(name: str) -> bool:
    try:
        get_collection(name, create=False)
        return True
    except Exception:
        return False


def forget_collections(*names: str) -> None:
    """Drop cached handles, after a collection was renamed or deleted."""
    with _collections_lock:
        for name in names:
            _collections.pop(name, None)


def list_collection_names() -> list[str]:
    names = []
    for item in get_client().list_collections():
        names.append(item if isinstance(item, str) else item.name)  # 0.6.0 returned names
    return sorted(names)


def rename_collection(old: str, new: str) -> None:
    """Rename in place (same id, same segments). Fails if `new` exists."""
    get_client().get_collection(name=old).modify(name=new)
    forget_collections(old, new)
    logger.info(f"[chroma] Renamed collection {old!r} -> {new!r}")


def swap_in_collection(live: str, staging: str, old: str | None) -> None:
    """
    Promotion: rename `live` -> `old` (when `old` is given), then `staging` ->
    `live`; if the second rename fails the first is undone. Runs under the
    handle-cache lock, so no request can create an empty `live` between the
    two renames (get_collection would otherwise do exactly that), and drops
    the cached handles of all three names. Callers fence writes first.
    """
    client = get_client()
    with _collections_lock:
        try:
            if old:
                client.get_collection(name=live).modify(name=old)
            try:
                client.get_collection(name=staging).modify(name=live)
            except Exception:
                if old:
                    client.get_collection(name=old).modify(name=live)
                raise
        finally:
            for name in (live, staging, old):
                if name:
                    _collections.pop(name, None)
    logger.info(f"[chroma] Promoted {staging!r} -> {live!r} (previous live kept as {old!r})")


def delete_collection(name: str) -> None:
    get_client().delete_collection(name)
    forget_collections(name)
    logger.info(f"[chroma] Deleted collection {name!r}")


def segments_by_collection() -> dict[str, list[dict]]:
    """
    {collection name: [{id, scope, type}]}, read from chroma.sqlite3 through a
    read-only connection. A VECTOR segment's id is the name of its HNSW
    directory on disk; deleting a collection leaves that directory behind.
    """
    out: dict[str, list[dict]] = {}
    for name, seg_id, scope, seg_type in _read_sqlite(
        "SELECT c.name, s.id, s.scope, s.type FROM segments s JOIN collections c ON s.collection = c.id"
    ):
        out.setdefault(name, []).append({"id": seg_id, "scope": scope, "type": seg_type})
    return out


def stored_dimensions() -> dict[str, int | None]:
    """{collection name: vector size Chroma recorded (None until the first add)}."""
    return {name: dim for name, dim in _read_sqlite("SELECT name, dimension FROM collections")}


def sqlite_path() -> str:
    return os.path.join(chroma_dir(), "chroma.sqlite3")


def _read_sqlite(sql: str) -> list[tuple]:
    db = sqlite_path()
    if not os.path.isfile(db):
        return []
    con = sqlite3.connect(Path(db).resolve().as_uri() + "?mode=ro", uri=True, timeout=10)
    try:
        return con.execute(sql).fetchall()
    finally:
        con.close()


# ─── Write fence ──────────────────────────────────────────────────────────────
# Promotion renames collections and VACUUM rewrites the database file; neither
# may run while this process is writing. Writers register here. Maintenance
# refuses to start while a write it cares about is in flight, and while it
# runs new writers fail fast with MaintenanceError (HTTP 503) instead of
# waiting — a wait would hold a worker thread, or the event loop, for minutes.
# The lock itself is only ever held for a counter update.

_write_lock = threading.Lock()
_writes: Counter = Counter()
_maintenance: str | None = None


class MaintenanceError(RuntimeError):
    """A write arrived while a promote/vacuum was running (retry later)."""


@contextmanager
def writing(name: str):
    """Mark a write to collection `name` as in flight (resolve handles inside)."""
    with _write_lock:
        if _maintenance:
            raise MaintenanceError(f"{_maintenance} in progress; retry the write in a few minutes")
        _writes[name] += 1
    try:
        yield
    finally:
        with _write_lock:
            _writes[name] -= 1
            if _writes[name] <= 0:
                del _writes[name]


@contextmanager
def maintenance(what: str):
    """
    Block new writes for the duration; yields the writes that were already in
    flight ({collection: count}) so the caller can refuse to proceed.
    """
    global _maintenance
    with _write_lock:
        if _maintenance:
            raise MaintenanceError(f"{_maintenance} already in progress")
        _maintenance = what
        in_flight = dict(_writes)
    try:
        yield in_flight
    finally:
        with _write_lock:
            _maintenance = None


def writes_in_flight() -> dict[str, int]:
    with _write_lock:
        return dict(_writes)


def maintenance_running() -> str | None:
    """What maintenance (promotion, vacuum, ...) is running now, if any."""
    return _maintenance


def count_for_notebooks(
    notebook_ids: list[str],
    collection: str | None = None,
    source_ids: list[str] | None = None,
) -> int:
    """
    Return the number of stored chunks for the given notebook (or source) IDs.
    Uses collection.get() with include=[] (IDs only) for efficiency.
    """
    trace = get_trace_id()
    col = get_collection(collection)
    where = _build_where(notebook_ids, source_ids)
    try:
        result = col.get(where=where, include=[])
        count = len(result["ids"])
        log_step(
            logger, "CHROMA_COUNT",
            f"collection={col.name} notebook_ids={notebook_ids!r} where={where!r} count={count}",
        )
        return count
    except Exception as e:
        log_step(
            logger, "CHROMA_COUNT",
            f"collection={col.name} notebook_ids={notebook_ids!r} ERROR: {e}",
        )
        return 0


def _field_in(field: str, values: list[str]) -> dict:
    if len(values) == 1:
        return {field: {"$eq": values[0]}}
    return {field: {"$in": list(values)}}


def _build_where(notebook_ids: list[str], source_ids: list[str] | None = None) -> dict:
    """
    Build a ChromaDB where-filter: chunks of any of `notebook_ids` OR any of
    `source_ids` (a source shared into a notebook from another one). A single
    id stays the `$eq` filter the server always used; several use `$in`,
    which matches exactly what the old `$or` of `$eq`s did.
    """
    branches = []
    if notebook_ids:
        branches.append(_field_in("notebookId", notebook_ids))
    if source_ids:
        branches.append(_field_in("sourceId", source_ids))
    if not branches:
        raise ValueError("A where-filter needs at least one notebook or source id")
    return branches[0] if len(branches) == 1 else {"$or": branches}


def reset_collection(collection: str | None = None) -> int:
    """
    Drop and recreate one app's collection, wiping all its documents.
    Returns the doc count before reset. The new collection is created like
    any new one: cosine, with the behaviour flags.
    """
    name = collection or COLLECTION_NAME
    client = get_client()
    try:
        old_count = get_collection(name).count()
    except Exception:
        old_count = 0
    client.delete_collection(name)
    with _collections_lock:
        _collections.pop(name, None)
    get_collection(name)  # recreate
    logger.info(f"[chroma] Collection '{name}' reset. Deleted {old_count} documents.")
    return old_count
