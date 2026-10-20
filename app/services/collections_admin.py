"""
Collection-level administration: inventory, deletion, staging promotion and
SQLite VACUUM, plus the process/runtime facts /admin/stats and /health report.

Disk is the scarce resource on the Render volume and Chroma reclaims none of
it on its own: deleted chunks leave free pages in chroma.sqlite3 (VACUUM gets
them back) and a deleted collection leaves its HNSW directory behind. The
inventory therefore maps every segment directory to its collection and lists
the orphans.

Every mutation here takes the write fence (app/db/chroma.py `maintenance`):
it refuses to start while a relevant write is in flight, and writes that
arrive meanwhile fail fast with HTTP 503.
"""

import logging
import os
import platform
import re
import shutil
import sqlite3
import time
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path

from app.apps import STAGING_SEPARATOR, App, app_serving, staging_owner
from app.db.chroma import (
    collection_exists,
    collection_space,
    delete_collection,
    flags_from_metadata,
    get_client,
    chroma_dir,
    list_collection_names,
    maintenance,
    segments_by_collection,
    space_from_info,
    sqlite_path,
    stored_collection_info,
    stored_counts,
    stored_dimensions,
    swap_in_collection,
    writes_in_flight,
)
from app.log_utils import log_step
from app.services.admin_index import _dir_size, human_bytes, invalidate_snapshot

logger = logging.getLogger(__name__)

_UUID_DIR = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
# VACUUM rewrites the whole file: the rollback journal needs room on the
# volume, the rebuilt copy needs room in SQLite's temp directory.
VACUUM_FREE_FACTOR = 1.2


class AdminConflict(RuntimeError):
    """The operation would be unsafe right now (HTTP 409)."""


class AdminNotFound(LookupError):
    """No such collection (HTTP 404)."""


# --- Runtime facts -------------------------------------------------------------

def _version(package: str) -> str | None:
    try:
        return importlib_metadata.version(package)
    except importlib_metadata.PackageNotFoundError:
        return None


def versions(full: bool = False) -> dict:
    """Library versions. /health gets the short list, /admin/stats the full one."""
    out = {
        "chromadb": _version("chromadb"),
        "litellm": _version("litellm"),
        "python": platform.python_version(),
    }
    if full:
        for package in ("fastapi", "pydantic", "numpy", "httpx", "pypdf", "pypdfium2", "pillow"):
            out[package] = _version(package)
    return out


def _read_int(path: str) -> int | None:
    try:
        with open(path, encoding="ascii") as f:
            raw = f.read().strip()
    except OSError:
        return None
    return int(raw) if raw.isdigit() else None  # "max" (no limit) -> None


def process_memory() -> dict:
    """This process's RSS and the container's cgroup memory limit/usage, when readable."""
    rss = None
    try:
        with open("/proc/self/status", encoding="ascii") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss = int(line.split()[1]) * 1024
                    break
    except OSError:
        try:
            import psutil  # type: ignore[import-not-found]

            rss = psutil.Process().memory_info().rss
        except Exception:
            rss = None
    # cgroup v2 first, then v1.
    limit = _read_int("/sys/fs/cgroup/memory.max")
    current = _read_int("/sys/fs/cgroup/memory.current")
    if limit is None and current is None:
        limit = _read_int("/sys/fs/cgroup/memory/memory.limit_in_bytes")
        current = _read_int("/sys/fs/cgroup/memory/memory.usage_in_bytes")
        if limit is not None and limit >= 1 << 60:  # v1's "unlimited"
            limit = None
    return {
        "rssBytes": rss,
        "rssHuman": human_bytes(rss),
        "cgroupMemoryMaxBytes": limit,
        "cgroupMemoryCurrentBytes": current,
        "cgroupMemoryMaxHuman": human_bytes(limit),
        "cgroupMemoryCurrentHuman": human_bytes(current),
    }


# --- Inventory -----------------------------------------------------------------

def collections_report() -> dict:
    """Every collection with its size, flags, real space and on-disk segments."""
    t0 = time.perf_counter()
    base = chroma_dir()
    segments = segments_by_collection()
    dims = stored_dimensions()
    # Everything comes from chroma.sqlite3: opening a collection and counting it
    # loads its HNSW index into RAM, and a large unused collection (the old
    # 3072-dim ones) is enough to run a 2 GB instance out of memory.
    counts = stored_counts()
    infos = stored_collection_info()

    rows = []
    for name in sorted(infos):
        info = infos[name]
        meta = dict(info.get("metadata") or {})
        count = counts.get(name)
        space = space_from_info(info)
        col_id = str(info.get("id"))
        error = None
        flags = flags_from_metadata(meta)
        segs = []
        disk = 0
        for seg in segments.get(name, []):
            path = os.path.join(base, seg["id"])
            size = _dir_size(path) if os.path.isdir(path) else 0
            disk += size
            segs.append({**seg, "diskBytes": size, "diskHuman": human_bytes(size)})
        rows.append({
            "name": name,
            "id": col_id,
            "count": count,
            "metadata": meta,
            "dimensions": dims.get(name) or (flags.dimensions if meta.get("embedding_dimensions") else None),
            "space": space,
            "normalized": flags.normalized,
            "flags": {
                "normalized": flags.normalized,
                "taskTypes": flags.task_types,
                "chunker": flags.chunker,
                "pdfVision": flags.pdf_vision,
            },
            "servesApp": app_serving(name),
            "stagingOf": staging_owner(name),
            "segments": segs,
            "diskBytes": disk,
            "diskHuman": human_bytes(disk),
            **({"error": error} if error else {}),
        })

    known = {seg["id"] for segs in segments.values() for seg in segs}
    orphans = []
    if os.path.isdir(base):
        for entry in sorted(os.listdir(base)):
            path = os.path.join(base, entry)
            if os.path.isdir(path) and _UUID_DIR.fullmatch(entry) and entry not in known:
                size = _dir_size(path)
                orphans.append({"id": entry, "diskBytes": size, "diskHuman": human_bytes(size)})

    db = sqlite_path()
    db_bytes = os.path.getsize(db) if os.path.isfile(db) else None
    return {
        "path": base,
        "sqliteBytes": db_bytes,
        "sqliteHuman": human_bytes(db_bytes),
        "sqlite": _sqlite_pages(db),
        "collections": rows,
        "orphanSegmentDirs": orphans,
        "writesInFlight": writes_in_flight(),
        "elapsedMs": int((time.perf_counter() - t0) * 1000),
    }


def _sqlite_pages(db: str) -> dict | None:
    if not os.path.isfile(db):
        return None
    try:
        con = sqlite3.connect(Path(db).resolve().as_uri() + "?mode=ro", uri=True, timeout=10)
        try:
            page_size, page_count, freelist = (
                con.execute(f"PRAGMA {p}").fetchone()[0] for p in ("page_size", "page_count", "freelist_count")
            )
        finally:
            con.close()
    except sqlite3.Error as exc:
        return {"error": str(exc)}
    return {
        "pageSize": page_size,
        "pageCount": page_count,
        "freelistPages": freelist,
        "reclaimableBytes": freelist * page_size,
        "reclaimableHuman": human_bytes(freelist * page_size),
    }


# --- Delete -------------------------------------------------------------------

def _reembed_touches(name: str) -> bool:
    from app.services import reembed

    job = reembed.status()
    return bool(job and job.get("state") == "running" and name in (job.get("source"), job.get("target")))


def delete_collection_checked(name: str) -> dict:
    """
    Drop a collection that serves no app, then remove its HNSW directories
    (Chroma leaves them behind). Refuses a live collection, one with writes in
    flight, and the source/target of a running re-embed job.
    """
    owner = app_serving(name)
    if owner:
        raise AdminConflict(f"{name!r} is the live collection of app {owner!r}")
    if _reembed_touches(name):
        raise AdminConflict(f"{name!r} is part of a running re-embed job")
    if not collection_exists(name):
        raise AdminNotFound(f"No collection {name!r}")

    seg_ids = [seg["id"] for seg in segments_by_collection().get(name, [])]
    with maintenance(f"delete of {name}") as in_flight:
        if in_flight.get(name):
            raise AdminConflict(f"{in_flight[name]} write(s) to {name!r} in flight")
        count = stored_counts().get(name)  # never load the HNSW index just to drop it
        delete_collection(name)

    base = chroma_dir()
    removed, kept = [], []
    for seg_id in seg_ids:
        path = os.path.join(base, seg_id)
        if not os.path.isdir(path):
            continue
        size = _dir_size(path)
        try:
            shutil.rmtree(path)
            removed.append({"id": seg_id, "diskBytes": size})
        except OSError as exc:  # e.g. still memory-mapped on Windows
            kept.append({"id": seg_id, "diskBytes": size, "error": str(exc)})
    invalidate_snapshot(name)
    log_step(
        logger, "ADMIN_DROP_COLLECTION",
        f"{name}: deleted {count} chunks, removed dirs={len(removed)} kept={len(kept)}",
    )
    return {
        "deleted": name,
        "count": count,
        "removedSegmentDirs": removed,
        "keptSegmentDirs": kept,
        "freedBytes": sum(r["diskBytes"] for r in removed),
        "note": "Run POST /admin/vacuum to reclaim the SQLite space of the deleted rows.",
    }


# --- Promote ------------------------------------------------------------------

def promote(app: App, staging: str) -> dict:
    """
    Make `staging` (an existing `<live>__<suffix>` collection) the app's live
    collection: live -> `<live>__old_<YYYYMMDDHHMM>`, staging -> live. The
    configured name never changes, so RAG_APPS stays as it is.
    """
    live = app.collection
    if not staging or staging == live or not staging.startswith(live + STAGING_SEPARATOR):
        raise ValueError(f"staging must be one of {live!r}'s '{live}{STAGING_SEPARATOR}*' collections")
    if not collection_exists(staging):
        raise AdminNotFound(f"No collection {staging!r}")
    for name in (live, staging):
        if _reembed_touches(name):
            raise AdminConflict(f"{name!r} is part of a running re-embed job")

    with maintenance(f"promotion of {staging}") as in_flight:
        busy = {name: n for name, n in in_flight.items() if name in (live, staging)}
        if busy:
            raise AdminConflict(f"writes in flight: {busy}")
        old = None
        if collection_exists(live):
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M")
            old = f"{live}{STAGING_SEPARATOR}old_{stamp}"
            suffix = 2
            while collection_exists(old):
                old = f"{live}{STAGING_SEPARATOR}old_{stamp}_{suffix}"
                suffix += 1
        swap_in_collection(live, staging, old)

    for name in (live, staging, old):
        if name:
            invalidate_snapshot(name)
    client = get_client()
    promoted = client.get_collection(name=live)
    counts = {live: promoted.count()}
    if old:
        counts[old] = client.get_collection(name=old).count()
    flags = flags_from_metadata(promoted.metadata)
    log_step(logger, "ADMIN_PROMOTE", f"app={app.id} {staging} -> {live}; previous live -> {old} counts={counts}")
    return {
        "app": app.id,
        "live": live,
        "old": old,
        "staging": staging,
        "counts": counts,
        "space": collection_space(promoted),
        "flags": {
            "normalized": flags.normalized,
            "taskTypes": flags.task_types,
            "chunker": flags.chunker,
            "pdfVision": flags.pdf_vision,
            "dimensions": flags.dimensions,
        },
    }


# --- VACUUM -------------------------------------------------------------------

def vacuum() -> dict:
    """
    SQLite VACUUM of chroma.sqlite3 through a separate connection while the
    Chroma client stays open. Verified on chromadb 1.5.9 (journal_mode=delete):
    concurrent Chroma reads/writes wait for it and work afterwards, and
    integrity_check passes. While it runs every Chroma call blocks (minutes on
    a multi-GB file), so run it in a quiet window. Refused (AdminConflict)
    when free space is short or a write is in flight.
    """
    import tempfile

    db = sqlite_path()
    if not os.path.isfile(db):
        raise AdminNotFound(f"No database at {db}")
    size = os.path.getsize(db)
    volume_free = shutil.disk_usage(os.path.dirname(db)).free
    temp_dir = os.environ.get("SQLITE_TMPDIR") or tempfile.gettempdir()
    temp_free = shutil.disk_usage(temp_dir).free
    if volume_free < VACUUM_FREE_FACTOR * size:
        raise AdminConflict(
            f"VACUUM needs {human_bytes(int(VACUUM_FREE_FACTOR * size))} free on the volume, "
            f"only {human_bytes(volume_free)} is free"
        )
    if temp_free < size:
        raise AdminConflict(
            f"VACUUM builds a temporary copy in {temp_dir}: needs {human_bytes(size)}, "
            f"only {human_bytes(temp_free)} is free"
        )

    before = _sqlite_pages(db) or {}
    with maintenance("vacuum") as in_flight:
        if in_flight:
            raise AdminConflict(f"writes in flight: {in_flight}")
        t0 = time.perf_counter()
        con = sqlite3.connect(db, timeout=120, isolation_level=None)
        try:
            con.execute("PRAGMA busy_timeout = 120000")
            con.execute("VACUUM")
            check = con.execute("PRAGMA quick_check").fetchone()[0]
        finally:
            con.close()
        elapsed = int((time.perf_counter() - t0) * 1000)
    after_size = os.path.getsize(db)
    log_step(
        logger, "ADMIN_VACUUM",
        f"{human_bytes(size)} -> {human_bytes(after_size)} in {elapsed}ms quick_check={check}",
    )
    return {
        "beforeBytes": size,
        "afterBytes": after_size,
        "freedBytes": size - after_size,
        "beforeHuman": human_bytes(size),
        "afterHuman": human_bytes(after_size),
        "freelistPagesBefore": before.get("freelistPages"),
        "elapsedMs": elapsed,
        "quickCheck": check,
    }
