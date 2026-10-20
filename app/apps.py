"""
Per-app (tenant) registry.

One deployment serves several products — quiz_main and daftar_ai today — and
each gets its own ChromaDB collection. Notebook IDs are only unique within a
product, so a shared collection means one app's `notebookId` filter can match
the other app's chunks, and "delete every chunk of notebook X" can reach
across the boundary. Separate collections make that structurally impossible.

An app is identified by the API key the caller presents, never by a field in
the request body: a leaked daftar_ai key must not be able to read or delete
quiz_main's index simply by naming it. Admin routes are the exception — they
hold a separate credential and select the app explicitly.

Configuration (all optional). With none of it set the server behaves exactly
as the single-tenant version did: one app, `notebook_docs`, RAG_API_KEY.

    RAG_APPS=quiz_main:notebook_docs,daftar_ai:daftar_notebook_docs
    RAG_API_KEY=<key for the first app listed>
    RAG_APP_KEYS=daftar_ai:<key>

`RAG_APPS` is a comma-separated list of `id` or `id:collection` entries; the
collection defaults to `<id>_docs` and the first entry is the default app.
`RAG_APP_KEYS` is a comma-separated list of `id:key` pairs; the default app
falls back to RAG_API_KEY, so an existing deployment needs no new secret.
"""

import logging
import re
import secrets
from dataclasses import dataclass

from app.config import settings

logger = logging.getLogger(__name__)

# The identity the single-tenant server had, kept as the fallback so an
# unconfigured deployment keeps serving the collection it already wrote.
DEFAULT_APP_ID = "quiz_main"
DEFAULT_COLLECTION = "notebook_docs"


@dataclass(frozen=True)
class App:
    id: str
    collection: str
    api_key: str

    def __repr__(self) -> str:  # never let a key reach a log line
        return f"App(id={self.id!r}, collection={self.collection!r})"


class AppConfigError(RuntimeError):
    """Raised at import time — a misconfigured tenant must not boot silently."""


def _parse_pairs(raw: str) -> list[tuple[str, str]]:
    """Parse `a:1,b:2` into [(a, 1), (b, 2)]. A bare `a` yields (a, "")."""
    pairs: list[tuple[str, str]] = []
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        key, sep, value = entry.partition(":")
        pairs.append((key.strip(), value.strip() if sep else ""))
    return pairs


def _build_registry() -> tuple[dict[str, App], str]:
    declared = _parse_pairs(settings.rag_apps)
    if not declared:
        declared = [(DEFAULT_APP_ID, DEFAULT_COLLECTION)]

    keys = dict(_parse_pairs(settings.rag_app_keys))
    default_id = declared[0][0]

    apps: dict[str, App] = {}
    for app_id, collection in declared:
        if app_id in apps:
            raise AppConfigError(f"RAG_APPS lists {app_id!r} twice")
        api_key = keys.get(app_id) or (settings.rag_api_key if app_id == default_id else "")
        if not api_key:
            raise AppConfigError(
                f"App {app_id!r} is declared in RAG_APPS but has no API key. "
                f"Add it to RAG_APP_KEYS as '{app_id}:<key>'."
            )
        apps[app_id] = App(
            id=app_id,
            collection=collection or f"{app_id}_docs",
            api_key=api_key,
        )

    # Two apps sharing a key would silently merge into whichever matched first.
    by_key: dict[str, str] = {}
    for app in apps.values():
        clash = by_key.get(app.api_key)
        if clash:
            raise AppConfigError(f"Apps {clash!r} and {app.id!r} share an API key")
        by_key[app.api_key] = app.id

    # Likewise for collections: the whole point is that they do not overlap.
    by_collection: dict[str, str] = {}
    for app in apps.values():
        clash = by_collection.get(app.collection)
        if clash:
            raise AppConfigError(
                f"Apps {clash!r} and {app.id!r} both map to collection {app.collection!r}"
            )
        by_collection[app.collection] = app.id

    return apps, default_id


APPS, DEFAULT_APP = _build_registry()


def all_apps() -> list[App]:
    return list(APPS.values())


def app_ids() -> list[str]:
    return list(APPS.keys())


def get_app(app_id: str | None) -> App:
    """Look an app up by id. `None` or empty resolves to the default app."""
    if not app_id:
        return APPS[DEFAULT_APP]
    app = APPS.get(app_id)
    if app is None:
        raise KeyError(app_id)
    return app


def app_for_key(token: str) -> App | None:
    """
    Resolve the app a bearer token belongs to.

    Compares against every app in constant time and without short-circuiting,
    so neither the match position nor the number of configured apps is
    observable from response timing.
    """
    matched: App | None = None
    for app in APPS.values():
        if secrets.compare_digest(token, app.api_key):
            matched = app
    return matched


def collection_for(app_id: str | None) -> str:
    return get_app(app_id).collection


# Chroma's own rule for names, narrowed to the characters this server uses.
_COLLECTION_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{1,61}[A-Za-z0-9]")
STAGING_SEPARATOR = "__"


class CollectionAccessError(ValueError):
    """A request named a collection its app may not use."""


def app_serving(collection: str) -> str | None:
    """The id of the app whose live collection this is, if any."""
    for app in APPS.values():
        if app.collection == collection:
            return app.id
    return None


def staging_owner(collection: str) -> str | None:
    """The app a `<live>__<suffix>` staging/old collection belongs to, if any."""
    for app in APPS.values():
        if collection.startswith(app.collection + STAGING_SEPARATOR):
            return app.id
    return None


def resolve_collection(app: App, requested: str | None) -> str:
    """
    The collection a request may address: the app's live collection, or one
    of its own staging collections — `<live>__<suffix>`, e.g.
    daftar_notebook_docs_768__next — which promotion swaps in later. A key can
    never reach another app's collections this way.
    """
    if not requested or requested == app.collection:
        return app.collection
    if not _COLLECTION_NAME.fullmatch(requested):
        raise CollectionAccessError(f"Invalid collection name {requested!r}")
    if not requested.startswith(app.collection + STAGING_SEPARATOR):
        raise CollectionAccessError(
            f"Collection {requested!r} is not {app.collection!r} or one of its "
            f"'{app.collection}{STAGING_SEPARATOR}*' staging collections"
        )
    # Even with overlapping names (live "x" vs another app's "x__y"), a name in
    # another app's live/staging namespace is never this app's.
    for other in APPS.values():
        if other.id != app.id and (
            requested == other.collection or requested.startswith(other.collection + STAGING_SEPARATOR)
        ):
            raise CollectionAccessError(f"Collection {requested!r} belongs to app {other.id!r}")
    return requested


def log_registry() -> None:
    logger.info(
        "Tenant registry: %s (default=%s)",
        ", ".join(f"{a.id}->{a.collection}" for a in APPS.values()),
        DEFAULT_APP,
    )
