import secrets
from dataclasses import replace

from fastapi import HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.apps import App, CollectionAccessError, app_for_key, get_app, resolve_collection
from app.config import settings

_bearer = HTTPBearer(auto_error=True)


def resolve_app(
    credentials: HTTPAuthorizationCredentials = Security(_bearer),
) -> App:
    """
    Authenticate an ingest/retrieve caller and return the app it belongs to.

    The presented key *is* the tenant selector — there is deliberately no way
    to name a different app in the request body, so one product's credential
    can never touch another product's collection.
    """
    app = app_for_key(credentials.credentials)
    if app is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
    return app


def verify_api_key(
    credentials: HTTPAuthorizationCredentials = Security(_bearer),
) -> None:
    """Back-compat alias for routes that only need the key checked."""
    resolve_app(credentials)


def verify_admin_key(
    credentials: HTTPAuthorizationCredentials = Security(_bearer),
) -> None:
    """
    Dependency for the /admin routes.

    Prefers ADMIN_API_KEY so the manager can hold a credential that is not the
    one every Cloud Function already carries. When it is unset, RAG_API_KEY is
    accepted instead — an existing deployment keeps working, at the cost of the
    ingest key also granting delete-anything access. Set ADMIN_API_KEY in
    production.

    Unlike resolve_app, this grants access to every app: the manager's whole
    job is to see across them.
    """
    if not settings.enable_admin:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Not Found",
        )
    expected = settings.admin_api_key or settings.rag_api_key
    if not secrets.compare_digest(credentials.credentials, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin key",
        )


def admin_app(app: str | None = None, collection: str | None = None) -> App:
    """
    Resolve the `?app=` query parameter on an admin route.

    Omitted means the default app, so every existing admin call keeps
    addressing the collection it addressed before tenants existed.

    `?collection=` points the route at one of that app's existing
    `<live>__<suffix>` staging (or kept old) collections instead of the live
    one, e.g. to inspect a rebuild before promoting it.
    """
    try:
        rag_app = get_app(app)
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown app {app!r}",
        ) from None
    if not collection or collection == rag_app.collection:
        return rag_app
    from app.db.chroma import collection_exists

    try:
        name = resolve_collection(rag_app, collection)
    except CollectionAccessError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from None
    if not collection_exists(name):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No collection {name!r}")
    return replace(rag_app, collection=name)
