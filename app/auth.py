import secrets
from fastapi import HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import settings

_bearer = HTTPBearer(auto_error=True)


def verify_api_key(
    credentials: HTTPAuthorizationCredentials = Security(_bearer),
) -> None:
    """
    Dependency that validates the Bearer token against RAG_API_KEY.
    Uses constant-time comparison to prevent timing attacks.
    """
    if not secrets.compare_digest(credentials.credentials, settings.rag_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )


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
