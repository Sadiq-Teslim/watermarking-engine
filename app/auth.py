"""Bearer API-key authentication."""
import hmac

from fastapi import Depends, Header, HTTPException, status

from app.config import Settings, get_settings


def require_api_key(
    authorization: str = Header(default=""),
    settings: Settings = Depends(get_settings),
) -> None:
    """Reject any request without a valid `Authorization: Bearer <FPWM_API_KEY>` header.

    Fails closed: if no API key is configured on the server, all requests are rejected.
    """
    if not settings.fpwm_api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server API key not configured",
        )

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or malformed Authorization header",
        )

    # Constant-time comparison to avoid timing leaks.
    if not hmac.compare_digest(token, settings.fpwm_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
