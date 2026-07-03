"""Liveness and readiness endpoints (unauthenticated)."""
import os

from fastapi import APIRouter, Depends, Response, status

from app.auth import require_api_key
from app.config import Settings, get_settings
from app.health import check_ffmpeg, check_redis, check_storage
from app.schemas import ReadyComponents, ReadyResponse

router = APIRouter(tags=["health"])


@router.get("/healthz")
def healthz() -> dict:
    response = {"status": "ok"}
    commit = os.environ.get("RENDER_GIT_COMMIT", "").strip()
    if commit:
        response["commit"] = commit[:12]
    return response


@router.get("/authz", dependencies=[Depends(require_api_key)])
def authz() -> dict:
    """Lightweight authenticated probe with no codec or model work."""
    return {"status": "ok", "authenticated": True}


@router.get("/readyz", response_model=ReadyResponse)
def readyz(response: Response, settings: Settings = Depends(get_settings)) -> ReadyResponse:
    components = ReadyComponents(
        redis=check_redis(settings),
        ffmpeg=check_ffmpeg(),
        storage=check_storage(settings),
    )
    ready = all(components.model_dump().values())
    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadyResponse(status="ok" if ready else "degraded", components=components)
