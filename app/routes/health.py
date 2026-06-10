"""Liveness and readiness endpoints (unauthenticated)."""
from fastapi import APIRouter, Depends, Response, status

from app.config import Settings, get_settings
from app.health import check_ffmpeg, check_redis, check_storage
from app.schemas import ReadyComponents, ReadyResponse

router = APIRouter(tags=["health"])


@router.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


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
