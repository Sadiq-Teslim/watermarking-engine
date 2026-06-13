"""Synchronous image watermark + detect endpoints (fast — no job queue needed)."""
import json

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile

from app import jobs
from app.auth import require_api_key
from app.config import Settings, get_settings
from app.schemas import ImageDetectResult
from engine import image_codec
from engine.constants import MAX_PAYLOAD_ID

router = APIRouter(prefix="/v1/image", tags=["image"], dependencies=[Depends(require_api_key)])


def _secret(settings: Settings) -> bytes | None:
    return settings.fpwm_hmac_secret.encode() if settings.fpwm_hmac_secret else None


def _dims(data: bytes) -> tuple[int, int] | None:
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        return None
    return (arr.shape[1], arr.shape[0])  # (width, height)


def _parse_sizes(raw: str) -> list[tuple[int, int]] | None:
    """Parse candidate_sizes form field: JSON like [[640,480],[1920,1080]]."""
    if not raw:
        return None
    try:
        sizes = json.loads(raw)
        return [(int(w), int(h)) for w, h in sizes][:50]
    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=422, detail="candidate_sizes must be JSON [[w,h],...]"
        ) from exc


def _image_job_status(settings: Settings, job_id: str) -> dict:
    job = jobs.fetch(settings, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    state, result, error = jobs.status_of(job)
    return {
        "job_id": job_id,
        "status": state,
        "result": result if state == "ready" else None,
        "error": error,
    }


@router.get("/capabilities")
async def image_capabilities() -> dict:
    from engine import image_neural

    return {
        "default_engine": "qim-dct",
        "engines": {
            "qim-dct": {
                "available": True,
                "tier": "standard",
                "survives": [
                    "jpeg",
                    "resize-with-size-hints",
                    "screenshots",
                    "social-reencode-with-size-hints",
                ],
            },
            "trustmark": {
                "available": image_neural.is_available(),
                "tier": "strong",
                "survives": ["crop", "rotation", "screenshots", "unknown-resize"],
            },
        },
    }


@router.post("/watermark")
async def watermark_image(
    file: UploadFile = File(...),
    payload: int = Form(...),
    engine: str = Form("qim-dct"),
    settings: Settings = Depends(get_settings),
) -> Response:
    if not (1 <= payload <= MAX_PAYLOAD_ID):
        raise HTTPException(status_code=422, detail=f"payload must be in [1,{MAX_PAYLOAD_ID}]")
    data = await file.read()
    try:
        if engine == "qim-dct":
            out = image_codec.embed_image(data, payload, secret=_secret(settings))
        elif engine == "trustmark":
            from engine import image_neural
            out = image_neural.embed_image(data, payload)
        else:
            raise HTTPException(status_code=422, detail=f"unknown engine: {engine}")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"watermark failed: {exc}") from exc

    # Original dimensions in headers so callers can store them and pass size hints later.
    headers = {}
    dims = _dims(data)
    if dims:
        headers["X-Original-Width"] = str(dims[0])
        headers["X-Original-Height"] = str(dims[1])
    return Response(content=out, media_type="image/png", headers=headers)


@router.post("/watermark/jobs")
async def create_watermark_image_job(
    file: UploadFile = File(...),
    payload: int = Form(...),
    engine: str = Form("qim-dct"),
    settings: Settings = Depends(get_settings),
) -> dict:
    if not (1 <= payload <= MAX_PAYLOAD_ID):
        raise HTTPException(status_code=422, detail=f"payload must be in [1,{MAX_PAYLOAD_ID}]")
    if engine not in {"qim-dct", "trustmark"}:
        raise HTTPException(status_code=422, detail=f"unknown engine: {engine}")
    if engine == "trustmark":
        from engine import image_neural

        if not image_neural.is_available():
            raise HTTPException(status_code=409, detail="trustmark is not enabled or not installed")

    data = await file.read()
    job_id = jobs.enqueue(
        settings,
        "worker.tasks.embed_image_task",
        {
            "data": data,
            "filename": file.filename or "image",
            "payload": payload,
            "engine": engine,
        },
        timeout=1800,
    )
    return {"job_id": job_id, "status": "processing"}


@router.get("/watermark/jobs/{job_id}")
async def watermark_image_job_status(
    job_id: str,
    settings: Settings = Depends(get_settings),
) -> dict:
    return _image_job_status(settings, job_id)


@router.post("/detect", response_model=ImageDetectResult)
async def detect_image(
    file: UploadFile = File(...),
    engine: str = Form("qim-dct"),
    candidate_sizes: str = Form(""),
    settings: Settings = Depends(get_settings),
) -> ImageDetectResult:
    data = await file.read()
    sizes = _parse_sizes(candidate_sizes)
    try:
        if engine == "qim-dct":
            marked, payload, conf = image_codec.detect_image(
                data, secret=_secret(settings), candidate_sizes=sizes
            )
        elif engine == "trustmark":
            from engine import image_neural
            marked, payload, conf = image_neural.detect_image(data)
        else:
            raise HTTPException(status_code=422, detail=f"unknown engine: {engine}")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"detect failed: {exc}") from exc
    return ImageDetectResult(marked=marked, payload=payload, confidence=conf, engine=engine)


@router.post("/detect/jobs")
async def create_detect_image_job(
    file: UploadFile = File(...),
    engine: str = Form("qim-dct"),
    candidate_sizes: str = Form(""),
    settings: Settings = Depends(get_settings),
) -> dict:
    if engine not in {"qim-dct", "trustmark"}:
        raise HTTPException(status_code=422, detail=f"unknown engine: {engine}")
    if engine == "trustmark":
        from engine import image_neural

        if not image_neural.is_available():
            raise HTTPException(status_code=409, detail="trustmark is not enabled or not installed")

    data = await file.read()
    job_id = jobs.enqueue(
        settings,
        "worker.tasks.detect_image_task",
        {
            "data": data,
            "filename": file.filename or "image",
            "engine": engine,
            "candidate_sizes": _parse_sizes(candidate_sizes),
        },
        timeout=1800,
    )
    return {"job_id": job_id, "status": "processing"}


@router.get("/detect/jobs/{job_id}")
async def detect_image_job_status(
    job_id: str,
    settings: Settings = Depends(get_settings),
) -> dict:
    return _image_job_status(settings, job_id)
