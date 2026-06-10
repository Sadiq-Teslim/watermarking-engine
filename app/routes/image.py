"""Synchronous image watermark + detect endpoints (fast — no job queue needed)."""
from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile

from app.auth import require_api_key
from app.config import Settings, get_settings
from app.schemas import ImageDetectResult
from engine import image_codec
from engine.constants import MAX_PAYLOAD_ID

router = APIRouter(prefix="/v1/image", tags=["image"], dependencies=[Depends(require_api_key)])


def _secret(settings: Settings) -> bytes | None:
    return settings.fpwm_hmac_secret.encode() if settings.fpwm_hmac_secret else None


@router.post("/watermark")
async def watermark_image(
    file: UploadFile = File(...),
    payload: int = Form(...),
    engine: str = Form("trustmark"),
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
    return Response(content=out, media_type="image/png")


@router.post("/detect", response_model=ImageDetectResult)
async def detect_image(
    file: UploadFile = File(...),
    engine: str = Form("trustmark"),
    settings: Settings = Depends(get_settings),
) -> ImageDetectResult:
    data = await file.read()
    try:
        if engine == "qim-dct":
            marked, payload, conf = image_codec.detect_image(data, secret=_secret(settings))
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
