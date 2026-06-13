"""RQ tasks: the real embed/detect pipelines executed by the worker."""
import hashlib
import hmac
import json
import os
import tempfile

import httpx

from app import storage
from app.config import get_settings
from engine import channels, ffmpeg_io, image_codec, metrics
from engine.constants import DEFAULT_Q, MAX_PAYLOAD_ID


def _secret() -> bytes | None:
    s = get_settings().fpwm_hmac_secret
    return s.encode() if s else None


def _guard_source(source_url: str) -> ffmpeg_io.VideoInfo:
    settings = get_settings()
    info = ffmpeg_io.probe(source_url)
    if settings.max_duration_s and info.duration > settings.max_duration_s:
        raise ValueError(
            f"source duration {info.duration:.0f}s exceeds max {settings.max_duration_s}s"
        )
    return info


def _post_callback(callback_url: str, body: dict) -> None:
    settings = get_settings()
    payload = json.dumps(body).encode()
    headers = {"Content-Type": "application/json"}
    if settings.fpwm_hmac_secret:
        sig = hmac.new(settings.fpwm_hmac_secret.encode(), payload, hashlib.sha256).hexdigest()
        headers["X-FPWM-Signature"] = f"sha256={sig}"
    try:
        httpx.post(callback_url, content=payload, headers=headers, timeout=30)
    except Exception:
        pass  # callback is best-effort; status is always pollable


def embed_video_task(
    source_url: str,
    payload: int,
    max_payload: int = 1_000_000,
    engine: str = "qim-dct",
    strength: float | None = None,
    callback_url: str | None = None,
) -> dict:
    if payload > max_payload:
        raise ValueError("payload exceeds max_payload")
    settings = get_settings()
    _guard_source(source_url)
    q = strength if strength else DEFAULT_Q

    with tempfile.TemporaryDirectory() as tmp:
        out_path = os.path.join(tmp, "marked.mp4")
        embed = channels.embed(
            source_url, out_path, payload, engine=engine, secret=_secret(), q=q,
            audio=settings.audio_watermark_enabled, audio_alpha=settings.audio_alpha,
        )
        psnr, ssim = metrics.quality_psnr_ssim(source_url, out_path)
        vmaf = metrics.quality_vmaf(source_url, out_path)
        watermarked_url = storage.upload_video(settings, out_path)

    result = {
        "watermarked_url": watermarked_url,
        "metrics": {
            "psnr": round(psnr, 2),
            "ssim": round(ssim, 4),
            "vmaf": round(vmaf, 2) if vmaf is not None else None,
            "frames_marked": embed.frames_marked,
            "frames_total": embed.frames_total,
        },
    }
    if callback_url:
        _post_callback(callback_url, {"event": "watermark.completed", **result})
    return result


def detect_video_task(
    source_url: str, engine: str = "qim-dct", callback_url: str | None = None
) -> dict:
    settings = get_settings()
    _guard_source(source_url)
    body = channels.detect(
        source_url, engine=engine, secret=_secret(), audio=settings.audio_watermark_enabled
    )
    if callback_url:
        _post_callback(callback_url, {"event": "detect.completed", **body})
    return body


def embed_image_task(
    data: bytes,
    filename: str,
    payload: int,
    engine: str = "qim-dct",
) -> dict:
    if not (1 <= payload <= MAX_PAYLOAD_ID):
        raise ValueError(f"payload must be in [1,{MAX_PAYLOAD_ID}]")
    settings = get_settings()
    if engine == "qim-dct":
        out = image_codec.embed_image(data, payload, secret=_secret())
    elif engine == "trustmark":
        from engine import image_neural

        if not image_neural.is_available():
            raise RuntimeError("trustmark is not enabled or not installed")
        out = image_neural.embed_image(data, payload)
    else:
        raise ValueError(f"unknown engine: {engine}")

    uploaded = storage.upload_image_bytes(settings, out, folder_suffix="images")
    return {
        "watermarked_url": uploaded["url"],
        "watermarked_public_id": uploaded["public_id"],
        "width": uploaded["width"],
        "height": uploaded["height"],
        "engine": engine,
        "filename": filename,
    }


def detect_image_task(
    data: bytes,
    filename: str,
    engine: str = "qim-dct",
    candidate_sizes: list[tuple[int, int]] | None = None,
) -> dict:
    settings = get_settings()
    if engine == "qim-dct":
        marked, payload, confidence = image_codec.detect_image(
            data, secret=_secret(), candidate_sizes=candidate_sizes
        )
    elif engine == "trustmark":
        from engine import image_neural

        if not image_neural.is_available():
            raise RuntimeError("trustmark is not enabled or not installed")
        marked, payload, confidence = image_neural.detect_image(data)
    else:
        raise ValueError(f"unknown engine: {engine}")
    return {
        "marked": marked,
        "payload": payload,
        "confidence": confidence,
        "engine": engine,
        "filename": filename,
    }
