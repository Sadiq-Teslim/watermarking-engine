"""Request/response models for the FPWM API."""
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class WatermarkRequest(BaseModel):
    source_url: str = Field(..., description="HTTP(S) URL of the source video")
    payload: int = Field(..., ge=1, description="Payload integer to embed (1..max_payload)")
    max_payload: int = Field(default=1_000_000, ge=1)
    engine: Literal["qim-dct", "videoseal"] = "qim-dct"
    strength: float | None = Field(default=None, gt=0)
    callback_url: str | None = None
    idempotency_key: str | None = None

    @field_validator("source_url", "callback_url")
    @classmethod
    def _http_url(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if not (v.startswith("http://") or v.startswith("https://")):
            raise ValueError("must be an http(s) URL")
        return v

    def check_payload_bounds(self) -> None:
        if self.payload > self.max_payload:
            raise ValueError("payload exceeds max_payload")


class JobAccepted(BaseModel):
    job_id: str
    status: Literal["processing"] = "processing"


class JobMetrics(BaseModel):
    psnr: float | None = None
    ssim: float | None = None
    vmaf: float | None = None
    frames_marked: int | None = None
    duration_s: float | None = None


class JobStatus(BaseModel):
    job_id: str
    status: Literal["processing", "ready", "error"]
    watermarked_url: str | None = None
    error: str | None = None
    metrics: JobMetrics | None = None


class DetectRequest(BaseModel):
    source_url: str
    max_payload: int = Field(default=1_000_000, ge=1)
    engine: Literal["qim-dct", "videoseal"] = "qim-dct"

    @field_validator("source_url")
    @classmethod
    def _http_url(cls, v: str) -> str:
        if not (v.startswith("http://") or v.startswith("https://")):
            raise ValueError("must be an http(s) URL")
        return v


class DetectResult(BaseModel):
    marked: bool
    payload: int | None = None
    confidence: float = 0.0
    frames_voted: int = 0
    # Audio channel (P6) — corroboration of the authoritative video payload.
    audio_detected: bool = False
    audio_short: int | None = None
    audio_probability: float = 0.0
    audio_corroborated: bool = False


class DetectJobStatus(BaseModel):
    job_id: str
    status: Literal["processing", "ready", "error"]
    result: DetectResult | None = None
    error: str | None = None


class ImageDetectResult(BaseModel):
    marked: bool
    payload: int | None = None
    confidence: float = 0.0
    engine: str


class ReadyComponents(BaseModel):
    redis: bool
    ffmpeg: bool
    storage: bool


class ReadyResponse(BaseModel):
    status: Literal["ok", "degraded"]
    components: ReadyComponents
