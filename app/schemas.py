"""Request/response models for the FPWM API."""
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class WatermarkRequest(BaseModel):
    source_url: str = Field(..., description="HTTP(S) URL of the source video")
    payload: int = Field(..., ge=1, description="Payload integer to embed (1..max_payload)")
    max_payload: int = Field(default=1_000_000, ge=1)
    engine: Literal["qim-dct", "videoseal"] = "qim-dct"
    strength: Optional[float] = Field(default=None, gt=0)
    callback_url: Optional[str] = None
    idempotency_key: Optional[str] = None

    @field_validator("source_url", "callback_url")
    @classmethod
    def _http_url(cls, v: Optional[str]) -> Optional[str]:
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
    psnr: Optional[float] = None
    ssim: Optional[float] = None
    vmaf: Optional[float] = None
    frames_marked: Optional[int] = None
    duration_s: Optional[float] = None


class JobStatus(BaseModel):
    job_id: str
    status: Literal["processing", "ready", "error"]
    watermarked_url: Optional[str] = None
    error: Optional[str] = None
    metrics: Optional[JobMetrics] = None


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
    payload: Optional[int] = None
    confidence: float = 0.0
    frames_voted: int = 0
    # Audio channel (P6) — corroboration of the authoritative video payload.
    audio_detected: bool = False
    audio_short: Optional[int] = None
    audio_probability: float = 0.0
    audio_corroborated: bool = False


class DetectJobStatus(BaseModel):
    job_id: str
    status: Literal["processing", "ready", "error"]
    result: Optional[DetectResult] = None
    error: Optional[str] = None


class ReadyComponents(BaseModel):
    redis: bool
    ffmpeg: bool
    storage: bool


class ReadyResponse(BaseModel):
    status: Literal["ok", "degraded"]
    components: ReadyComponents
