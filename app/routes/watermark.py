"""Watermark embed endpoints (async job model)."""
from fastapi import APIRouter, Depends, HTTPException, status

from app import jobs, security
from app.auth import require_api_key
from app.config import Settings, get_settings
from app.schemas import JobAccepted, JobMetrics, JobStatus, WatermarkRequest

router = APIRouter(
    prefix="/v1/watermark", tags=["watermark"], dependencies=[Depends(require_api_key)]
)


@router.post("/video", status_code=status.HTTP_202_ACCEPTED, response_model=JobAccepted)
def create_watermark(
    req: WatermarkRequest, settings: Settings = Depends(get_settings)
) -> JobAccepted:
    try:
        req.check_payload_bounds()
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc

    security.validate_source_url(req.source_url)
    if req.callback_url:
        security.validate_source_url(req.callback_url)
    if not settings.storage_configured():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="storage not configured"
        )

    job_id = jobs.enqueue(
        settings,
        "worker.tasks.embed_video_task",
        {
            "source_url": req.source_url,
            "payload": req.payload,
            "max_payload": req.max_payload,
            "engine": req.engine,
            "strength": req.strength,
            "callback_url": req.callback_url,
        },
        idempotency_key=req.idempotency_key,
    )
    return JobAccepted(job_id=job_id)


@router.get("/jobs/{job_id}", response_model=JobStatus)
def watermark_status(job_id: str, settings: Settings = Depends(get_settings)) -> JobStatus:
    job = jobs.fetch(settings, job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="job not found")
    state, result, error = jobs.status_of(job)
    metrics = None
    watermarked_url = None
    if state == "ready" and result:
        watermarked_url = result.get("watermarked_url")
        metrics = JobMetrics(**result.get("metrics", {}))
    return JobStatus(
        job_id=job_id, status=state, watermarked_url=watermarked_url, error=error, metrics=metrics
    )
