"""Watermark detection endpoints (async job model)."""
from fastapi import APIRouter, Depends, HTTPException, status

from app import jobs, security
from app.auth import require_api_key
from app.config import Settings, get_settings
from app.schemas import DetectRequest, DetectResult, DetectJobStatus, JobAccepted

router = APIRouter(prefix="/v1/detect", tags=["detect"], dependencies=[Depends(require_api_key)])


@router.post("/video", status_code=status.HTTP_202_ACCEPTED, response_model=JobAccepted)
def create_detect(req: DetectRequest, settings: Settings = Depends(get_settings)) -> JobAccepted:
    security.validate_source_url(req.source_url)
    job_id = jobs.enqueue(
        settings,
        "worker.tasks.detect_video_task",
        {"source_url": req.source_url, "engine": req.engine},
    )
    return JobAccepted(job_id=job_id)


@router.get("/jobs/{job_id}", response_model=DetectJobStatus)
def detect_status(job_id: str, settings: Settings = Depends(get_settings)) -> DetectJobStatus:
    job = jobs.fetch(settings, job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="job not found")
    state, result, error = jobs.status_of(job)
    detect_result = DetectResult(**result) if (state == "ready" and result) else None
    return DetectJobStatus(job_id=job_id, status=state, result=detect_result, error=error)
