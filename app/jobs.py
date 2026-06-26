"""RQ job layer: enqueue embed/detect tasks, fetch status, idempotency."""
from typing import Any

import redis
from rq import Queue
from rq.exceptions import NoSuchJobError
from rq.job import Job

from app.config import Settings

QUEUE_NAME = "fpwm"
RESULT_TTL = 86_400      # keep results 24h
IDEM_TTL = 86_400
TERMINAL_REQUEUE_STATUSES = {"failed", "stopped", "canceled"}


def _connection(settings: Settings) -> redis.Redis:
    return redis.Redis.from_url(settings.redis_url)


def enqueue(
    settings: Settings,
    func_path: str,
    kwargs: dict[str, Any],
    idempotency_key: str | None = None,
    timeout: int = 3600,
) -> str:
    conn = _connection(settings)
    if idempotency_key:
        idem_key = f"fpwm:idem:{idempotency_key}"
        existing = conn.get(idem_key)
        if existing:
            existing_job_id = existing.decode()
            try:
                existing_job = Job.fetch(existing_job_id, connection=conn)
                existing_status = existing_job.get_status(refresh=True)
                if existing_status not in TERMINAL_REQUEUE_STATUSES:
                    return existing_job_id
            except NoSuchJobError:
                pass
            conn.delete(idem_key)

    queue = Queue(QUEUE_NAME, connection=conn)
    job = queue.enqueue_call(
        func=func_path,
        kwargs=kwargs,
        timeout=timeout,
        result_ttl=RESULT_TTL,
        failure_ttl=RESULT_TTL,
    )
    if idempotency_key:
        conn.set(f"fpwm:idem:{idempotency_key}", job.id, ex=IDEM_TTL)
    return job.id


def fetch(settings: Settings, job_id: str) -> Job | None:
    try:
        return Job.fetch(job_id, connection=_connection(settings))
    except (NoSuchJobError, Exception):
        return None


def status_of(job: Job) -> tuple[str, dict | None, str | None]:
    """Map an RQ job to (status, result, error) where status is processing|ready|error."""
    rq_status = job.get_status(refresh=True)
    if rq_status == "finished":
        return ("ready", job.result, None)
    if rq_status in ("failed", "stopped", "canceled"):
        error_text = job.exc_info or "job failed"
        error = error_text.strip().splitlines()[-1] if job.exc_info else "job failed"
        return ("error", None, error)
    return ("processing", None, None)
