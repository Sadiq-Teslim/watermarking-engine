"""HTTP layer tests: auth, validation, job lifecycle (Redis + SSRF mocked)."""
import io

import pytest

from app import jobs, security


@pytest.fixture(autouse=True)
def _no_ssrf(monkeypatch):
    monkeypatch.setattr(security, "validate_source_url", lambda *_a, **_k: None)


def test_watermark_requires_auth(client):
    resp = client.post("/v1/watermark/video", json={"source_url": "https://x/v.mp4", "payload": 5})
    assert resp.status_code == 401


def test_watermark_create_returns_job(client, auth_headers, monkeypatch):
    monkeypatch.setattr(jobs, "enqueue", lambda *a, **k: "job-123")
    resp = client.post(
        "/v1/watermark/video",
        headers=auth_headers,
        json={"source_url": "https://cdn/v.mp4", "payload": 42, "max_payload": 1_000_000},
    )
    assert resp.status_code == 202
    assert resp.json() == {"job_id": "job-123", "status": "processing"}


def test_watermark_payload_exceeds_max(client, auth_headers):
    resp = client.post(
        "/v1/watermark/video",
        headers=auth_headers,
        json={"source_url": "https://cdn/v.mp4", "payload": 2_000_000, "max_payload": 1_000_000},
    )
    assert resp.status_code == 422


def test_watermark_payload_zero_rejected_by_schema(client, auth_headers):
    resp = client.post(
        "/v1/watermark/video",
        headers=auth_headers,
        json={"source_url": "https://cdn/v.mp4", "payload": 0},
    )
    assert resp.status_code == 422


def test_watermark_status_not_found(client, auth_headers, monkeypatch):
    monkeypatch.setattr(jobs, "fetch", lambda *a, **k: None)
    resp = client.get("/v1/watermark/jobs/missing", headers=auth_headers)
    assert resp.status_code == 404


def test_watermark_status_ready(client, auth_headers, monkeypatch):
    ready = {
        "watermarked_url": "https://cdn/marked.mp4",
        "metrics": {"psnr": 44.1, "ssim": 0.991, "frames_marked": 75},
    }
    monkeypatch.setattr(jobs, "fetch", lambda *a, **k: object())
    monkeypatch.setattr(jobs, "status_of", lambda job: ("ready", ready, None))
    resp = client.get("/v1/watermark/jobs/job-123", headers=auth_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ready"
    assert body["watermarked_url"] == "https://cdn/marked.mp4"
    assert body["metrics"]["psnr"] == 44.1


def test_detect_create_and_status(client, auth_headers, monkeypatch):
    monkeypatch.setattr(jobs, "enqueue", lambda *a, **k: "det-1")
    resp = client.post(
        "/v1/detect/video", headers=auth_headers, json={"source_url": "https://cdn/suspect.mp4"}
    )
    assert resp.status_code == 202
    assert resp.json()["job_id"] == "det-1"

    detected = {"marked": True, "payload": 42, "confidence": 0.98, "frames_voted": 50}
    monkeypatch.setattr(jobs, "fetch", lambda *a, **k: object())
    monkeypatch.setattr(jobs, "status_of", lambda job: ("ready", detected, None))
    resp2 = client.get("/v1/detect/jobs/det-1", headers=auth_headers)
    assert resp2.status_code == 200
    body = resp2.json()
    assert body["status"] == "ready"
    assert body["result"]["payload"] == 42


def test_image_watermark_job_returns_job(client, auth_headers, monkeypatch):
    monkeypatch.setattr(jobs, "enqueue", lambda *a, **k: "img-1")
    resp = client.post(
        "/v1/image/watermark/jobs",
        headers=auth_headers,
        data={"payload": "42", "engine": "qim-dct"},
        files={"file": ("image.png", io.BytesIO(b"fake-image"), "image/png")},
    )
    assert resp.status_code == 200
    assert resp.json() == {"job_id": "img-1", "status": "processing"}


def test_image_watermark_job_rejects_disabled_trustmark(client, auth_headers, monkeypatch):
    from engine import image_neural

    monkeypatch.setattr(image_neural, "is_available", lambda: False)
    resp = client.post(
        "/v1/image/watermark/jobs",
        headers=auth_headers,
        data={"payload": "42", "engine": "trustmark"},
        files={"file": ("image.png", io.BytesIO(b"fake-image"), "image/png")},
    )
    assert resp.status_code == 409


def test_image_watermark_job_status_ready(client, auth_headers, monkeypatch):
    result = {
        "watermarked_url": "https://cdn/marked.png",
        "watermarked_public_id": "proofmark/marked",
        "width": 512,
        "height": 512,
    }
    monkeypatch.setattr(jobs, "fetch", lambda *a, **k: object())
    monkeypatch.setattr(jobs, "status_of", lambda job: ("ready", result, None))
    resp = client.get("/v1/image/watermark/jobs/img-1", headers=auth_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ready"
    assert body["result"]["watermarked_url"] == "https://cdn/marked.png"
