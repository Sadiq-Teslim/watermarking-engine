"""Health/readiness and root endpoint tests."""
from fastapi.testclient import TestClient


def test_healthz_ok(client: TestClient) -> None:
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_healthz_reports_render_commit(client: TestClient, monkeypatch) -> None:
    monkeypatch.setenv("RENDER_GIT_COMMIT", "1234567890abcdef")
    resp = client.get("/healthz")
    assert resp.json() == {"status": "ok", "commit": "1234567890ab"}


def test_authz_requires_and_accepts_api_key(client: TestClient, auth_headers: dict) -> None:
    assert client.get("/authz").status_code == 401
    resp = client.get("/authz", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "authenticated": True}


def test_root(client: TestClient) -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert body["service"] == "fpwm"
    assert "version" in body


def test_readyz_reports_components(client: TestClient) -> None:
    resp = client.get("/readyz")
    # 200 when all real checks pass (in-container), 503 when a dependency is down.
    assert resp.status_code in (200, 503)
    body = resp.json()
    assert set(body["components"].keys()) == {"redis", "ffmpeg", "storage"}
    assert isinstance(body["components"]["ffmpeg"], bool)
    assert body["status"] in ("ok", "degraded")
