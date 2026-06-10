"""Auth dependency tests (exercised against a protected probe route)."""
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from app.auth import require_api_key
from app.config import Settings, get_settings

TEST_API_KEY = "test-key-do-not-use-in-prod"


def _build_app(api_key: str) -> FastAPI:
    probe = FastAPI()

    @probe.get("/protected", dependencies=[Depends(require_api_key)])
    def protected() -> dict:
        return {"ok": True}

    probe.dependency_overrides[get_settings] = lambda: Settings(fpwm_api_key=api_key)
    return probe


def test_missing_header_rejected() -> None:
    client = TestClient(_build_app(TEST_API_KEY))
    assert client.get("/protected").status_code == 401


def test_wrong_key_rejected() -> None:
    client = TestClient(_build_app(TEST_API_KEY))
    resp = client.get("/protected", headers={"Authorization": "Bearer nope"})
    assert resp.status_code == 401


def test_valid_key_allowed() -> None:
    client = TestClient(_build_app(TEST_API_KEY))
    resp = client.get("/protected", headers={"Authorization": f"Bearer {TEST_API_KEY}"})
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


def test_unconfigured_key_fails_closed() -> None:
    client = TestClient(_build_app(""))
    resp = client.get("/protected", headers={"Authorization": "Bearer anything"})
    assert resp.status_code == 503
