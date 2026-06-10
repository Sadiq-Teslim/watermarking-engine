"""Shared test fixtures."""
import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import app

TEST_API_KEY = "test-key-do-not-use-in-prod"


@pytest.fixture
def settings() -> Settings:
    return Settings(
        fpwm_api_key=TEST_API_KEY,
        storage_backend="cloudinary",
        cloudinary_cloud_name="demo",
        cloudinary_api_key="demo",
        cloudinary_api_secret="demo",
    )


@pytest.fixture
def client(settings: Settings) -> TestClient:
    app.dependency_overrides[get_settings] = lambda: settings
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def auth_headers() -> dict:
    return {"Authorization": f"Bearer {TEST_API_KEY}"}
