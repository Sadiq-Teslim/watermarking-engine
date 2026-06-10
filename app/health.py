"""Real readiness checks (no stubs): redis ping, ffmpeg presence, storage config."""
import shutil
import subprocess

import redis

from app.config import Settings


def check_redis(settings: Settings) -> bool:
    try:
        client = redis.Redis.from_url(settings.redis_url, socket_connect_timeout=2)
        return bool(client.ping())
    except Exception:
        return False


def check_ffmpeg() -> bool:
    if shutil.which("ffmpeg") is None:
        return False
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            timeout=5,
            check=False,
        )
        return result.returncode == 0
    except Exception:
        return False


def check_storage(settings: Settings) -> bool:
    return settings.storage_configured()
