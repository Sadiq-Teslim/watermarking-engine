"""End-to-end video embed/detect (requires ffmpeg; runs in container/CI)."""
import shutil
import subprocess
from pathlib import Path

import pytest

from engine.video_mark import detect_video, embed_video

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def _make_clip(path: str, seconds: int = 3, size: str = "640x480", fps: int = 25) -> None:
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y",
         "-f", "lavfi", "-i", f"testsrc2=size={size}:rate={fps}",
         "-t", str(seconds), "-pix_fmt", "yuv420p", path],
        check=True,
    )


def test_embed_then_detect_recovers_payload(tmp_path: Path):
    src = str(tmp_path / "src.mp4")
    marked = str(tmp_path / "marked.mp4")
    _make_clip(src)

    payload_id = 9_001
    result = embed_video(src, marked, payload_id, crf=16)
    assert result.frames_marked > 0
    assert Path(marked).exists()

    detected = detect_video(marked)
    assert detected.marked is True
    assert detected.payload == payload_id


def test_detect_recovers_with_q_drift(tmp_path: Path):
    src = str(tmp_path / "src-q.mp4")
    marked = str(tmp_path / "marked-q.mp4")
    _make_clip(src)

    payload_id = 9_001
    embed_video(src, marked, payload_id, crf=16, q=12)

    detected = detect_video(marked, q=16)
    assert detected.marked is True
    assert detected.payload == payload_id


def test_unmarked_video_not_detected(tmp_path: Path):
    src = str(tmp_path / "plain.mp4")
    _make_clip(src)
    detected = detect_video(src)
    assert detected.marked is False
