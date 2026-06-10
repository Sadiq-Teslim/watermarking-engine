"""ffmpeg-backed video I/O: probe, streaming frame-by-frame watermark (audio preserved),
and frame sampling for detection. Operates directly on http(s) URLs or local paths."""
import json
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    duration: float
    has_audio: bool


def _run(cmd: list[str], timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, timeout=timeout, check=False)


def probe(src: str, timeout: int = 60) -> VideoInfo:
    cmd = [
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_streams", "-show_format", src,
    ]
    result = _run(cmd, timeout)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr.decode(errors='ignore')[:500]}")
    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise RuntimeError("no video stream found")
    has_audio = any(s.get("codec_type") == "audio" for s in streams)

    num, _, den = video.get("avg_frame_rate", "0/1").partition("/")
    fps = (float(num) / float(den)) if den and float(den) != 0 else 0.0
    duration = float(data.get("format", {}).get("duration", 0.0) or 0.0)
    return VideoInfo(
        width=int(video["width"]),
        height=int(video["height"]),
        fps=fps or 25.0,
        duration=duration,
        has_audio=has_audio,
    )


def watermark_video(
    src: str,
    out_path: str,
    transform,                       # (frame_bgr: np.ndarray, idx: int) -> np.ndarray
    crf: int = 18,
    preset: str = "medium",
    timeout: int = 3600,
) -> int:
    """Stream frames through `transform`, re-encode H.264, copy original audio. Returns the
    number of frames written. Back-pressure between decoder/encoder bounds memory."""
    info = probe(src)
    w, h = info.width, info.height
    frame_size = w * h * 3

    dec = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", src,
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-vsync", "0", "pipe:1"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )

    enc_cmd = [
        "ffmpeg", "-v", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
        "-framerate", f"{info.fps}", "-i", "pipe:0",
        "-i", src,
        "-map", "0:v:0", "-map", "1:a?",
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p",
        "-c:a", "copy", "-shortest", out_path,
    ]
    enc = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    idx = 0
    try:
        assert dec.stdout is not None and enc.stdin is not None
        while True:
            raw = dec.stdout.read(frame_size)
            if len(raw) < frame_size:
                break
            frame = np.frombuffer(raw, np.uint8).reshape(h, w, 3).copy()
            out = transform(frame, idx)
            enc.stdin.write(np.ascontiguousarray(out, dtype=np.uint8).tobytes())
            idx += 1
    finally:
        if enc.stdin is not None:
            enc.stdin.close()
        if dec.stdout is not None:
            dec.stdout.close()
        dec.wait(timeout=timeout)
        enc_rc = enc.wait(timeout=timeout)
    if enc_rc != 0:
        raise RuntimeError(f"ffmpeg encode failed (rc={enc_rc})")
    return idx


def extract_audio_wav(src: str, out_wav: str, sample_rate: int = 16000, timeout: int = 600) -> bool:
    """Extract a mono WAV at `sample_rate`. Returns False if the source has no audio."""
    if not probe(src).has_audio:
        return False
    result = _run(
        ["ffmpeg", "-v", "error", "-y", "-i", src,
         "-vn", "-ac", "1", "-ar", str(sample_rate), "-f", "wav", out_wav],
        timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(f"audio extract failed: {result.stderr.decode(errors='ignore')[:300]}")
    return True


def replace_audio(video_in: str, wav_in: str, out_path: str, timeout: int = 1800) -> None:
    """Mux `wav_in` as the audio track of `video_in` (video copied, audio re-encoded AAC)."""
    result = _run(
        ["ffmpeg", "-v", "error", "-y", "-i", video_in, "-i", wav_in,
         "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac",
         "-shortest", out_path],
        timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(f"audio mux failed: {result.stderr.decode(errors='ignore')[:300]}")


def iter_frames(src: str, sample_fps: float | None = None, timeout: int = 1800) -> Iterator[np.ndarray]:
    """Yield decoded BGR frames. If sample_fps is set, decode at that rate (frame size is
    unchanged by the fps filter)."""
    info = probe(src)
    w, h = info.width, info.height
    frame_size = w * h * 3

    args = ["ffmpeg", "-v", "error", "-i", src]
    if sample_fps:
        args += ["-vf", f"fps={sample_fps}"]
    args += ["-f", "rawvideo", "-pix_fmt", "bgr24", "-vsync", "0", "pipe:1"]

    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        assert proc.stdout is not None
        while True:
            raw = proc.stdout.read(frame_size)
            if len(raw) < frame_size:
                break
            yield np.frombuffer(raw, np.uint8).reshape(h, w, 3).copy()
    finally:
        if proc.stdout is not None:
            proc.stdout.close()
        proc.wait(timeout=timeout)
