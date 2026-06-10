"""ffmpeg attack battery. Each function transforms an input clip into an attacked copy and
returns the output path. These simulate what pirates / re-upload pipelines do to content."""
import subprocess


def _ff(args: list[str]) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


def transcode_h264(src: str, out: str, crf: int = 23) -> str:
    _ff(["-i", src, "-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p", out])
    return out


def transcode_h265(src: str, out: str, crf: int = 28) -> str:
    _ff(["-i", src, "-c:v", "libx265", "-crf", str(crf), "-pix_fmt", "yuv420p", out])
    return out


def bitrate_cap(src: str, out: str, kbps: int = 500) -> str:
    _ff(["-i", src, "-c:v", "libx264", "-b:v", f"{kbps}k", "-pix_fmt", "yuv420p", out])
    return out


def resize_height(src: str, out: str, height: int = 720) -> str:
    _ff(["-i", src, "-vf", f"scale=-2:{height}", "-c:v", "libx264", "-crf", "20", out])
    return out


def crop_pct(src: str, out: str, pct: float = 0.15) -> str:
    keep = 1.0 - pct
    _ff(["-i", src, "-vf", f"crop=iw*{keep}:ih*{keep},scale=iw:ih",
         "-c:v", "libx264", "-crf", "20", out])
    return out


def rotate_deg(src: str, out: str, deg: float = 2.0) -> str:
    rad = deg * 3.14159265 / 180.0
    _ff(["-i", src, "-vf", f"rotate={rad}:fillcolor=black",
         "-c:v", "libx264", "-crf", "20", out])
    return out


def fps_change(src: str, out: str, fps: int = 24) -> str:
    _ff(["-i", src, "-vf", f"fps={fps}", "-c:v", "libx264", "-crf", "20", out])
    return out


def brightness(src: str, out: str, delta: float = 0.1) -> str:
    _ff(["-i", src, "-vf", f"eq=brightness={delta}", "-c:v", "libx264", "-crf", "20", out])
    return out


def add_noise(src: str, out: str, strength: int = 12) -> str:
    _ff(["-i", src, "-vf", f"noise=alls={strength}:allf=t",
         "-c:v", "libx264", "-crf", "20", out])
    return out


# Attacks Tier 1 is designed to pass (same-resolution recompression / photometric / temporal).
TIER1_ATTACKS = {
    "h264_crf18": lambda s, o: transcode_h264(s, o, 18),
    "h264_crf23": lambda s, o: transcode_h264(s, o, 23),
    "h264_crf28": lambda s, o: transcode_h264(s, o, 28),
    "h265_crf28": lambda s, o: transcode_h265(s, o, 28),
    "bitrate_500k": lambda s, o: bitrate_cap(s, o, 500),
    "fps_24": lambda s, o: fps_change(s, o, 24),
    "brightness": lambda s, o: brightness(s, o, 0.08),
    "noise": lambda s, o: add_noise(s, o, 10),
}

# Geometric attacks: measured for Tier 1 (best-effort multi-scale), gated at Tier 2 (neural).
GEOMETRIC_ATTACKS = {
    "resize_720": lambda s, o: resize_height(s, o, 720),
    "resize_480": lambda s, o: resize_height(s, o, 480),
    "crop_15": lambda s, o: crop_pct(s, o, 0.15),
    "rotate_2deg": lambda s, o: rotate_deg(s, o, 2.0),
}
