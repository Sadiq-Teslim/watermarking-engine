"""Benchmark corpus: synthetic clips generated with ffmpeg (no external assets needed),
plus any real sample clips dropped into bench/corpus/real/."""
import subprocess
from pathlib import Path

CORPUS_DIR = Path(__file__).parent / "corpus"
REAL_DIR = CORPUS_DIR / "real"

# Varied lavfi sources: detail, motion, gradients, bars — different watermark conditions.
_SOURCES = [
    "testsrc2=size=1280x720:rate=25",
    "mandelbrot=size=1280x720:rate=25",
    "smptebars=size=1280x720:rate=25",
    "rgbtestsrc=size=854x480:rate=25",
    "testsrc2=size=1920x1080:rate=25",
]


def _gen(source: str, out: Path, seconds: int) -> None:
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", source,
         "-t", str(seconds), "-pix_fmt", "yuv420p", str(out)],
        check=True,
    )


def generate_synthetic(seconds: int = 4) -> list[Path]:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i, source in enumerate(_SOURCES):
        out = CORPUS_DIR / f"synthetic_{i}.mp4"
        if not out.exists():
            _gen(source, out, seconds)
        paths.append(out)
    return paths


def load_real() -> list[Path]:
    if not REAL_DIR.exists():
        return []
    return sorted(p for p in REAL_DIR.iterdir() if p.suffix.lower() in {".mp4", ".mov", ".mkv"})


def all_clips(seconds: int = 4) -> list[Path]:
    return generate_synthetic(seconds) + load_real()
