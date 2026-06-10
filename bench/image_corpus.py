"""Synthetic image corpus (no external assets) + any real images in bench/images/."""
from pathlib import Path

import cv2
import numpy as np

IMAGES_DIR = Path(__file__).parent / "images"


def _png(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("png encode failed")
    return buf.tobytes()


def _gradient(h: int, w: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = (
        np.sin(xx / (12 + seed)) * 50
        + np.cos(yy / (17 + seed)) * 50
        + rng.normal(0, 6, size=(h, w))
        + 128
    )
    img = np.stack([base, np.roll(base, 9, 0), np.roll(base, 15, 1)], axis=2)
    return np.clip(img, 0, 255).astype(np.uint8)


def generate_synthetic(count: int = 5) -> list[bytes]:
    sizes = [(512, 512), (720, 480), (1080, 720), (480, 854), (640, 640)]
    return [_png(_gradient(*sizes[i % len(sizes)], seed=i)) for i in range(count)]


def load_real() -> list[bytes]:
    if not IMAGES_DIR.exists():
        return []
    out = []
    for p in sorted(IMAGES_DIR.iterdir()):
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
            out.append(p.read_bytes())
    return out


def all_images(count: int = 5) -> list[bytes]:
    return generate_synthetic(count) + load_real()
