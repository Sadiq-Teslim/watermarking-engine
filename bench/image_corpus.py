"""Synthetic image corpus (no external assets) + any real images in bench/images/.

Variety matters: odd (non-multiple-of-8) dimensions exercise block-grid edge handling,
portrait orientations exercise the resize-back path, and different texture classes
(smooth gradients, fine detail, flat regions) span easy-to-hard embedding conditions.
"""
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
    """Smooth waves + mild noise (photo-like low-frequency content)."""
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


def _detail(h: int, w: int, seed: int) -> np.ndarray:
    """High-frequency texture (foliage/fabric-like) — hard case for imperceptibility."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(128, 40, size=(h, w)).astype(np.float32)
    blurred = cv2.GaussianBlur(noise, (0, 0), 1.2)
    img = np.stack([blurred, cv2.GaussianBlur(noise, (0, 0), 2.0), noise], axis=2)
    return np.clip(img, 0, 255).astype(np.uint8)


def _flat_regions(h: int, w: int, seed: int) -> np.ndarray:
    """Large flat areas + sharp edges (screenshots/graphics-like) — hard case for QIM."""
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 235, dtype=np.float32)
    for _ in range(8):
        x0, y0 = rng.integers(0, w - 20), rng.integers(0, h - 20)
        x1 = int(min(w, x0 + rng.integers(40, w // 2)))
        y1 = int(min(h, y0 + rng.integers(40, h // 2)))
        img[y0:y1, x0:x1] = rng.integers(30, 220, size=3)
    img += rng.normal(0, 2, size=(h, w, 1))
    return np.clip(img, 0, 255).astype(np.uint8)


def generate_synthetic() -> list[bytes]:
    # Mix of landscape/portrait, multiple-of-8 and odd dimensions, three texture classes.
    specs = [
        (_gradient, 512, 512),
        (_gradient, 480, 720),     # landscape
        (_gradient, 854, 480),     # portrait
        (_gradient, 761, 1013),    # odd dims (not multiples of 8)
        (_detail, 720, 1080),
        (_detail, 600, 450),       # portrait, odd-ish
        (_flat_regions, 768, 1024),
        (_flat_regions, 1080, 607),  # tall portrait, odd width
    ]
    return [_png(fn(h, w, seed=i)) for i, (fn, h, w) in enumerate(specs)]


def load_real() -> list[bytes]:
    if not IMAGES_DIR.exists():
        return []
    out = []
    for p in sorted(IMAGES_DIR.iterdir()):
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
            out.append(p.read_bytes())
    return out


def all_images() -> list[bytes]:
    return generate_synthetic() + load_real()
