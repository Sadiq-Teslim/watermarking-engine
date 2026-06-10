"""Image attack battery (operates on encoded image bytes -> attacked bytes)."""
import cv2
import numpy as np


def _dec(data: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def _png(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("png encode failed")
    return buf.tobytes()


def _jpg(img: np.ndarray, q: int) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return buf.tobytes()


def jpeg(data: bytes, q: int) -> bytes:
    return _jpg(_dec(data), q)


def resize(data: bytes, scale: float) -> bytes:
    img = _dec(data)
    h, w = img.shape[:2]
    dim = (max(8, int(w * scale)), max(8, int(h * scale)))
    return _png(cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC))


def crop(data: bytes, pct: float) -> bytes:
    img = _dec(data)
    h, w = img.shape[:2]
    ch, cw = int(h * pct / 2), int(w * pct / 2)
    return _png(img[ch:h - ch, cw:w - cw])


def rotate(data: bytes, deg: float) -> bytes:
    img = _dec(data)
    h, w = img.shape[:2]
    mat = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    return _png(cv2.warpAffine(img, mat, (w, h), borderMode=cv2.BORDER_REFLECT))


def brightness(data: bytes, delta: int) -> bytes:
    img = _dec(data).astype(np.int16) + delta
    return _png(np.clip(img, 0, 255).astype(np.uint8))


def screenshot_sim(data: bytes) -> bytes:
    # Approximate a screenshot: mild downscale + JPEG.
    return jpeg(resize(data, 0.85), 80)


def social_sim(data: bytes) -> bytes:
    # Approximate social-platform re-encode: heavier JPEG + slight downscale.
    return jpeg(resize(data, 0.9), 60)


# Tier-1 (qim-dct) is gated on these (recompression survives block-DCT).
RECOMPRESSION_ATTACKS = {
    "jpeg_q90": lambda b: jpeg(b, 90),
    "jpeg_q75": lambda b: jpeg(b, 75),
    "jpeg_q50": lambda b: jpeg(b, 50),
    "brightness": lambda b: brightness(b, 12),
}

# Geometric/screenshot attacks: measured for qim-dct, gated for the neural (trustmark) tier.
GEOMETRIC_ATTACKS = {
    "resize_50": lambda b: resize(b, 0.5),
    "resize_150": lambda b: resize(b, 1.5),
    "crop_10": lambda b: crop(b, 0.10),
    "crop_25": lambda b: crop(b, 0.25),
    "rotate_3deg": lambda b: rotate(b, 3.0),
    "screenshot": screenshot_sim,
    "social": social_sim,
}
