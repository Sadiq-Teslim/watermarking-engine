"""DCT-QIM image watermark round-trip tests (no torch/ffmpeg required)."""
import cv2
import numpy as np

from engine.image_codec import detect_image, embed_image


def _png_bytes(h=512, w=512) -> bytes:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = np.sin(xx / 23.0) * 40 + np.cos(yy / 19.0) * 40 + 128
    frame = np.stack([base, np.roll(base, 7, axis=0), np.roll(base, 13, axis=1)], axis=2)
    ok, buf = cv2.imencode(".png", np.clip(frame, 0, 255).astype(np.uint8))
    assert ok
    return buf.tobytes()


def test_image_qim_roundtrip():
    payload_id = 1_234_567
    marked = embed_image(_png_bytes(), payload_id, q=12.0)
    found, got, conf = detect_image(marked, q=12.0)
    assert found is True
    assert got == payload_id
    assert conf == 1.0


def test_image_survives_jpeg():
    payload_id = 222_333
    marked_png = embed_image(_png_bytes(), payload_id, q=12.0)
    arr = cv2.imdecode(np.frombuffer(marked_png, np.uint8), cv2.IMREAD_COLOR)
    ok, jpg = cv2.imencode(".jpg", arr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    assert ok
    found, got, _ = detect_image(jpg.tobytes(), q=12.0)
    assert found is True
    assert got == payload_id


def test_unmarked_image_not_detected():
    found, got, _ = detect_image(_png_bytes(), q=12.0)
    assert found is False
    assert got is None
