"""DCT-QIM image watermarking (fast, CPU). Reuses the Tier-1 frame engine for stills.
Survives JPEG recompression / resize; use the neural tier for screenshots/social re-encode."""
import cv2
import numpy as np

from engine.constants import CODEWORD_BITS, DEFAULT_Q, NSYM
from engine.ecc import ReedSolomonError, bits_to_bytes, bytes_to_bits, rs_decode, rs_encode
from engine.image_mark import embed_bits_in_frame, extract_bits_from_frame
from engine.payload import decode_message, encode_message


def _decode(data: bytes) -> np.ndarray:
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise ValueError("could not decode image")
    return arr


def embed_image(
    data: bytes, payload_id: int, secret: bytes | None = None, q: float = DEFAULT_Q
) -> bytes:
    img = _decode(data)
    bits = bytes_to_bits(rs_encode(encode_message(payload_id, secret=secret), NSYM))
    marked = embed_bits_in_frame(img, bits, q)
    ok, buf = cv2.imencode(".png", marked)
    if not ok:
        raise RuntimeError("failed to encode watermarked image")
    return buf.tobytes()


def detect_image(
    data: bytes, secret: bytes | None = None, q: float = DEFAULT_Q
) -> tuple[bool, int | None, float]:
    img = _decode(data)
    bits = extract_bits_from_frame(img, CODEWORD_BITS, q)
    try:
        msg = rs_decode(bits_to_bytes(bits), NSYM)
    except ReedSolomonError:
        return (False, None, 0.0)
    ok, payload_id, _ = decode_message(msg, secret=secret)
    return (ok, payload_id if ok else None, 1.0 if ok else 0.0)
