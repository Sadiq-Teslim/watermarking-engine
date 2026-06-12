"""DCT-QIM image watermarking (fast, CPU). Reuses the Tier-1 frame engine for stills.

Survives JPEG recompression down to ~q35 unhinted (COEF/Q tuned via bench/tune_qim.py).
With `candidate_sizes` hints (the original dimensions, which the product registry always
knows), it also survives resize + social/screenshot re-encodes: the detector rescales the
suspect back onto the original block grid before extracting. Crop/rotation/unknown-size
cases are the neural (TrustMark) tier's job.
"""
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


def _try_decode(arr: np.ndarray, secret: bytes | None, q: float) -> int | None:
    # Recompression shifts effective coefficient magnitudes; probe a small q ladder.
    # Wrong steps can't false-positive — they fail the CRC gate and are discarded.
    for qq in (q, q * 0.75, q * 0.5):
        bits = extract_bits_from_frame(arr, CODEWORD_BITS, qq)
        try:
            msg = rs_decode(bits_to_bytes(bits), NSYM)
        except ReedSolomonError:
            continue
        ok, payload_id, _ = decode_message(msg, secret=secret)
        if ok:
            return payload_id
    return None


def detect_image(
    data: bytes,
    secret: bytes | None = None,
    q: float = DEFAULT_Q,
    candidate_sizes: list[tuple[int, int]] | None = None,
) -> tuple[bool, int | None, float]:
    """Detect a payload. `candidate_sizes` are (width, height) hints — the original
    dimensions of registered images — used to undo platform resizes by mapping the
    suspect back onto the embedding block grid."""
    img = _decode(data)

    pid = _try_decode(img, secret, q)
    if pid is not None:
        return (True, pid, 1.0)

    h, w = img.shape[:2]
    for cw, ch in candidate_sizes or []:
        if cw < 16 or ch < 16 or (cw == w and ch == h):
            continue
        restored = cv2.resize(img, (int(cw), int(ch)), interpolation=cv2.INTER_CUBIC)
        pid = _try_decode(restored, secret, q)
        if pid is not None:
            return (True, pid, 1.0)

    return (False, None, 0.0)
