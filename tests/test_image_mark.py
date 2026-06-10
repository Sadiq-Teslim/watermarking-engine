"""Frame-level DCT-QIM embed/extract tests (no ffmpeg required)."""
import cv2
import numpy as np

from engine.constants import CODEWORD_BITS, DEFAULT_Q, NSYM
from engine.ecc import bits_to_bytes, bytes_to_bits, rs_decode, rs_encode
from engine.image_mark import embed_bits_in_frame, extract_bits_from_frame
from engine.payload import decode_message, encode_message


def _natural_frame(h=256, w=256) -> np.ndarray:
    """A smooth gradient + low-freq texture (closer to real content than white noise)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = (np.sin(xx / 23.0) * 40 + np.cos(yy / 19.0) * 40 + 128)
    frame = np.stack([base, np.roll(base, 7, axis=0), np.roll(base, 13, axis=1)], axis=2)
    return np.clip(frame, 0, 255).astype(np.uint8)


def test_clean_bit_roundtrip_exact():
    rng = np.random.default_rng(1)
    bits = rng.integers(0, 2, size=CODEWORD_BITS).tolist()
    frame = _natural_frame()
    marked = embed_bits_in_frame(frame, bits, DEFAULT_Q)
    out = extract_bits_from_frame(marked, CODEWORD_BITS, DEFAULT_Q)
    assert out == bits


def test_full_payload_roundtrip_through_ecc():
    payload_id = 1_234_567
    codeword = rs_encode(encode_message(payload_id), NSYM)
    bits = bytes_to_bits(codeword)
    marked = embed_bits_in_frame(_natural_frame(), bits, DEFAULT_Q)
    out_bits = extract_bits_from_frame(marked, CODEWORD_BITS, DEFAULT_Q)
    msg = rs_decode(bits_to_bytes(out_bits), NSYM)
    ok, got, _ = decode_message(msg)
    assert ok and got == payload_id


def test_survives_jpeg_recompression():
    payload_id = 555_000
    codeword = rs_encode(encode_message(payload_id), NSYM)
    bits = bytes_to_bits(codeword)
    marked = embed_bits_in_frame(_natural_frame(512, 512), bits, DEFAULT_Q)
    ok_enc, buf = cv2.imencode(".jpg", marked, [cv2.IMWRITE_JPEG_QUALITY, 90])
    assert ok_enc
    recompressed = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    out_bits = extract_bits_from_frame(recompressed, CODEWORD_BITS, DEFAULT_Q)
    msg = rs_decode(bits_to_bytes(out_bits), NSYM)
    ok, got, _ = decode_message(msg)
    assert ok and got == payload_id


def test_unmarked_frame_yields_no_valid_payload():
    """A frame that was never marked must NOT decode to a valid (checksum-passing) payload."""
    from engine.ecc import ReedSolomonError
    frame = _natural_frame()
    out_bits = extract_bits_from_frame(frame, CODEWORD_BITS, DEFAULT_Q)
    try:
        msg = rs_decode(bits_to_bytes(out_bits), NSYM)
    except ReedSolomonError:
        return  # ECC rejected -> correctly treated as unmarked
    ok, _, _ = decode_message(msg)
    assert ok is False
