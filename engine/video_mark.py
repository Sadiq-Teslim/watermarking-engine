"""Video-level orchestration: embed a payload across frames; detect by per-frame decode
(with CRC gate) + multi-frame majority voting, plus best-effort multi-scale resync."""
from dataclasses import dataclass

import cv2
import numpy as np

from engine import ffmpeg_io
from engine.constants import (
    CANONICAL_HEIGHTS,
    CODEWORD_BITS,
    DEFAULT_MARK_STRIDE,
    DEFAULT_Q,
    DETECT_SAMPLE_FPS,
    MIN_CONFIDENCE,
    MIN_VALID_FRAMES,
    NSYM,
)
from engine.ecc import ReedSolomonError, bits_to_bytes, bytes_to_bits, rs_decode, rs_encode
from engine.image_mark import embed_bits_in_frame, extract_bits_from_frame
from engine.payload import decode_message, encode_message


@dataclass
class EmbedResult:
    frames_total: int
    frames_marked: int


@dataclass
class DetectResult:
    marked: bool
    payload: int | None = None
    confidence: float = 0.0
    frames_voted: int = 0


def _codeword_bits(payload_id: int, secret: bytes | None) -> list[int]:
    message = encode_message(payload_id, secret=secret)
    codeword = rs_encode(message, NSYM)
    return bytes_to_bits(codeword)


def embed_video(
    src: str,
    out_path: str,
    payload_id: int,
    secret: bytes | None = None,
    q: float = DEFAULT_Q,
    stride: int = DEFAULT_MARK_STRIDE,
    crf: int = 18,
    preset: str = "medium",
) -> EmbedResult:
    bits = _codeword_bits(payload_id, secret)
    marked = {"n": 0}

    def transform(frame: np.ndarray, idx: int) -> np.ndarray:
        if stride <= 1 or idx % stride == 0:
            marked["n"] += 1
            return embed_bits_in_frame(frame, bits, q)
        return frame

    total = ffmpeg_io.watermark_video(src, out_path, transform, crf=crf, preset=preset)
    return EmbedResult(frames_total=total, frames_marked=marked["n"])


def _decode_frame(frame: np.ndarray, secret: bytes | None, q: float) -> int | None:
    """Return a payload_id only if the CRC/HMAC checksum validates after ECC correction."""
    bits = extract_bits_from_frame(frame, CODEWORD_BITS, q)
    try:
        message = rs_decode(bits_to_bytes(bits), NSYM)
    except ReedSolomonError:
        return None
    ok, payload_id, _version = decode_message(message, secret=secret)
    return payload_id if ok else None


def _decode_frame_multiscale(frame: np.ndarray, secret: bytes | None, q: float) -> int | None:
    # Re-encoding can shift coefficient magnitudes, so probe nearby effective QIM steps.
    q_candidates = [q, q * 0.75, q * (2.0 / 3.0), q * 0.5]
    q_candidates = [qq for qq in q_candidates if qq > 0]

    # Native scale first (the common transcode-only case).
    for qq in q_candidates:
        pid = _decode_frame(frame, secret, qq)
        if pid is not None:
            return pid
    # Best-effort resync: assume the pirate resized to a standard height; try mapping back.
    h = frame.shape[0]
    for target_h in CANONICAL_HEIGHTS:
        if target_h == h:
            continue
        scale = target_h / h
        target_w = int(round(frame.shape[1] * scale))
        if target_w < 16 or target_h < 16:
            continue
        resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        for qq in q_candidates:
            pid = _decode_frame(resized, secret, qq)
            if pid is not None:
                return pid
    return None


def detect_video(
    src: str,
    secret: bytes | None = None,
    q: float = DEFAULT_Q,
    sample_fps: float = DETECT_SAMPLE_FPS,
    multiscale: bool = True,
) -> DetectResult:
    decoder = _decode_frame_multiscale if multiscale else (
        lambda f, s, qq: _decode_frame(f, s, qq)
    )

    votes: dict[int, int] = {}
    frames_seen = 0
    valid = 0
    for frame in ffmpeg_io.iter_frames(src, sample_fps=sample_fps):
        frames_seen += 1
        pid = decoder(frame, secret, q)
        if pid is not None:
            votes[pid] = votes.get(pid, 0) + 1
            valid += 1

    if not votes:
        return DetectResult(marked=False)

    winner = max(votes, key=votes.get)
    confidence = votes[winner] / valid if valid else 0.0
    if votes[winner] >= MIN_VALID_FRAMES and confidence >= MIN_CONFIDENCE:
        return DetectResult(
            marked=True, payload=winner, confidence=round(confidence, 4),
            frames_voted=votes[winner],
        )
    return DetectResult(marked=False, confidence=round(confidence, 4))
