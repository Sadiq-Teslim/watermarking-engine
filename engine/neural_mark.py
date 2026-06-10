"""Video neural tier via Meta VideoSeal (opt-in). Trained for robustness to geometric and
photometric attacks (resize, crop, rotation, screen-record) that Tier-1 DCT-QIM can't cover.

Reuses the SAME payload/ECC/CRC/voting machinery as Tier-1, so the payload registry is
unchanged: VideoSeal carries our codeword bits per frame; detection RS-decodes + CRC-gates +
majority-votes across sampled frames. ECC parity auto-scales to the model's bit capacity.

NOTE: VideoSeal's exact embed/detect signatures should be validated against the installed
package at deploy (covered by the benchmark with INSTALL_NEURAL=true and FPWM_BENCH_ENGINE=videoseal).
"""
import cv2
import numpy as np

from engine import ffmpeg_io
from engine.constants import (
    DEFAULT_MARK_STRIDE,
    DETECT_SAMPLE_FPS,
    MESSAGE_BYTES,
    MIN_CONFIDENCE,
    MIN_VALID_FRAMES,
    NEURAL_DETECT_THRESHOLD,
)
from engine.ecc import ReedSolomonError, bits_to_bytes, bytes_to_bits, rs_decode, rs_encode
from engine.payload import decode_message, encode_message
from engine.video_mark import DetectResult, EmbedResult

_model = None


def _load():
    global _model
    if _model is None:
        import videoseal  # deferred heavy import
        model = videoseal.load("videoseal")
        model.eval()
        _model = model
    return _model


def neural_nbits() -> int:
    return int(getattr(_load(), "nbits"))


def _nsym_for(nbits: int) -> tuple[int, int]:
    codeword_bytes = nbits // 8
    nsym = codeword_bytes - MESSAGE_BYTES
    if nsym < 0:
        raise ValueError(f"VideoSeal nbits={nbits} too small for a {MESSAGE_BYTES}-byte message")
    return codeword_bytes, nsym


def _frame_to_tensor(frame_bgr: np.ndarray):
    import torch
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]


def _tensor_to_frame(tensor) -> np.ndarray:
    arr = tensor.squeeze(0).permute(1, 2, 0).clamp(0, 1).detach().cpu().numpy()
    return cv2.cvtColor((arr * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR)


def _codeword_bits(payload_id: int, secret: bytes | None, nbits: int) -> list[int]:
    _, nsym = _nsym_for(nbits)
    message = encode_message(payload_id, secret=secret)
    return bytes_to_bits(rs_encode(message, nsym))


def embed_video_neural(
    src: str,
    out_path: str,
    payload_id: int,
    secret: bytes | None = None,
    stride: int = DEFAULT_MARK_STRIDE,
    crf: int = 18,
) -> EmbedResult:
    import torch

    model = _load()
    nbits = neural_nbits()
    bits = _codeword_bits(payload_id, secret, nbits)
    msg = torch.tensor(bits, dtype=torch.float32).unsqueeze(0)  # [1, nbits]
    marked = {"n": 0}

    def transform(frame: np.ndarray, idx: int) -> np.ndarray:
        if stride > 1 and idx % stride != 0:
            return frame
        with torch.no_grad():
            out = model.embed(_frame_to_tensor(frame), msgs=msg, is_video=False)["imgs_w"]
        marked["n"] += 1
        return _tensor_to_frame(out)

    total = ffmpeg_io.watermark_video(src, out_path, transform, crf=crf)
    return EmbedResult(frames_total=total, frames_marked=marked["n"])


def _decode_frame_neural(frame: np.ndarray, secret: bytes | None, nbits: int, nsym: int) -> int | None:
    import torch

    model = _load()
    with torch.no_grad():
        preds = model.detect(_frame_to_tensor(frame), is_video=False)["preds"][0]
    detection = float(torch.sigmoid(preds[0]))
    if detection < NEURAL_DETECT_THRESHOLD:
        return None
    bits = (torch.sigmoid(preds[1:1 + nbits]) > 0.5).int().tolist()
    try:
        message = rs_decode(bits_to_bytes(bits), nsym)
    except ReedSolomonError:
        return None
    ok, payload_id, _ = decode_message(message, secret=secret)
    return payload_id if ok else None


def detect_video_neural(
    src: str,
    secret: bytes | None = None,
    sample_fps: float = DETECT_SAMPLE_FPS,
) -> DetectResult:
    nbits = neural_nbits()
    _, nsym = _nsym_for(nbits)
    votes: dict[int, int] = {}
    valid = 0
    for frame in ffmpeg_io.iter_frames(src, sample_fps=sample_fps):
        pid = _decode_frame_neural(frame, secret, nbits, nsym)
        if pid is not None:
            votes[pid] = votes.get(pid, 0) + 1
            valid += 1

    if not votes:
        return DetectResult(marked=False)
    winner = max(votes, key=votes.get)
    confidence = votes[winner] / valid if valid else 0.0
    if votes[winner] >= MIN_VALID_FRAMES and confidence >= MIN_CONFIDENCE:
        return DetectResult(
            marked=True, payload=winner, confidence=round(confidence, 4), frames_voted=votes[winner]
        )
    return DetectResult(marked=False, confidence=round(confidence, 4))
