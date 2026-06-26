"""Multi-channel orchestration: authoritative video watermark (Tier 1) + optional
independent audio watermark (P6). Detection reconciles both channels."""
import os
import shutil
import tempfile

from engine import ffmpeg_io, video_mark
from engine.constants import AUDIO_ALPHA, AUDIO_SR, DEFAULT_Q
from engine.video_mark import EmbedResult


def _embed_video_only(src, out_path, payload_id, engine, secret, q, crf, preset) -> EmbedResult:
    if engine == "videoseal":
        from engine import neural_mark
        return neural_mark.embed_video_neural(src, out_path, payload_id, secret=secret, crf=crf)
    return video_mark.embed_video(src, out_path, payload_id, secret=secret, q=q, crf=crf, preset=preset)


def _detect_video_only(src, engine, secret, q):
    if engine == "videoseal":
        from engine import neural_mark
        return neural_mark.detect_video_neural(src, secret=secret)
    return video_mark.detect_video(src, secret=secret, q=q)


def embed(
    src: str,
    out_path: str,
    payload_id: int,
    engine: str = "qim-dct",
    secret: bytes | None = None,
    q: float = DEFAULT_Q,
    crf: int = 18,
    preset: str = "medium",
    audio: bool = False,
    audio_alpha: float = AUDIO_ALPHA,
) -> EmbedResult:
    if not audio:
        return _embed_video_only(src, out_path, payload_id, engine, secret, q, crf, preset)

    with tempfile.TemporaryDirectory() as tmp:
        marked_video = os.path.join(tmp, "v.mp4")
        result = _embed_video_only(src, marked_video, payload_id, engine, secret, q, crf, preset)

        in_wav = os.path.join(tmp, "a.wav")
        wm_wav = os.path.join(tmp, "a_wm.wav")
        if not ffmpeg_io.extract_audio_wav(marked_video, in_wav, AUDIO_SR):
            shutil.copy(marked_video, out_path)  # no audio track to mark
            return result

        from engine.audio_mark import embed_audio_file
        embed_audio_file(in_wav, wm_wav, payload_id, audio_alpha)
        ffmpeg_io.replace_audio(marked_video, wm_wav, out_path)
        return result


def detect(
    src: str,
    engine: str = "qim-dct",
    secret: bytes | None = None,
    q: float = DEFAULT_Q,
    audio: bool = False,
) -> dict:
    video = _detect_video_only(src, engine, secret, q)
    out = {
        "marked": video.marked,
        "payload": video.payload,
        "confidence": video.confidence,
        "frames_voted": video.frames_voted,
        "audio_detected": False,
        "audio_short": None,
        "audio_probability": 0.0,
        "audio_corroborated": False,
    }
    if not audio:
        return out

    with tempfile.TemporaryDirectory() as tmp:
        wav = os.path.join(tmp, "a.wav")
        try:
            if ffmpeg_io.extract_audio_wav(src, wav, AUDIO_SR):
                from engine.audio_mark import detect_audio_file
                detected, short, prob = detect_audio_file(wav)
                out["audio_detected"] = detected
                out["audio_short"] = short
                out["audio_probability"] = round(prob, 4)
        except Exception:
            pass  # audio channel is corroboration only; never fail the whole detect

    out["marked"] = bool(out["marked"] or out["audio_detected"])
    if video.marked and out["audio_detected"] and video.payload is not None:
        out["audio_corroborated"] = (video.payload & 0xFFFF) == out["audio_short"]
    return out
