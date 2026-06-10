"""Audio watermark channel via Meta AudioSeal (neural, opt-in).

This is an INDEPENDENT second channel: it carries the low 16 bits of the payload as
corroboration of the authoritative 28-bit video watermark. torch/audioseal are heavy and
only installed when INSTALL_NEURAL=true, so imports are deferred.

NOTE: the exact AudioSeal call signatures should be validated against the installed
package version at deploy (covered by the benchmark harness with INSTALL_NEURAL=true).
"""
import numpy as np
import soundfile as sf

from engine.constants import AUDIO_DETECT_THRESHOLD, AUDIO_NBITS, AUDIO_SR

_generator = None
_detector = None


def _models():
    global _generator, _detector
    if _generator is None or _detector is None:
        from audioseal import AudioSeal  # deferred heavy import
        _generator = AudioSeal.load_generator("audioseal_wm_16bits")
        _detector = AudioSeal.load_detector("audioseal_detector_16bits")
    return _generator, _detector


def payload_to_audio_bits(payload_id: int) -> list[int]:
    """Low AUDIO_NBITS bits of the payload, MSB-first."""
    short = payload_id & ((1 << AUDIO_NBITS) - 1)
    return [(short >> (AUDIO_NBITS - 1 - i)) & 1 for i in range(AUDIO_NBITS)]


def audio_bits_to_short(bits: list[int]) -> int:
    value = 0
    for bit in bits[:AUDIO_NBITS]:
        value = (value << 1) | int(bit)
    return value


def embed_audio_file(in_wav: str, out_wav: str, payload_id: int, alpha: float) -> None:
    import torch

    generator, _ = _models()
    wav, sr = sf.read(in_wav, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)  # mono
    tensor = torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0)  # [1,1,samples]
    bits = payload_to_audio_bits(payload_id)
    message = torch.tensor(bits, dtype=torch.int32).unsqueeze(0)      # [1,16]

    watermarked = generator(tensor, sample_rate=sr, message=message, alpha=alpha)
    out = watermarked.squeeze().detach().cpu().numpy().astype(np.float32)
    sf.write(out_wav, out, sr)


def detect_audio_file(wav_path: str) -> tuple[bool, int | None, float]:
    """Return (detected, short_payload, probability)."""
    import torch

    _, detector = _models()
    wav, sr = sf.read(wav_path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    tensor = torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0)

    prob, message = detector.detect_watermark(tensor, sample_rate=sr)
    probability = float(prob)
    if probability < AUDIO_DETECT_THRESHOLD:
        return (False, None, probability)
    bits = message.squeeze().round().int().tolist()
    if isinstance(bits, int):
        bits = [bits]
    return (True, audio_bits_to_short(bits), probability)


__all__ = [
    "AUDIO_SR",
    "embed_audio_file",
    "detect_audio_file",
    "payload_to_audio_bits",
    "audio_bits_to_short",
]
