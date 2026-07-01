"""TrustMark neural image watermarking (opt-in). Robust to screenshots / social re-encode.

The payload is carried as a short hex string; TrustMark's built-in BCH + validity bit gate
false positives. torch/trustmark are heavy and only installed when INSTALL_NEURAL=true, so
imports are deferred.

NOTE: validate TrustMark's encode/decode signatures against the installed version at deploy.
"""
import io
import importlib.util
import os

from engine.constants import MAX_PAYLOAD_ID

_model = None


def is_available() -> bool:
    if os.environ.get("FPWM_TRUSTMARK_ENABLED", "").lower() not in {"1", "true", "yes", "on"}:
        return False
    # Checking capabilities must stay lightweight. Importing TrustMark loads the
    # neural runtime and can exhaust the web process before a watermark job starts.
    return importlib.util.find_spec("trustmark") is not None


def _load():
    global _model
    if _model is None:
        from trustmark import TrustMark  # deferred heavy import
        model_type = os.environ.get("FPWM_TRUSTMARK_MODEL", "C").strip().upper() or "C"
        _model = TrustMark(verbose=False, model_type=model_type)
    return _model


def _payload_to_secret(payload_id: int) -> str:
    return f"{payload_id:07X}"  # 28-bit id -> 7 hex chars


def _secret_to_payload(secret: str) -> int | None:
    try:
        value = int(str(secret).strip(), 16)
    except (ValueError, AttributeError):
        return None
    return value if 1 <= value <= MAX_PAYLOAD_ID else None


def _max_side() -> int:
    try:
        return max(0, int(os.environ.get("FPWM_TRUSTMARK_MAX_SIDE", "768")))
    except ValueError:
        return 768


def _bounded_rgb(data: bytes):
    from PIL import Image

    img = Image.open(io.BytesIO(data)).convert("RGB")
    max_side = _max_side()
    if max_side and max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return img


def embed_image(data: bytes, payload_id: int) -> bytes:
    tm = _load()
    cover = _bounded_rgb(data)
    stego = tm.encode(cover, _payload_to_secret(payload_id))
    buf = io.BytesIO()
    stego.save(buf, format="PNG")
    return buf.getvalue()


def detect_image(data: bytes) -> tuple[bool, int | None, float]:
    tm = _load()
    img = _bounded_rgb(data)
    secret, present, _schema = tm.decode(img)
    if not present:
        return (False, None, 0.0)
    payload = _secret_to_payload(secret)
    return (payload is not None, payload, 1.0 if payload is not None else 0.0)
