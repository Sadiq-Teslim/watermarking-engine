"""TrustMark neural image watermarking (opt-in). Robust to screenshots / social re-encode.

The payload is carried as a short hex string; TrustMark's built-in BCH + validity bit gate
false positives. torch/trustmark are heavy and only installed when INSTALL_NEURAL=true, so
imports are deferred.

NOTE: validate TrustMark's encode/decode signatures against the installed version at deploy.
"""
import io

from engine.constants import MAX_PAYLOAD_ID

_model = None


def is_available() -> bool:
    try:
        import trustmark  # noqa: F401
    except ImportError:
        return False
    return True


def _load():
    global _model
    if _model is None:
        from trustmark import TrustMark  # deferred heavy import
        _model = TrustMark(verbose=False, model_type="Q")
    return _model


def _payload_to_secret(payload_id: int) -> str:
    return f"{payload_id:07X}"  # 28-bit id -> 7 hex chars


def _secret_to_payload(secret: str) -> int | None:
    try:
        value = int(str(secret).strip(), 16)
    except (ValueError, AttributeError):
        return None
    return value if 1 <= value <= MAX_PAYLOAD_ID else None


def embed_image(data: bytes, payload_id: int) -> bytes:
    from PIL import Image

    tm = _load()
    cover = Image.open(io.BytesIO(data)).convert("RGB")
    stego = tm.encode(cover, _payload_to_secret(payload_id))
    buf = io.BytesIO()
    stego.save(buf, format="PNG")
    return buf.getvalue()


def detect_image(data: bytes) -> tuple[bool, int | None, float]:
    from PIL import Image

    tm = _load()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    secret, present, _schema = tm.decode(img)
    if not present:
        return (False, None, 0.0)
    payload = _secret_to_payload(secret)
    return (payload is not None, payload, 1.0 if payload is not None else 0.0)
