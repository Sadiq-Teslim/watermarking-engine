import io
import sys
import types

from PIL import Image


def _png(size=(1200, 900)) -> bytes:
    img = Image.new("RGB", size, (20, 80, 140))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_trustmark_disabled_without_explicit_env(monkeypatch):
    from engine import image_neural

    monkeypatch.delenv("FPWM_TRUSTMARK_ENABLED", raising=False)
    monkeypatch.setitem(sys.modules, "trustmark", types.SimpleNamespace())

    assert image_neural.is_available() is False


def test_trustmark_available_when_package_and_env_enabled(monkeypatch):
    from engine import image_neural

    monkeypatch.setenv("FPWM_TRUSTMARK_ENABLED", "true")
    monkeypatch.setitem(sys.modules, "trustmark", types.SimpleNamespace())

    assert image_neural.is_available() is True


def test_embed_uses_configured_model_and_bounds_image(monkeypatch):
    from engine import image_neural

    calls = {}

    class FakeTrustMark:
        def __init__(self, verbose, model_type):
            calls["verbose"] = verbose
            calls["model_type"] = model_type

        def encode(self, cover, secret):
            calls["cover_size"] = cover.size
            calls["secret"] = secret
            return cover

    monkeypatch.setenv("FPWM_TRUSTMARK_MODEL", "c")
    monkeypatch.setenv("FPWM_TRUSTMARK_MAX_SIDE", "768")
    monkeypatch.setattr(image_neural, "_model", None)
    monkeypatch.setitem(
        sys.modules,
        "trustmark",
        types.SimpleNamespace(TrustMark=FakeTrustMark),
    )

    out = image_neural.embed_image(_png(), 12345)
    assert calls == {
        "verbose": False,
        "model_type": "C",
        "cover_size": (768, 576),
        "secret": "0003039",
    }
    assert Image.open(io.BytesIO(out)).size == (768, 576)


def test_detect_bounds_image_before_decoding(monkeypatch):
    from engine import image_neural

    calls = {}

    class FakeTrustMark:
        def __init__(self, verbose, model_type):
            pass

        def decode(self, image):
            calls["decode_size"] = image.size
            return ("0003039", True, "schema")

    monkeypatch.setenv("FPWM_TRUSTMARK_MAX_SIDE", "512")
    monkeypatch.setattr(image_neural, "_model", FakeTrustMark(False, "C"))

    marked, payload, confidence = image_neural.detect_image(_png())

    assert calls["decode_size"] == (512, 384)
    assert marked is True
    assert payload == 12345
    assert confidence == 1.0
