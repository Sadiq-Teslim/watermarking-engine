"""Run the image robustness benchmark -> bench/image_report.json.

For each corpus image: embed a known payload, apply each attack, attempt detection, and
record exact-payload recovery. Also measures imperceptibility (PSNR/SSIM) and the
false-positive rate on UNMARKED images. `image_gates.py` turns this into a CI pass/fail.

Engine selected via FPWM_BENCH_ENGINE:
  qim-dct         classical, no hints (recompression-only survival)
  qim-dct-hinted  classical + original-size hints — the PRODUCT mode (the registry knows
                  every original's dimensions), survives resize/social/screenshot too
  trustmark       neural tier (survives crop/rotate/unknown sizes)
"""
import json
import os
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from bench import image_attacks, image_corpus

REPORT_PATH = Path(__file__).parent / "image_report.json"
BASE_PAYLOAD = 5_000_000
ENGINE = os.environ.get("FPWM_BENCH_ENGINE", "qim-dct")


def _arr(data: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def _make_engine():
    """Return (embed, detect) where detect(data, original_size) -> (found, payload)."""
    if ENGINE == "trustmark":
        from engine import image_neural

        def detect(data, _size):
            found, pid, _ = image_neural.detect_image(data)
            return found, pid
        return image_neural.embed_image, detect

    from engine import image_codec
    hinted = ENGINE == "qim-dct-hinted"

    def detect(data, size):
        sizes = [size] if (hinted and size) else None
        found, pid, _ = image_codec.detect_image(data, candidate_sizes=sizes)
        return found, pid
    return image_codec.embed_image, detect


def run() -> dict:
    embed_image, detect = _make_engine()
    images = image_corpus.all_images()
    if not images:
        raise RuntimeError("no corpus images available")

    attack_set = {**image_attacks.RECOMPRESSION_ATTACKS, **image_attacks.GEOMETRIC_ATTACKS,
                  **image_attacks.CROP_ROTATE_ATTACKS}
    per_attack = {name: {"recovered": 0, "total": 0} for name in attack_set}
    per_attack["clean"] = {"recovered": 0, "total": 0}
    psnrs: list[float] = []
    ssims: list[float] = []
    false_positives = 0

    for i, data in enumerate(images):
        payload = BASE_PAYLOAD + i
        marked = embed_image(data, payload)

        a, b = _arr(data), _arr(marked)
        size = (a.shape[1], a.shape[0])  # (width, height) of the original
        if a.shape == b.shape:
            psnrs.append(float(peak_signal_noise_ratio(a, b, data_range=255)))
            ssims.append(float(structural_similarity(
                a.mean(axis=2), b.mean(axis=2), data_range=255)))

        found, pid = detect(marked, size)
        per_attack["clean"]["total"] += 1
        per_attack["clean"]["recovered"] += int(found and pid == payload)

        for name, fn in attack_set.items():
            per_attack[name]["total"] += 1
            try:
                attacked = fn(marked)
                found, pid = detect(attacked, size)
                per_attack[name]["recovered"] += int(found and pid == payload)
            except Exception:
                pass

        found, _ = detect(data, size)
        if found:
            false_positives += 1

    def rate(d: dict) -> float:
        return (d["recovered"] / d["total"]) if d["total"] else 0.0

    report = {
        "engine": ENGINE,
        "images": len(images),
        "recovery": {name: round(rate(d), 4) for name, d in per_attack.items()},
        "imperceptibility": {
            "psnr": round(sum(psnrs) / len(psnrs), 2) if psnrs else 0.0,
            "ssim": round(sum(ssims) / len(ssims), 4) if ssims else 0.0,
        },
        "false_positives": false_positives,
        "false_positive_total": len(images),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
