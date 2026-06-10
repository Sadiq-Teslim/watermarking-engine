"""Run the image robustness benchmark -> bench/image_report.json.

For each corpus image: embed a known payload, apply each attack, attempt detection, and
record exact-payload recovery. Also measures imperceptibility (PSNR/SSIM) and the
false-positive rate on UNMARKED images. `image_gates.py` turns this into a CI pass/fail.

Engine selected via FPWM_BENCH_ENGINE: qim-dct (default) or trustmark (neural).
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


def _engine():
    if ENGINE == "trustmark":
        from engine import image_neural
        return image_neural.embed_image, image_neural.detect_image
    from engine import image_codec
    return image_codec.embed_image, image_codec.detect_image


def _arr(data: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def run() -> dict:
    embed_image, detect_image = _engine()
    images = image_corpus.all_images()
    if not images:
        raise RuntimeError("no corpus images available")

    attack_set = {**image_attacks.RECOMPRESSION_ATTACKS, **image_attacks.GEOMETRIC_ATTACKS}
    per_attack = {name: {"recovered": 0, "total": 0} for name in attack_set}
    per_attack["clean"] = {"recovered": 0, "total": 0}
    psnrs: list[float] = []
    ssims: list[float] = []
    false_positives = 0

    for i, data in enumerate(images):
        payload = BASE_PAYLOAD + i
        marked = embed_image(data, payload)

        a, b = _arr(data), _arr(marked)
        if a.shape == b.shape:
            psnrs.append(float(peak_signal_noise_ratio(a, b, data_range=255)))
            ssims.append(float(structural_similarity(
                a.mean(axis=2), b.mean(axis=2), data_range=255)))

        marked_det = detect_image(marked)
        per_attack["clean"]["total"] += 1
        per_attack["clean"]["recovered"] += int(marked_det[0] and marked_det[1] == payload)

        for name, fn in attack_set.items():
            per_attack[name]["total"] += 1
            try:
                attacked = fn(marked)
                det = detect_image(attacked)
                per_attack[name]["recovered"] += int(det[0] and det[1] == payload)
            except Exception:
                pass

        if detect_image(data)[0]:
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
