"""Parameter sweep for the QIM image mark: coefficient x strength Q.

Finds the (COEF, Q) combo that maximizes recompression survival while holding
imperceptibility (PSNR >= 38). Run locally/CI: python -m bench.tune_qim
"""
import json

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio

from bench import image_attacks, image_corpus
from engine import image_mark
from engine.constants import CODEWORD_BITS, NSYM
from engine.ecc import ReedSolomonError, bits_to_bytes, bytes_to_bits, rs_decode, rs_encode
from engine.payload import decode_message, encode_message

ATTACKS = {
    "jpeg_q75": lambda b: image_attacks.jpeg(b, 75),
    "jpeg_q50": lambda b: image_attacks.jpeg(b, 50),
    "jpeg_q35": lambda b: image_attacks.jpeg(b, 35),
    "social": image_attacks.social_sim,
    "screenshot": image_attacks.screenshot_sim,
}

COEFS = [(4, 3), (3, 2), (2, 1), (2, 2)]
QS = [12, 20, 28, 36, 44]


def _codeword(payload: int) -> list[int]:
    return bytes_to_bits(rs_encode(encode_message(payload), NSYM))


def _try_detect(arr: np.ndarray, q: float, payload: int) -> bool:
    # detect with a q-drift ladder (recompression shifts effective step)
    for qq in (q, q * 0.75, q * 0.5):
        bits = image_mark.extract_bits_from_frame(arr, CODEWORD_BITS, qq)
        try:
            msg = rs_decode(bits_to_bytes(bits), NSYM)
        except ReedSolomonError:
            continue
        ok, pid, _ = decode_message(msg)
        if ok and pid == payload:
            return True
    return False


def run() -> None:
    images = [cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
              for b in image_corpus.all_images()]
    results = []

    for coef in COEFS:
        image_mark.COEF = coef  # monkeypatch the module constant for the sweep
        for q in QS:
            psnrs, rec = [], {name: 0 for name in ATTACKS}
            clean_ok = 0
            for i, img in enumerate(images):
                payload = 6_000_000 + i
                marked = image_mark.embed_bits_in_frame(img, _codeword(payload), q)
                psnrs.append(float(peak_signal_noise_ratio(img, marked, data_range=255)))
                ok, buf = cv2.imencode(".png", marked)
                marked_bytes = buf.tobytes()
                if _try_detect(marked, q, payload):
                    clean_ok += 1
                for name, fn in ATTACKS.items():
                    attacked = cv2.imdecode(
                        np.frombuffer(fn(marked_bytes), np.uint8), cv2.IMREAD_COLOR)
                    if _try_detect(attacked, q, payload):
                        rec[name] += 1
            n = len(images)
            row = {
                "coef": str(coef), "q": q,
                "psnr": round(sum(psnrs) / n, 1),
                "clean": clean_ok / n,
                **{k: v / n for k, v in rec.items()},
            }
            results.append(row)
            print(json.dumps(row))


if __name__ == "__main__":
    run()
