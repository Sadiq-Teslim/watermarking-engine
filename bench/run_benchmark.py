"""Run the robustness benchmark and emit bench/report.json.

For each corpus clip: embed a known payload, apply each attack, attempt detection, and
record whether the exact payload was recovered. Also measures imperceptibility (PSNR/SSIM/
VMAF) and the false-positive rate on UNMARKED clips. `gates.py` turns this report into a
pass/fail CI decision.
"""
import json
import os
import tempfile
from pathlib import Path

from bench import attacks, corpus
from engine import channels, metrics

REPORT_PATH = Path(__file__).parent / "report.json"
BASE_PAYLOAD = 4_242_424
ENGINE = os.environ.get("FPWM_BENCH_ENGINE", "qim-dct")


def _recovered(path: str, expected: int) -> tuple[bool, float]:
    result = channels.detect(path, engine=ENGINE)
    return (bool(result["marked"]) and result["payload"] == expected, result.get("confidence", 0.0))


def run() -> dict:
    clips = corpus.all_clips()
    if not clips:
        raise RuntimeError("no corpus clips available")

    attack_set = {**attacks.TIER1_ATTACKS, **attacks.GEOMETRIC_ATTACKS}
    per_attack: dict[str, dict] = {name: {"recovered": 0, "total": 0} for name in attack_set}
    per_attack["clean"] = {"recovered": 0, "total": 0}
    psnrs: list[float] = []
    ssims: list[float] = []
    vmafs: list[float] = []
    false_positives = 0
    fp_total = 0

    with tempfile.TemporaryDirectory() as tmp:
        tmpd = Path(tmp)
        for i, clip in enumerate(clips):
            payload = BASE_PAYLOAD + i
            marked = tmpd / f"marked_{i}.mp4"
            channels.embed(str(clip), str(marked), payload, engine=ENGINE, crf=18)

            # Imperceptibility.
            psnr, ssim = metrics.quality_psnr_ssim(str(clip), str(marked))
            psnrs.append(psnr)
            ssims.append(ssim)
            vmaf = metrics.quality_vmaf(str(clip), str(marked))
            if vmaf is not None:
                vmafs.append(vmaf)

            # Clean (no attack) recovery.
            ok, _ = _recovered(str(marked), payload)
            per_attack["clean"]["total"] += 1
            per_attack["clean"]["recovered"] += int(ok)

            # Attack battery.
            for name, fn in attack_set.items():
                out = tmpd / f"att_{i}_{name}.mp4"
                try:
                    fn(str(marked), str(out))
                except Exception:
                    per_attack[name]["total"] += 1
                    continue
                ok, _ = _recovered(str(out), payload)
                per_attack[name]["total"] += 1
                per_attack[name]["recovered"] += int(ok)

            # False positives: the ORIGINAL unmarked clip must not be detected as marked.
            fp_total += 1
            if channels.detect(str(clip), engine=ENGINE)["marked"]:
                false_positives += 1

    def rate(d: dict) -> float:
        return (d["recovered"] / d["total"]) if d["total"] else 0.0

    report = {
        "engine": ENGINE,
        "clips": len(clips),
        "recovery": {name: round(rate(d), 4) for name, d in per_attack.items()},
        "recovery_counts": per_attack,
        "imperceptibility": {
            "psnr": round(sum(psnrs) / len(psnrs), 2) if psnrs else 0.0,
            "ssim": round(sum(ssims) / len(ssims), 4) if ssims else 0.0,
            "vmaf": round(sum(vmafs) / len(vmafs), 2) if vmafs else None,
        },
        "false_positives": false_positives,
        "false_positive_total": fp_total,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
