"""Pass/fail gates for the image benchmark (bench/image_report.json).

Gating is per engine mode:
  qim-dct         recompression survival + imperceptibility + zero false positives
  qim-dct-hinted  + resize/screenshot/social/platform pipelines (product mode: detection
                  gets original-size hints from the registry)
  trustmark       + crop/rotate (neural tier owns grid-destroying attacks)
"""
import json
import sys
from pathlib import Path

REPORT_PATH = Path(__file__).parent / "image_report.json"

RECOMPRESSION_GATES = {
    "clean": 1.00,
    "jpeg_q90": 0.99,
    "jpeg_q75": 0.99,
    "jpeg_q50": 0.95,
    "jpeg_q35": 0.90,
    "brightness": 0.95,
}
HINTED_GATES = {
    "resize_50": 0.90,
    "resize_150": 0.95,
    "screenshot": 0.90,
    "social": 0.90,
    "instagram": 0.90,
    "twitter": 0.95,
    "whatsapp": 0.90,
    "reshare": 0.80,
}
NEURAL_GATES = {
    **{k: 0.80 for k in HINTED_GATES},
    "crop_10": 0.85,
    "crop_25": 0.75,
    "rotate_3deg": 0.75,
}
IMPERCEPTIBILITY_GATES = {"psnr": 38.0, "ssim": 0.96}


def _check(recovery: dict, gates: dict, label: str) -> list[str]:
    failures = []
    for name, threshold in gates.items():
        got = recovery.get(name)
        if got is None:
            failures.append(f"missing recovery metric: {name}")
        elif got < threshold:
            failures.append(f"recovery[{name}]={got:.4f} < {threshold:.2f} ({label})")
    return failures


def evaluate(report: dict) -> list[str]:
    recovery = report.get("recovery", {})
    engine = report.get("engine", "qim-dct")
    failures = _check(recovery, RECOMPRESSION_GATES, "recompression")

    if engine == "qim-dct-hinted":
        failures += _check(recovery, HINTED_GATES, "hinted")
    if engine == "trustmark":
        failures += _check(recovery, NEURAL_GATES, "neural")

    if report.get("false_positives", 1) != 0:
        failures.append(f"false_positives={report.get('false_positives')} (must be 0)")

    imp = report.get("imperceptibility", {})
    for metric, threshold in IMPERCEPTIBILITY_GATES.items():
        got = imp.get(metric)
        if got is None or got < threshold:
            failures.append(f"imperceptibility[{metric}]={got} < {threshold}")

    return failures


def main() -> int:
    if not REPORT_PATH.exists():
        print("image_report.json not found — run bench.run_image_benchmark first", file=sys.stderr)
        return 2
    report = json.loads(REPORT_PATH.read_text())
    failures = evaluate(report)

    print(f"=== image recovery ({report.get('engine')}) ===")
    for name, got in report.get("recovery", {}).items():
        print(f"  {name:14s} {got:.4f}")
    print("=== imperceptibility ===", report.get("imperceptibility"))
    fp = report.get("false_positives")
    fp_total = report.get("false_positive_total")
    print(f"=== false positives === {fp}/{fp_total}")

    if failures:
        print("\nGATES FAILED:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print("\nALL IMAGE GATES PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
