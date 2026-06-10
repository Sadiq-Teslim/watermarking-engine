"""Pass/fail gates for the image benchmark (bench/image_report.json).

qim-dct is gated on recompression + imperceptibility + zero false positives. Geometric and
screenshot/social attacks are REPORTED for qim-dct but GATED only for the neural tier.
"""
import json
import sys
from pathlib import Path

REPORT_PATH = Path(__file__).parent / "image_report.json"

RECOMPRESSION_GATES = {
    "clean": 1.00,
    "jpeg_q90": 0.99,
    "jpeg_q75": 0.95,
    "jpeg_q50": 0.85,
    "brightness": 0.95,
}
NEURAL_GEOMETRIC_GATES = {
    "resize_50": 0.90,
    "resize_150": 0.95,
    "crop_10": 0.85,
    "screenshot": 0.80,
    "social": 0.80,
}
IMPERCEPTIBILITY_GATES = {"psnr": 38.0, "ssim": 0.96}


def evaluate(report: dict) -> list[str]:
    failures: list[str] = []
    recovery = report.get("recovery", {})
    engine = report.get("engine", "qim-dct")

    for name, threshold in RECOMPRESSION_GATES.items():
        got = recovery.get(name)
        if got is None:
            failures.append(f"missing recovery metric: {name}")
        elif got < threshold:
            failures.append(f"recovery[{name}]={got:.4f} < {threshold:.2f}")

    if engine == "trustmark":
        for name, threshold in NEURAL_GEOMETRIC_GATES.items():
            got = recovery.get(name)
            if got is None:
                failures.append(f"missing geometric metric: {name}")
            elif got < threshold:
                failures.append(f"recovery[{name}]={got:.4f} < {threshold:.2f} (neural gate)")

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
