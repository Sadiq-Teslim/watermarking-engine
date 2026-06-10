"""Turn bench/report.json into a pass/fail CI decision.

Tier 1 is GATED on same-resolution recompression / photometric / temporal attacks and on
imperceptibility + zero false positives. Geometric attacks (resize/crop/rotate) are
REPORTED for Tier 1 but gated only once the neural tier (P7) lands.
"""
import json
import sys
from pathlib import Path

REPORT_PATH = Path(__file__).parent / "report.json"

# (metric_path, predicate, human description)
TIER1_RECOVERY_GATES = {
    "clean": 1.00,
    "h264_crf23": 0.99,
    "h264_crf28": 0.97,
    "h265_crf28": 0.97,
    "bitrate_500k": 0.95,
    "fps_24": 0.99,
    "brightness": 0.99,
    "noise": 0.95,
}
IMPERCEPTIBILITY_GATES = {"psnr": 40.0, "ssim": 0.98}  # vmaf checked if present
VMAF_GATE = 93.0

# Geometric attacks: only the neural tier (videoseal) is gated on these.
NEURAL_RECOVERY_GATES = {
    "resize_720": 0.97,
    "resize_480": 0.95,
    "crop_15": 0.90,
    "rotate_2deg": 0.85,
}


def evaluate(report: dict) -> list[str]:
    failures: list[str] = []
    recovery = report.get("recovery", {})
    engine = report.get("engine", "qim-dct")

    for name, threshold in TIER1_RECOVERY_GATES.items():
        got = recovery.get(name)
        if got is None:
            failures.append(f"missing recovery metric: {name}")
        elif got < threshold:
            failures.append(f"recovery[{name}]={got:.4f} < {threshold:.2f}")

    if engine == "videoseal":
        for name, threshold in NEURAL_RECOVERY_GATES.items():
            got = recovery.get(name)
            if got is None:
                failures.append(f"missing geometric recovery metric: {name}")
            elif got < threshold:
                failures.append(f"recovery[{name}]={got:.4f} < {threshold:.2f} (neural gate)")

    if report.get("false_positives", 1) != 0:
        failures.append(f"false_positives={report.get('false_positives')} (must be 0)")

    imp = report.get("imperceptibility", {})
    for metric, threshold in IMPERCEPTIBILITY_GATES.items():
        got = imp.get(metric)
        if got is None or got < threshold:
            failures.append(f"imperceptibility[{metric}]={got} < {threshold}")
    if imp.get("vmaf") is not None and imp["vmaf"] < VMAF_GATE:
        failures.append(f"imperceptibility[vmaf]={imp['vmaf']} < {VMAF_GATE}")

    return failures


def main() -> int:
    if not REPORT_PATH.exists():
        print("report.json not found — run `python -m bench.run_benchmark` first", file=sys.stderr)
        return 2
    report = json.loads(REPORT_PATH.read_text())
    failures = evaluate(report)

    print("=== Tier 1 recovery ===")
    for name, got in report.get("recovery", {}).items():
        print(f"  {name:16s} {got:.4f}")
    print("=== imperceptibility ===", report.get("imperceptibility"))
    fp = report.get("false_positives")
    fp_total = report.get("false_positive_total")
    print(f"=== false positives === {fp}/{fp_total}")

    if failures:
        print("\nGATES FAILED:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print("\nALL GATES PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
