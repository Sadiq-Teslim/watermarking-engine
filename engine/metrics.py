"""Imperceptibility metrics: PSNR, SSIM (scikit-image) and VMAF (ffmpeg libvmaf).

PSNR/SSIM are computed on aligned frame samples decoded from the source and the
watermarked output. VMAF is optional (returns None if the local ffmpeg lacks libvmaf)."""
import json
import re
import subprocess
import tempfile
from itertools import islice

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from engine import ffmpeg_io


def quality_psnr_ssim(src: str, watermarked: str, n_samples: int = 5) -> tuple[float, float]:
    src_frames = ffmpeg_io.iter_frames(src, sample_fps=1.0)
    wm_frames = ffmpeg_io.iter_frames(watermarked, sample_fps=1.0)
    psnrs: list[float] = []
    ssims: list[float] = []
    for a, b in islice(zip(src_frames, wm_frames, strict=False), n_samples):
        if a.shape != b.shape:
            continue
        psnrs.append(float(peak_signal_noise_ratio(a, b, data_range=255)))
        gray_a = np.asarray(a).mean(axis=2)
        gray_b = np.asarray(b).mean(axis=2)
        ssims.append(float(structural_similarity(gray_a, gray_b, data_range=255)))
    if not psnrs:
        return (0.0, 0.0)
    return (sum(psnrs) / len(psnrs), sum(ssims) / len(ssims))


def quality_vmaf(src: str, watermarked: str, timeout: int = 600) -> float | None:
    """Run libvmaf comparing `watermarked` (distorted) against `src` (reference)."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        log_path = tmp.name
    cmd = [
        "ffmpeg", "-v", "error",
        "-i", watermarked, "-i", src,
        "-lavfi", f"libvmaf=log_fmt=json:log_path={log_path}",
        "-f", "null", "-",
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=timeout, check=False)
    if result.returncode != 0:
        return None
    try:
        with open(log_path) as fh:
            data = json.load(fh)
        return float(data["pooled_metrics"]["vmaf"]["mean"])
    except Exception:
        # Fallback: some ffmpeg builds print VMAF to stderr instead of JSON.
        match = re.search(r"VMAF score:\s*([0-9.]+)", result.stderr.decode(errors="ignore"))
        return float(match.group(1)) if match else None
