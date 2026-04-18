"""
OmniShield – backend/services/forensics.py
Mock implementations of:
  1. run_deepfake_shield  – GAN artifact detection on JPEG frames
  2. run_rppg_estimation  – Remote photoplethysmography (contactless heart rate)

Both functions accept a Base64-encoded JPEG frame and return structured dicts.

Production replacement notes:
  • Deepfake detection: run a MobileNet-based binary classifier fine-tuned on
    FaceForensics++ / DFDC datasets. Typical pipeline:
      base64 → JPEG bytes → PIL Image → face-crop → model inference
  • rPPG: extract a face ROI, compute mean R/G/B signal across N frames,
      apply bandpass filter (0.7–3 Hz), and estimate HR via FFT peak.
      Libraries: py-rPPG, CHROM, POS algorithm implementations.
"""

import asyncio
import base64
import logging
import random
from typing import Any

logger = logging.getLogger("omnishield.forensics")


# ─── Deepfake shield ──────────────────────────────────────────────────────────

async def run_deepfake_shield(frame_b64: str) -> dict[str, Any]:
    """
    Analyses a Base64 JPEG for deepfake GAN artifacts.

    Mock behaviour:
      • Returns is_deepfake=True ~5% of the time (base rate in our test set).
      • Simulates 0.8–1.8 s model inference latency.
      • Confidence is drawn from a realistic distribution (0.82–0.99).

    Production interface:
        frame_bytes = base64.b64decode(frame_b64)
        img = Image.open(io.BytesIO(frame_bytes)).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(img_tensor)
            prob_fake = torch.sigmoid(logits).item()
        return {
            "is_deepfake": prob_fake > DEEPFAKE_THRESHOLD,
            "confidence":  prob_fake,
            "artifacts":   ["splicing", "blending"] if prob_fake > 0.8 else [],
        }
    """
    await asyncio.sleep(random.uniform(0.8, 1.8))

    if not frame_b64:
        logger.warning("Deepfake shield received empty frame; defaulting to safe.")
        return {
            "is_deepfake": False,
            "confidence":  0.0,
            "artifacts":   [],
            "note":        "No frame provided",
        }

    # Validate the Base64 payload is a real image (basic sanity check)
    try:
        raw_bytes = base64.b64decode(frame_b64)
        if len(raw_bytes) < 1_000:  # JPEG must be > 1 KB
            raise ValueError("Frame too small to be a valid JPEG")
    except Exception as exc:
        logger.warning("Invalid frame payload: %s", exc)
        return {"is_deepfake": False, "confidence": 0.0, "artifacts": [], "note": "Invalid frame"}

    # Simulate inference
    confidence_fake = random.betavariate(1.5, 28)   # skewed toward real (low prob)
    is_fake         = confidence_fake > 0.50
    artifacts       = _sample_artifacts() if is_fake else []

    result = {
        "is_deepfake": is_fake,
        "confidence":  round(confidence_fake, 4),
        "artifacts":   artifacts,
        "frame_size_kb": round(len(raw_bytes) / 1024, 1),
    }
    logger.info("Deepfake result: %s", result)
    return result


def _sample_artifacts() -> list[str]:
    """Returns a random subset of GAN artifact classes for demo richness."""
    all_artifacts = [
        "facial_boundary_blur",
        "inconsistent_lighting",
        "eye_region_anomaly",
        "compression_inconsistency",
        "temporal_jitter",
    ]
    k = random.randint(1, 3)
    return random.sample(all_artifacts, k)


# ─── rPPG heart-rate estimator ────────────────────────────────────────────────

async def run_rppg_estimation(frame_b64: str) -> dict[str, Any]:
    """
    Estimates heart rate from facial skin colour micro-variations (rPPG).

    Mock behaviour:
      • Draws HR from a realistic resting distribution (58–92 bpm).
      • Draws SpO₂ (oxygen saturation) from 96–100%.
      • Simulates 0.5–1.5 s signal processing latency.
      • signal_quality: EXCELLENT / GOOD / POOR (affects PFL trust score).

    Liveness signal interpretation:
      • HR outside [45, 120] bpm → flag for manual review (possible mask/replay).
      • SpO₂ < 94 % → also flagged (physiologically unlikely in normal conditions).

    Production implementation (CHROM algorithm sketch):
        frames_rgb = [extract_face_roi(f) for f in recent_30_frames]
        r_signal   = [np.mean(f[:,:,0]) for f in frames_rgb]
        g_signal   = [np.mean(f[:,:,1]) for f in frames_rgb]
        b_signal   = [np.mean(f[:,:,2]) for f in frames_rgb]
        # CHROM method (de Haan & Jeanne, 2013)
        Xs = 3*r - 2*g
        Ys = 1.5*r + g - 1.5*b
        α  = np.std(Xs) / np.std(Ys)
        S  = Xs - α * Ys
        freqs = np.fft.rfftfreq(len(S), d=1/fps)
        peak  = freqs[np.argmax(np.abs(np.fft.rfft(S)))]
        hr    = int(peak * 60)
    """
    await asyncio.sleep(random.uniform(0.5, 1.5))

    if not frame_b64:
        return {
            "heart_rate_bpm":  None,
            "spo2_percent":    None,
            "signal_quality":  "POOR",
            "liveness_flag":   False,
            "note":            "No frame provided",
        }

    hr        = random.gauss(72, 8)     # realistic resting HR distribution
    hr        = max(48, min(115, int(hr)))
    spo2      = round(random.uniform(97.0, 99.5), 1)
    quality   = random.choices(
        ["EXCELLENT", "GOOD", "POOR"],
        weights=[0.55, 0.35, 0.10],
    )[0]

    # Liveness check: HR outside physiological range → possible replay attack
    liveness_flag = not (50 <= hr <= 110) or spo2 < 94.0

    result = {
        "heart_rate_bpm": hr,
        "spo2_percent":   spo2,
        "signal_quality": quality,
        "liveness_flag":  liveness_flag,
    }
    logger.info("rPPG result: %s", result)
    return result
