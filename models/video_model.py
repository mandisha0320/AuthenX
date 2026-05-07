"""
authenx/models/video_model.py

Video Deepfake Detection Module.

Strategy:
    1. Accept uploaded video file
    2. Extract 1 frame per second using OpenCV
    3. Run each frame through the image detection model
    4. Average the per-frame fake-probability scores
    5. Threshold the average to produce a final prediction

This approach reuses the image model, avoiding training a separate
temporal network. For production, consider adding a temporal model
(e.g., TimeSFormer or LSTM over frame features).
"""

import os
import logging
import tempfile
import cv2
import numpy as np
import torch
from PIL import Image

from models.image_model import get_image_model, BASE_TRANSFORM as INFERENCE_TRANSFORM, DEVICE

logger = logging.getLogger("authenx.video_model")

# ─── Constants ─────────────────────────────────────────────────────────────────

BATCH_SIZE = int(os.getenv("VIDEO_BATCH_SIZE", 8))   # frames per GPU batch
MAX_FRAMES  = int(os.getenv("VIDEO_MAX_FRAMES", 120)) # cap at 120 sampled frames


# ─── Frame Extraction ─────────────────────────────────────────────────────────

def extract_frames_1fps(video_path: str) -> list[np.ndarray]:
    """
    Extract exactly 1 frame per second from a video file using OpenCV.

    Args:
        video_path: Absolute path to the video file on disk.

    Returns:
        List of BGR numpy arrays (H, W, 3), one per second of video.
        Capped at MAX_FRAMES to prevent OOM on very long videos.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25  # fallback

    frame_interval = max(1, int(round(fps)))  # grab every N-th frame ≈ 1 fps
    frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            frames.append(frame)
            if len(frames) >= MAX_FRAMES:
                logger.info(f"Reached MAX_FRAMES={MAX_FRAMES} cap; stopping extraction.")
                break
        frame_idx += 1

    cap.release()
    logger.info(f"Extracted {len(frames)} frames at ~1fps from {video_path}")
    return frames


# ─── Frame Preprocessing ──────────────────────────────────────────────────────

def _frames_to_tensors(frames: list[np.ndarray]) -> torch.Tensor:
    """Convert a list of BGR OpenCV frames to a batched float tensor."""
    tensors = []
    for bgr in frames:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        tensors.append(INFERENCE_TRANSFORM(pil))
    return torch.stack(tensors)  # (N, 3, 224, 224)


# ─── Batch Inference ──────────────────────────────────────────────────────────

def _run_batch_inference(frame_tensors: torch.Tensor) -> list[float]:
    """
    Run model inference in batches to avoid OOM errors.

    Returns:
        List of per-frame fake probabilities (float in [0, 1]).
    """
    model = get_image_model()
    fake_probs = []

    for start in range(0, len(frame_tensors), BATCH_SIZE):
        batch = frame_tensors[start : start + BATCH_SIZE].to(DEVICE)
        with torch.no_grad():
            logits = model(batch)                          # (B, 2)
            probs  = torch.softmax(logits, dim=1)          # (B, 2)
            fake_p = probs[:, 1].cpu().tolist()            # index 1 = AI Generated
        fake_probs.extend(fake_p)

    return fake_probs


# ─── Main Prediction Function ─────────────────────────────────────────────────

def predict_video(video_bytes: bytes, original_filename: str = "video.mp4") -> dict:
    """
    Run deepfake detection on a video file given as raw bytes.

    Args:
        video_bytes:       Raw bytes of the uploaded video.
        original_filename: Used to infer file extension for temp file.

    Returns:
        dict with keys:
            prediction       (str)   – e.g. "Likely AI Generated"
            confidence_score (float) – averaged fake probability × 100
            frames_analyzed  (int)   – number of frames processed
            per_frame_scores (list)  – [{"frame": N, "score": X.XX}, ...]
    """
    # Determine extension from original filename
    ext = os.path.splitext(original_filename)[-1].lower() or ".mp4"

    # Write bytes to a named temp file (OpenCV needs a real path)
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    try:
        # ── Extract frames ──
        frames = extract_frames_1fps(tmp_path)
        if not frames:
            raise ValueError("No frames could be extracted from the video.")

        # ── Preprocess ──
        frame_tensors = _frames_to_tensors(frames)

        # ── Inference ──
        fake_probs = _run_batch_inference(frame_tensors)

        # ── Aggregate ──
        avg_fake_prob = float(np.mean(fake_probs))
        confidence = round(avg_fake_prob * 100, 2)

        # ── Labeling with graduated thresholds ──
        if avg_fake_prob >= 0.75:
            prediction = "Highly Likely AI Generated"
        elif avg_fake_prob >= 0.50:
            prediction = "Likely AI Generated"
        elif avg_fake_prob >= 0.30:
            prediction = "Uncertain / Possibly AI Generated"
        else:
            prediction = "Likely Authentic"

        per_frame = [
            {"frame": i + 1, "score": round(p * 100, 2)}
            for i, p in enumerate(fake_probs)
        ]

        return {
            "prediction": prediction,
            "confidence_score": confidence,
            "frames_analyzed": len(frames),
            "per_frame_scores": per_frame,
        }

    finally:
        # Always clean up the temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
