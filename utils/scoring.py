"""
authenx/utils/scoring.py

Unified confidence score utilities.

All scoring functions normalize outputs to a percentage in [0, 100].

Functions:
    softmax_confidence  – for image/video models (from raw logits or probabilities)
    cosine_to_score     – for text similarity (from cosine similarity in [0,1])
    scale_to_percentage – generic linear scaling helper
"""

import torch
import numpy as np


def softmax_confidence(logits_or_probs: list[float], target_class: int = 1) -> float:
    """
    Compute a confidence percentage from raw logits or pre-softmax probabilities.

    Args:
        logits_or_probs: List or array of raw logits (or unnormalized scores) for each class.
        target_class:    Index of the class whose confidence we want. Default 1 (Fake/AI).

    Returns:
        Confidence score as a percentage (0.0 – 100.0), rounded to 2 decimal places.

    Example:
        logits = [1.2, 3.4]  # [real, fake]
        softmax_confidence(logits, target_class=1) → ~87.56
    """
    t = torch.tensor(logits_or_probs, dtype=torch.float32)
    probs = torch.softmax(t, dim=0)
    return round(probs[target_class].item() * 100, 2)


def cosine_to_score(cosine_sim: float, low: float = 0.0, high: float = 1.0) -> float:
    """
    Map a cosine similarity value to a 0–100 percentage.

    Performs linear min-max scaling between the given `low` and `high` bounds.

    Args:
        cosine_sim: Cosine similarity value (typically in [0, 1] for normalized embeddings).
        low:        Value that maps to 0% (default 0.0).
        high:       Value that maps to 100% (default 1.0).

    Returns:
        Score as a percentage (0.0 – 100.0).
    """
    if high == low:
        return 50.0  # undefined range → neutral
    scaled = (cosine_sim - low) / (high - low)
    return round(float(np.clip(scaled * 100, 0.0, 100.0)), 2)


def scale_to_percentage(value: float, min_val: float, max_val: float) -> float:
    """
    Generic linear scaling of `value` from [min_val, max_val] → [0, 100].

    Args:
        value:   The raw value to scale.
        min_val: Minimum of the input range (maps to 0%).
        max_val: Maximum of the input range (maps to 100%).

    Returns:
        Percentage as a float, clamped to [0, 100].
    """
    if max_val == min_val:
        return 50.0
    pct = (value - min_val) / (max_val - min_val) * 100
    return round(float(np.clip(pct, 0.0, 100.0)), 2)


def aggregate_frame_scores(frame_probs: list[float]) -> dict:
    """
    Aggregate per-frame fake probabilities into a single video-level score.

    Args:
        frame_probs: List of per-frame fake probabilities in [0, 1].

    Returns:
        dict with:
            mean_score     (float) – mean fake prob × 100
            median_score   (float) – median × 100
            max_score      (float) – max × 100
            std_dev        (float) – standard deviation × 100
    """
    arr = np.array(frame_probs)
    return {
        "mean_score":   round(float(arr.mean()) * 100, 2),
        "median_score": round(float(np.median(arr)) * 100, 2),
        "max_score":    round(float(arr.max()) * 100, 2),
        "std_dev":      round(float(arr.std()) * 100, 2),
    }
