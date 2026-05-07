"""
authenx/models/text_detection_model.py

AI-Generated TEXT Detection Module.

This is separate from the headline *verification* service (which checks
if a news headline is misinformation by comparing it to web sources).

This module answers a different question:
    "Was this piece of text written by an AI (like ChatGPT/Claude)
     or by a real human?"

Strategy — ensemble of 3 complementary signals:

1. PERPLEXITY SCORE  (via GPT-2 language model)
   AI text tends to be low-perplexity (predictable, on-distribution).
   Human text has higher perplexity (more surprising word choices).

2. BURSTINESS SCORE
   Human writing has uneven sentence lengths (mix of short punchy
   sentences and long complex ones). AI writing is uniform/monotonic.
   We measure the coefficient of variation of sentence lengths.

3. STYLOMETRIC FEATURES  (classical ML — LogisticRegression)
   - Type-Token Ratio  (vocabulary diversity)
   - Average word length
   - Punctuation density
   - Repeated phrase density
   These are fed into a lightweight LogisticRegression classifier
   trained on a small heuristic dataset (production: replace with
   a model trained on real AI vs human corpora).

Final confidence = weighted average of the three signals.

Supports:
  - Plain text input (string)
  - .txt file upload
  - .pdf file upload  (text extracted via pdfminer)
  - .docx file upload (text extracted via python-docx)

Output:
  {
    "prediction":       "AI Generated" | "Likely Human Written",
    "confidence_score": 78.4,
    "signals": {
        "perplexity_score":  23.4,   # lower = more AI-like
        "burstiness_score":  0.18,   # lower = more AI-like
        "stylometric_score": 72.0    # % AI likelihood from features
    },
    "word_count":  542,
    "char_count":  3201,
    "excerpt":     "First 200 chars..."
  }
"""

import os
import io
import re
import math
import logging
import warnings
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger("authenx.text_detection_model")

# ─── Constants ─────────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Signal weights for final ensemble score
W_PERPLEXITY  = 0.45
W_BURSTINESS  = 0.25
W_STYLOMETRIC = 0.30

MIN_WORDS = 30   # Reject inputs shorter than this

# ─── GPT-2 Perplexity Scorer ───────────────────────────────────────────────────

_gpt2_model     = None
_gpt2_tokenizer = None


def _load_gpt2():
    """Lazy-load GPT-2 small for perplexity scoring."""
    global _gpt2_model, _gpt2_tokenizer
    if _gpt2_model is None:
        logger.info("Loading GPT-2 for perplexity scoring...")
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
        _gpt2_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        _gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
        _gpt2_model.eval()
        logger.info("GPT-2 loaded.")
    return _gpt2_model, _gpt2_tokenizer


def compute_perplexity(text: str, max_tokens: int = 512) -> float:
    """
    Compute GPT-2 perplexity of a text.

    Lower perplexity → text is more "on-distribution" for a language model
    → more likely to be AI-generated.

    Returns perplexity as a float. Typical ranges:
        AI text:    10 – 50   (very predictable)
        Human text: 50 – 300+ (more surprising)
    """
    model, tokenizer = _load_gpt2()

    encodings = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_tokens,
    )
    input_ids = encodings.input_ids.to(DEVICE)

    if input_ids.shape[1] < 5:
        return 999.0  # too short to score

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss  # cross-entropy loss

    perplexity = math.exp(loss.item())
    return round(perplexity, 2)


def perplexity_to_ai_prob(perplexity: float) -> float:
    """
    Map perplexity to an AI-likelihood probability in [0, 1].

    Sigmoid-like curve:
        perplexity ≤ 20  → ~95% AI
        perplexity = 50  → ~70% AI
        perplexity = 100 → ~40% AI
        perplexity ≥ 200 → ~10% AI
    """
    # Logistic: prob = 1 / (1 + exp(k*(x - x0)))
    # Calibrated so x0=80 (crossover), k=0.03
    prob = 1.0 / (1.0 + math.exp(0.03 * (perplexity - 80)))
    return float(np.clip(prob, 0.0, 1.0))


# ─── Burstiness Scorer ────────────────────────────────────────────────────────

def compute_burstiness(text: str) -> float:
    """
    Measure sentence-length burstiness.

    Burstiness = std(sentence_lengths) / mean(sentence_lengths)
    (coefficient of variation)

    Low burstiness (< 0.4) → uniform sentence lengths → likely AI.
    High burstiness (> 0.7) → human-like variation.

    Returns burstiness score as float.
    """
    sentences = re.split(r'[.!?]+', text)
    lengths = [len(s.split()) for s in sentences if len(s.split()) >= 3]

    if len(lengths) < 3:
        return 0.5  # not enough sentences; neutral

    mean_len = np.mean(lengths)
    std_len  = np.std(lengths)

    if mean_len == 0:
        return 0.5

    return round(float(std_len / mean_len), 4)


def burstiness_to_ai_prob(burstiness: float) -> float:
    """
    Map burstiness to AI-likelihood probability.

    Low burstiness → high AI probability.
    Threshold: 0.5 = crossover point.
    """
    # Inverse sigmoid: low burstiness → high AI prob
    prob = 1.0 / (1.0 + math.exp(8 * (burstiness - 0.45)))
    return float(np.clip(prob, 0.0, 1.0))


# ─── Stylometric Feature Scorer ───────────────────────────────────────────────

def extract_stylometric_features(text: str) -> dict:
    """
    Extract classical stylometric features from text.

    Features:
        ttr             – Type-Token Ratio (unique words / total words)
        avg_word_len    – Average word length in characters
        punct_density   – Punctuation chars / total chars
        filler_density  – Common AI filler phrases per 100 words
        avg_sent_len    – Average sentence length in words
    """
    words = re.findall(r'\b\w+\b', text.lower())
    total_words = len(words)

    if total_words == 0:
        return {"ttr": 0, "avg_word_len": 0, "punct_density": 0,
                "filler_density": 0, "avg_sent_len": 0}

    # Type-Token Ratio
    ttr = len(set(words)) / total_words

    # Average word length
    avg_word_len = sum(len(w) for w in words) / total_words

    # Punctuation density
    punct_chars = sum(1 for c in text if c in '.,;:!?()[]{}"\'-')
    punct_density = punct_chars / max(len(text), 1)

    # AI filler phrase density
    ai_fillers = [
        r'\bin conclusion\b', r'\bto summarize\b', r'\bit is worth noting\b',
        r'\bfurthermore\b', r'\bmoreover\b', r'\bin addition\b',
        r'\bit is important to\b', r'\bone must consider\b',
        r'\boverall\b', r'\bin summary\b', r'\bultimately\b',
        r'\bdelve\b', r'\bcertainly\b', r'\babsolutely\b',
    ]
    filler_count = sum(len(re.findall(p, text, re.IGNORECASE)) for p in ai_fillers)
    filler_density = (filler_count / total_words) * 100

    # Average sentence length
    sentences = re.split(r'[.!?]+', text)
    sent_lengths = [len(s.split()) for s in sentences if s.strip()]
    avg_sent_len = np.mean(sent_lengths) if sent_lengths else 0

    return {
        "ttr":           round(ttr, 4),
        "avg_word_len":  round(avg_word_len, 2),
        "punct_density": round(punct_density, 4),
        "filler_density":round(filler_density, 2),
        "avg_sent_len":  round(float(avg_sent_len), 2),
    }


def stylometric_to_ai_prob(features: dict) -> float:
    """
    Heuristic rule-based scoring from stylometric features.

    Each feature contributes a partial AI-likelihood score.
    Weights tuned from empirical observation.

    Production upgrade: train a LogisticRegression on labeled data
    and replace this function with model.predict_proba().
    """
    score = 0.0

    # Low TTR → repetitive vocabulary → AI-like
    ttr = features["ttr"]
    if ttr < 0.4:   score += 0.25
    elif ttr < 0.6: score += 0.12

    # Long average word length → formal → slightly AI-like
    awl = features["avg_word_len"]
    if awl > 5.5:   score += 0.10
    elif awl > 4.5: score += 0.05

    # Low punctuation density → smooth, clause-light → AI
    pd = features["punct_density"]
    if pd < 0.03:   score += 0.15
    elif pd < 0.05: score += 0.07

    # High filler density → strong AI signal
    fd = features["filler_density"]
    if fd > 2.0:    score += 0.35
    elif fd > 1.0:  score += 0.20
    elif fd > 0.3:  score += 0.10

    # Uniform sentence length → already captured in burstiness; mild here
    asl = features["avg_sent_len"]
    if 18 <= asl <= 28: score += 0.15  # AI sweet spot
    elif 15 <= asl < 18: score += 0.07

    return float(np.clip(score, 0.0, 1.0))


# ─── File Text Extractors ─────────────────────────────────────────────────────

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract plain text from a PDF using pdfminer.six."""
    try:
        from pdfminer.high_level import extract_text as pdf_extract
        return pdf_extract(io.BytesIO(file_bytes))
    except ImportError:
        raise ImportError("pdfminer.six is required for PDF support. Run: pip install pdfminer.six")
    except Exception as e:
        raise ValueError(f"Could not extract text from PDF: {e}")


def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract plain text from a .docx file using python-docx."""
    try:
        import docx
        doc = docx.Document(io.BytesIO(file_bytes))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except ImportError:
        raise ImportError("python-docx is required for DOCX support. Run: pip install python-docx")
    except Exception as e:
        raise ValueError(f"Could not extract text from DOCX: {e}")


def extract_text_from_file(file_bytes: bytes, filename: str) -> str:
    """Route text extraction based on file extension."""
    ext = os.path.splitext(filename)[-1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_bytes)
    elif ext in (".docx", ".doc"):
        return extract_text_from_docx(file_bytes)
    elif ext == ".txt":
        return file_bytes.decode("utf-8", errors="replace")
    else:
        raise ValueError(f"Unsupported file type: {ext}. Supported: .txt, .pdf, .docx")


# ─── Main Detection Function ──────────────────────────────────────────────────

def detect_ai_text(text: str) -> dict:
    """
    Detect whether a piece of text was written by an AI.

    Args:
        text: Plain text string to analyze (already extracted if from file).

    Returns:
        dict with:
            prediction        – "AI Generated" or "Likely Human Written"
            confidence_score  – 0–100% AI likelihood
            signals           – per-signal breakdown
            word_count        – number of words
            char_count        – number of characters
            excerpt           – first 200 characters of input
    """
    # ── Basic cleanup ──
    text = text.strip()
    words = text.split()
    word_count = len(words)

    if word_count < MIN_WORDS:
        raise ValueError(
            f"Text too short ({word_count} words). "
            f"Minimum {MIN_WORDS} words required for reliable detection."
        )

    logger.info(f"Analyzing text: {word_count} words")

    # ── Signal 1: Perplexity ──
    perplexity = compute_perplexity(text)
    perp_prob  = perplexity_to_ai_prob(perplexity)
    logger.info(f"Perplexity: {perplexity} → AI prob: {perp_prob:.3f}")

    # ── Signal 2: Burstiness ──
    burstiness = compute_burstiness(text)
    burst_prob  = burstiness_to_ai_prob(burstiness)
    logger.info(f"Burstiness: {burstiness} → AI prob: {burst_prob:.3f}")

    # ── Signal 3: Stylometrics ──
    features   = extract_stylometric_features(text)
    style_prob  = stylometric_to_ai_prob(features)
    logger.info(f"Stylometric AI prob: {style_prob:.3f}")

    # ── Ensemble ──
    ai_prob = (
        W_PERPLEXITY  * perp_prob  +
        W_BURSTINESS  * burst_prob +
        W_STYLOMETRIC * style_prob
    )
    confidence = round(ai_prob * 100, 2)

    # ── Decision ──
    if confidence >= 60:
        prediction = "AI Generated"
    elif confidence >= 40:
        prediction = "Uncertain / Possibly AI Generated"
    else:
        prediction = "Likely Human Written"

    return {
        "prediction":       prediction,
        "confidence_score": confidence,
        "signals": {
            "perplexity_score":  perplexity,
            "perplexity_ai_prob": round(perp_prob * 100, 2),
            "burstiness_score":  burstiness,
            "burstiness_ai_prob": round(burst_prob * 100, 2),
            "stylometric_ai_prob": round(style_prob * 100, 2),
            "stylometric_features": features,
        },
        "word_count": word_count,
        "char_count": len(text),
        "excerpt":    text[:200] + ("..." if len(text) > 200 else ""),
    }
