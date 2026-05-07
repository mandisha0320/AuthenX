"""
authenx/services/verification_service.py

News Headline Verification Service.

Workflow:
    1. Accept a news headline (text string)
    2. Fetch top-N web search results via search_service
    3. Encode the headline and each result snippet using sentence-transformers
    4. Compute cosine similarity between headline embedding and each snippet
    5. Average the similarity scores and map to an authenticity confidence %
    6. Return structured result with prediction label and sources

Similarity → Authenticity mapping:
    High similarity (>= 0.6): headline aligns with known sources → Likely Real
    Medium similarity (0.3-0.6): partial match → Unverified
    Low similarity (< 0.3): headline not corroborated → Likely Misinformation
"""

import os
import logging
import re
import html as html_lib
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from services.search_service import fetch_search_results, build_corpus_texts

logger = logging.getLogger("authenx.verification_service")

# ─── Constants ─────────────────────────────────────────────────────────────────

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
MAX_HEADLINE_LEN = 512   # characters; sanitization cap
SEARCH_RESULTS_N = 5

# Thresholds for mapping cosine similarity → label
THRESHOLD_REAL = 0.55     # avg similarity above this → Likely Real
THRESHOLD_UNCERTAIN = 0.30  # above this → Unverified; below → Misinformation


# ─── Singleton BERT Embedder ──────────────────────────────────────────────────

_embedder: Optional[SentenceTransformer] = None


def get_embedder() -> SentenceTransformer:
    """Lazy-load the sentence-transformer model once."""
    global _embedder
    if _embedder is None:
        logger.info(f"Loading sentence-transformer: {EMBED_MODEL_NAME}")
        _embedder = SentenceTransformer(EMBED_MODEL_NAME)
        logger.info("Sentence-transformer loaded.")
    return _embedder


# ─── Text Sanitization ────────────────────────────────────────────────────────

def sanitize_headline(text: str) -> str:
    """
    Sanitize user-supplied headline text.
    - Decode HTML entities
    - Strip HTML tags
    - Normalize whitespace
    - Truncate to MAX_HEADLINE_LEN characters
    """
    text = html_lib.unescape(text)
    text = re.sub(r"<[^>]+>", "", text)           # Strip HTML tags
    text = re.sub(r"[^\w\s.,!?'\"-]", " ", text)  # Keep safe chars
    text = re.sub(r"\s+", " ", text).strip()       # Collapse whitespace
    return text[:MAX_HEADLINE_LEN]


# ─── Similarity Scoring ───────────────────────────────────────────────────────

def _embed_texts(texts: list[str]) -> np.ndarray:
    """Encode a list of strings to embeddings using sentence-transformers."""
    embedder = get_embedder()
    return embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)


def _cosine_sim_score(headline_emb: np.ndarray, corpus_embs: np.ndarray) -> list[float]:
    """
    Compute cosine similarity between headline embedding and each corpus embedding.
    Both should be L2-normalised (normalize_embeddings=True → dot product = cosine sim).
    """
    sims = cosine_similarity(headline_emb.reshape(1, -1), corpus_embs)[0]
    return sims.tolist()


def _map_similarity_to_label(avg_sim: float) -> tuple[str, float]:
    """
    Map average cosine similarity [0,1] to:
        - Human-readable label
        - Authenticity confidence percentage (0-100)

    Mapping logic:
        sim >= THRESHOLD_REAL      → "Likely Real News",      confidence high
        sim >= THRESHOLD_UNCERTAIN → "Unverified",            confidence medium
        sim <  THRESHOLD_UNCERTAIN → "Likely Misinformation", confidence low
    """
    if avg_sim >= THRESHOLD_REAL:
        label = "Likely Real News"
        # Map [THRESHOLD_REAL, 1.0] → [70, 100]
        confidence = 70 + (avg_sim - THRESHOLD_REAL) / (1.0 - THRESHOLD_REAL) * 30
    elif avg_sim >= THRESHOLD_UNCERTAIN:
        label = "Unverified / Inconclusive"
        # Map [THRESHOLD_UNCERTAIN, THRESHOLD_REAL) → [30, 70)
        confidence = 30 + (avg_sim - THRESHOLD_UNCERTAIN) / (THRESHOLD_REAL - THRESHOLD_UNCERTAIN) * 40
    else:
        label = "Likely Misinformation"
        # Map [0, THRESHOLD_UNCERTAIN) → [0, 30)
        confidence = (avg_sim / THRESHOLD_UNCERTAIN) * 30

    return label, round(min(max(confidence, 0.0), 100.0), 2)


# ─── Main Verification Function ───────────────────────────────────────────────

def verify_headline(raw_headline: str) -> dict:
    """
    Verify a news headline against web search results.

    Args:
        raw_headline: User-submitted headline string (unsanitized).

    Returns:
        dict with keys:
            prediction        (str)   – e.g. "Likely Misinformation"
            confidence_score  (float) – 0–100 authenticity score
            headline          (str)   – sanitized input headline
            sources_checked   (list)  – [{title, snippet, url, similarity_score}, ...]
            avg_similarity    (float) – raw cosine similarity average
    """
    # ── Sanitize input ──
    headline = sanitize_headline(raw_headline)
    if not headline:
        raise ValueError("Headline is empty or contains only invalid characters.")

    logger.info(f"Verifying headline: {headline!r}")

    # ── Web Search ──
    raw_results = fetch_search_results(headline, num_results=SEARCH_RESULTS_N)
    corpus_texts = build_corpus_texts(raw_results)

    if not corpus_texts:
        logger.warning("No search results returned; defaulting to low-confidence response.")
        return {
            "prediction": "Unverified / No Sources Found",
            "confidence_score": 0.0,
            "headline": headline,
            "sources_checked": [],
            "avg_similarity": 0.0,
        }

    # ── Embed headline and corpus ──
    all_texts = [headline] + corpus_texts
    embeddings = _embed_texts(all_texts)

    headline_emb = embeddings[0]
    corpus_embs  = embeddings[1:]

    # ── Compute per-source similarities ──
    similarities = _cosine_sim_score(headline_emb, corpus_embs)
    avg_sim = float(np.mean(similarities))

    # ── Build label + confidence ──
    prediction, confidence = _map_similarity_to_label(avg_sim)

    # ── Annotate sources with similarity scores ──
    sources_checked = []
    for i, result in enumerate(raw_results[:len(corpus_texts)]):
        sources_checked.append({
            "title":            result.get("title", ""),
            "snippet":          result.get("snippet", ""),
            "url":              result.get("url", ""),
            "similarity_score": round(similarities[i] * 100, 2),
        })

    return {
        "prediction":       prediction,
        "confidence_score": confidence,
        "headline":         headline,
        "sources_checked":  sources_checked,
        "avg_similarity":   round(avg_sim, 4),
    }
