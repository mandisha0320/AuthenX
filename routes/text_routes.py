"""
authenx/routes/text_routes.py

REST API route: POST /verify/headline

Accepts JSON body with "headline" field.
Verifies against web search results using BERT cosine similarity.
"""

import logging
from datetime import datetime, timezone

from flask import Blueprint, request, jsonify, current_app

from services.verification_service import verify_headline
from utils.preprocessing import validate_text_input

logger = logging.getLogger("authenx.text_routes")

text_bp = Blueprint("text", __name__)


@text_bp.route("/verify/headline", methods=["POST"])
def verify_headline_endpoint():
    """
    POST /verify/headline

    Expects JSON body:
        { "headline": "Breaking: Scientists discover new planet..." }

    Returns JSON:
        {
            "prediction": "Likely Misinformation",
            "confidence_score": 68.2,
            "headline": "Breaking: Scientists...",
            "sources_checked": [
                {
                    "title": "...",
                    "snippet": "...",
                    "url": "https://...",
                    "similarity_score": 72.3
                }
            ],
            "avg_similarity": 0.3412,
            "timestamp": "2024-01-01T12:00:00Z"
        }
    """
    # ── Parse JSON body ──
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON with Content-Type: application/json"}), 400

    raw_headline = data.get("headline", "")

    # ── Validate text input ──
    try:
        headline = validate_text_input(raw_headline, max_len=1000)
    except ValueError as e:
        return jsonify({"error": str(e)}), 422

    # ── Run headline verification ──
    try:
        result = verify_headline(headline)
    except ValueError as e:
        return jsonify({"error": str(e)}), 422
    except Exception as e:
        logger.exception("Headline verification failed")
        return jsonify({"error": f"Verification failed: {str(e)}"}), 500

    # ── Persist to MongoDB ──
    timestamp = datetime.now(timezone.utc).isoformat()
    try:
        mongo = current_app.extensions.get("mongo")
        if mongo:
            mongo.db.detections.insert_one({
                "type":             "text",
                "headline":         result["headline"],
                "prediction":       result["prediction"],
                "confidence_score": result["confidence_score"],
                "avg_similarity":   result.get("avg_similarity", 0.0),
                "sources_count":    len(result.get("sources_checked", [])),
                "timestamp":        timestamp,
            })
    except Exception as db_err:
        logger.warning(f"MongoDB write failed (non-fatal): {db_err}")

    # ── Return response ──
    return jsonify({
        "prediction":       result["prediction"],
        "confidence_score": result["confidence_score"],
        "headline":         result["headline"],
        "sources_checked":  result.get("sources_checked", []),
        "avg_similarity":   result.get("avg_similarity", 0.0),
        "timestamp":        timestamp,
    }), 200
