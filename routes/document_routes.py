"""
authenx/routes/document_routes.py

REST API routes for AI-Generated TEXT Detection.

Endpoints:
    POST /detect/text      – plain text input (JSON body)
    POST /detect/document  – file upload (.txt / .pdf / .docx)

These are DIFFERENT from /verify/headline:
    /verify/headline  → checks if a NEWS HEADLINE is misinformation (web search)
    /detect/text      → checks if TEXT was WRITTEN BY AN AI (GPT-2 + stylometrics)
"""

import logging
from datetime import datetime, timezone

from flask import Blueprint, request, jsonify, current_app

from models.text_detection_model import (
    detect_ai_text,
    extract_text_from_file,
)
from utils.preprocessing import validate_text_input

logger = logging.getLogger("authenx.document_routes")

document_bp = Blueprint("document", __name__)

# Allowed file extensions for document upload
ALLOWED_DOC_EXTENSIONS = {".txt", ".pdf", ".docx"}


# ─── Helper: Save to MongoDB ───────────────────────────────────────────────────

def _save_to_db(record: dict):
    try:
        mongo = current_app.extensions.get("mongo")
        if mongo:
            mongo.db.detections.insert_one(record)
    except Exception as e:
        logger.warning(f"MongoDB write failed (non-fatal): {e}")


# ─── POST /detect/text ─────────────────────────────────────────────────────────

@document_bp.route("/detect/text", methods=["POST"])
def detect_text():
    """
    POST /detect/text

    Detect whether a piece of text was written by an AI.

    Request body (JSON):
        { "text": "The emergence of large language models has..." }

    Response:
        {
            "prediction":       "AI Generated",
            "confidence_score": 78.4,
            "signals": {
                "perplexity_score":   23.4,
                "perplexity_ai_prob": 88.2,
                "burstiness_score":   0.18,
                "burstiness_ai_prob": 82.0,
                "stylometric_ai_prob": 55.0,
                "stylometric_features": { ... }
            },
            "word_count":  542,
            "char_count":  3201,
            "excerpt":     "The emergence of...",
            "timestamp":   "2025-01-15T10:30:00Z"
        }
    """
    # ── Parse JSON ──
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON with Content-Type: application/json"}), 400

    raw_text = data.get("text", "")

    # ── Validate ──
    try:
        text = validate_text_input(raw_text, max_len=50_000)
    except ValueError as e:
        return jsonify({"error": str(e)}), 422

    # ── Run detection ──
    try:
        result = detect_ai_text(text)
    except ValueError as e:
        return jsonify({"error": str(e)}), 422
    except Exception as e:
        logger.exception("Text AI detection failed")
        return jsonify({"error": f"Detection failed: {str(e)}"}), 500

    # ── Persist ──
    timestamp = datetime.now(timezone.utc).isoformat()
    _save_to_db({
        "type":             "text_detection",
        "input_source":     "raw_text",
        "prediction":       result["prediction"],
        "confidence_score": result["confidence_score"],
        "word_count":       result["word_count"],
        "timestamp":        timestamp,
    })

    return jsonify({**result, "timestamp": timestamp}), 200


# ─── POST /detect/document ─────────────────────────────────────────────────────

@document_bp.route("/detect/document", methods=["POST"])
def detect_document():
    """
    POST /detect/document

    Upload a .txt, .pdf, or .docx file. Text is extracted automatically,
    then run through the AI text detection pipeline.

    Request: multipart/form-data, field: file

    Response: same shape as /detect/text, plus:
        {
            "filename": "essay.pdf",
            "file_type": ".pdf",
            ...
        }
    """
    # ── File presence ──
    if "file" not in request.files:
        return jsonify({
            "error": "No file field in request. Use multipart/form-data with key 'file'."
        }), 400

    file = request.files["file"]

    if not file.filename:
        return jsonify({"error": "No file selected."}), 400

    # ── Extension check ──
    import os
    ext = os.path.splitext(file.filename)[-1].lower()
    if ext not in ALLOWED_DOC_EXTENSIONS:
        return jsonify({
            "error": f"File type '{ext}' not supported. Allowed: .txt, .pdf, .docx"
        }), 422

    # ── Read bytes ──
    file_bytes = file.read()
    if not file_bytes:
        return jsonify({"error": "Uploaded file is empty."}), 422

    # ── Extract text ──
    try:
        text = extract_text_from_file(file_bytes, file.filename)
    except ImportError as e:
        return jsonify({"error": str(e)}), 501   # 501 = feature not implemented (missing dep)
    except ValueError as e:
        return jsonify({"error": str(e)}), 422
    except Exception as e:
        logger.exception("Text extraction from file failed")
        return jsonify({"error": f"Could not extract text: {str(e)}"}), 500

    # ── Validate extracted text ──
    text = text.strip()
    if not text:
        return jsonify({"error": "No readable text found in the uploaded file."}), 422

    # ── Run detection ──
    try:
        result = detect_ai_text(text)
    except ValueError as e:
        return jsonify({"error": str(e)}), 422
    except Exception as e:
        logger.exception("Text AI detection failed")
        return jsonify({"error": f"Detection failed: {str(e)}"}), 500

    # ── Persist ──
    timestamp = datetime.now(timezone.utc).isoformat()
    _save_to_db({
        "type":             "text_detection",
        "input_source":     "file_upload",
        "filename":         file.filename,
        "file_type":        ext,
        "prediction":       result["prediction"],
        "confidence_score": result["confidence_score"],
        "word_count":       result["word_count"],
        "timestamp":        timestamp,
    })

    return jsonify({
        **result,
        "filename":  file.filename,
        "file_type": ext,
        "timestamp": timestamp,
    }), 200
