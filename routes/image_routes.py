"""
authenx/routes/image_routes.py
POST /detect/image

Fix: Strict file-present + non-empty guard before ANY processing.
     Returns 400 immediately if no file is attached — model never runs.
"""

import logging
from datetime import datetime, timezone

from flask import Blueprint, request, jsonify, current_app, g

from models.image_model import predict_image
from utils.preprocessing import validate_image_file
from utils.auth_required import login_required   # ← JWT guard (optional)

logger = logging.getLogger("authenx.image_routes")
image_bp = Blueprint("image", __name__)


@image_bp.route("/detect/image", methods=["POST"])
def detect_image():
    # ── 1. File presence — MUST be first, before anything else ──
    if "file" not in request.files:
        return jsonify({"error": "No file attached. Send a multipart/form-data request with field 'file'."}), 400

    file = request.files["file"]

    # Empty filename = user submitted the form without choosing a file
    if not file or not file.filename or file.filename.strip() == "":
        return jsonify({"error": "No file selected. Please choose an image before submitting."}), 400

    # ── 2. Validate type + magic bytes ──
    try:
        image_bytes = validate_image_file(file)
    except ValueError as e:
        return jsonify({"error": str(e)}), 422

    # ── 3. Guard: reject zero-byte files ──
    if len(image_bytes) == 0:
        return jsonify({"error": "Uploaded file is empty."}), 422

    # ── 4. Run model inference ──
    try:
        result = predict_image(image_bytes)
    except Exception as e:
        logger.exception("Image prediction failed")
        return jsonify({"error": f"Detection failed: {str(e)}"}), 500

    # ── 5. Persist to MongoDB (includes user_id if logged in) ──
    timestamp = datetime.now(timezone.utc).isoformat()
    user_id = getattr(g, "user_id", None)
    try:
        mongo = current_app.extensions.get("mongo")
        if mongo:
            mongo.db.detections.insert_one({
                "type":             "image",
                "filename":         file.filename,
                "prediction":       result["prediction"],
                "confidence_score": result["confidence_score"],
                "timestamp":        timestamp,
                "user_id":          user_id,
            })
    except Exception as db_err:
        logger.warning(f"MongoDB write failed (non-fatal): {db_err}")

    return jsonify({
        "prediction":       result["prediction"],
        "confidence_score": result["confidence_score"],
        "filename":         file.filename,
        "raw_probs":        result.get("raw_probs", {}),
        "timestamp":        timestamp,
    }), 200
