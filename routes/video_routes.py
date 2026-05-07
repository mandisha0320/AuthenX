"""
authenx/routes/video_routes.py
POST /detect/video

Fix: Same strict file guard as image route.
"""

import logging
from datetime import datetime, timezone

from flask import Blueprint, request, jsonify, current_app, g

from models.video_model import predict_video
from utils.preprocessing import validate_video_file

logger = logging.getLogger("authenx.video_routes")
video_bp = Blueprint("video", __name__)


@video_bp.route("/detect/video", methods=["POST"])
def detect_video():
    # ── 1. Strict file guard ──
    if "file" not in request.files:
        return jsonify({"error": "No file attached. Send a multipart/form-data request with field 'file'."}), 400

    file = request.files["file"]

    if not file or not file.filename or file.filename.strip() == "":
        return jsonify({"error": "No file selected. Please choose a video before submitting."}), 400

    # ── 2. Validate ──
    try:
        video_bytes, original_filename = validate_video_file(file)
    except ValueError as e:
        return jsonify({"error": str(e)}), 422

    if len(video_bytes) == 0:
        return jsonify({"error": "Uploaded video file is empty."}), 422

    # ── 3. Run inference ──
    try:
        result = predict_video(video_bytes, original_filename=original_filename)
    except Exception as e:
        logger.exception("Video prediction failed")
        return jsonify({"error": f"Detection failed: {str(e)}"}), 500

    # ── 4. Persist ──
    timestamp = datetime.now(timezone.utc).isoformat()
    user_id = getattr(g, "user_id", None)
    try:
        mongo = current_app.extensions.get("mongo")
        if mongo:
            mongo.db.detections.insert_one({
                "type":             "video",
                "filename":         original_filename,
                "prediction":       result["prediction"],
                "confidence_score": result["confidence_score"],
                "frames_analyzed":  result["frames_analyzed"],
                "timestamp":        timestamp,
                "user_id":          user_id,
            })
    except Exception as db_err:
        logger.warning(f"MongoDB write failed (non-fatal): {db_err}")

    return jsonify({
        "prediction":       result["prediction"],
        "confidence_score": result["confidence_score"],
        "frames_analyzed":  result["frames_analyzed"],
        "per_frame_scores": result.get("per_frame_scores", []),
        "filename":         original_filename,
        "timestamp":        timestamp,
    }), 200
