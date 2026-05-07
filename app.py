"""
AuthenX — Flask application entry point (with Auth)
"""

import os
import logging
from flask import Flask, jsonify, send_from_directory
from flask_pymongo import PyMongo
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("authenx")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

app.config["MAX_CONTENT_LENGTH"]        = int(os.getenv("MAX_UPLOAD_MB", 500)) * 1024 * 1024
app.config["MONGO_URI"]                 = os.getenv("MONGO_URI", "mongodb://localhost:27017/authenx")
app.config["SECRET_KEY"]                = os.getenv("SECRET_KEY", "dev-secret-change-in-production")
app.config["ALLOWED_IMAGE_EXTENSIONS"]  = {"jpg", "jpeg", "png", "webp", "bmp"}
app.config["ALLOWED_VIDEO_EXTENSIONS"]  = {"mp4", "avi", "mov", "mkv", "webm"}
app.config["ALLOWED_DOCUMENT_EXTENSIONS"] = {"txt", "pdf", "docx"}

mongo = PyMongo(app)
app.extensions["mongo"] = mongo

# ── Blueprints ──
from routes.image_routes    import image_bp
from routes.video_routes    import video_bp
from routes.text_routes     import text_bp
from routes.document_routes import document_bp
from routes.auth_routes     import auth_bp       # ← NEW

app.register_blueprint(image_bp)
app.register_blueprint(video_bp)
app.register_blueprint(text_bp)
app.register_blueprint(document_bp)
app.register_blueprint(auth_bp)                  # ← NEW  (prefix: /auth)

# ── Static pages ──
@app.route("/", methods=["GET"])
def dashboard():
    return send_from_directory(".", "authenx_dashboard.html")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "AuthenX"}), 200

# ── Error handlers ──
@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large."}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found."}), 404

@app.errorhandler(500)
def internal(e):
    logger.exception("Unhandled error")
    return jsonify({"error": "Internal server error."}), 500

if __name__ == "__main__":
    port  = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    logger.info(f"Starting AuthenX on port {port}")
    app.run(host="0.0.0.0", port=port, debug=debug)
