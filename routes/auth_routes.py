"""
authenx/routes/auth_routes.py

Authentication endpoints:
    POST /auth/signup   — create account
    POST /auth/login    — get JWT token
    GET  /auth/me       — get current user info (requires token)
    GET  /auth/history  — get current user's detection history (requires token)
    DELETE /auth/history/<id> — delete a single history item
"""

import os
import logging
import re
from datetime import datetime, timezone, timedelta

import jwt
import bcrypt
from bson import ObjectId
from flask import Blueprint, request, jsonify, current_app, g

from utils.auth_required import login_required

logger = logging.getLogger("authenx.auth")
auth_bp = Blueprint("auth", __name__, url_prefix="/auth")

JWT_SECRET    = os.getenv("JWT_SECRET", "authenx-jwt-secret-change-in-prod")
JWT_ALGORITHM = "HS256"
JWT_EXPIRES_HOURS = int(os.getenv("JWT_EXPIRES_HOURS", 72))  # 3 days


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_token(user_id: str, username: str) -> str:
    payload = {
        "user_id":  user_id,
        "username": username,
        "exp":      datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRES_HOURS),
        "iat":      datetime.now(timezone.utc),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def _validate_email(email: str) -> bool:
    return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email))


def _validate_password(password: str) -> str | None:
    """Return error message or None if valid."""
    if len(password) < 8:
        return "Password must be at least 8 characters."
    if not re.search(r"[A-Za-z]", password):
        return "Password must contain at least one letter."
    if not re.search(r"[0-9]", password):
        return "Password must contain at least one number."
    return None


def _user_dict(user: dict) -> dict:
    """Serialize MongoDB user doc for API response (never expose password hash)."""
    return {
        "id":         str(user["_id"]),
        "username":   user["username"],
        "email":      user["email"],
        "created_at": user.get("created_at", ""),
    }


# ─── POST /auth/signup ────────────────────────────────────────────────────────

@auth_bp.route("/signup", methods=["POST"])
def signup():
    """
    Create a new account.
    Body: { "username": "alice", "email": "alice@example.com", "password": "Secret123" }
    """
    data = request.get_json(silent=True) or {}
    username = data.get("username", "").strip()
    email    = data.get("email", "").strip().lower()
    password = data.get("password", "")

    # ── Validate ──
    if not username or len(username) < 3:
        return jsonify({"error": "Username must be at least 3 characters."}), 422
    if not re.match(r"^[a-zA-Z0-9_]+$", username):
        return jsonify({"error": "Username can only contain letters, numbers and underscores."}), 422
    if not _validate_email(email):
        return jsonify({"error": "Invalid email address."}), 422
    pwd_err = _validate_password(password)
    if pwd_err:
        return jsonify({"error": pwd_err}), 422

    mongo = current_app.extensions["mongo"]

    # ── Check uniqueness ──
    if mongo.db.users.find_one({"username": {"$regex": f"^{re.escape(username)}$", "$options": "i"}}):
        return jsonify({"error": "Username already taken."}), 409
    if mongo.db.users.find_one({"email": email}):
        return jsonify({"error": "An account with this email already exists."}), 409

    # ── Hash password ──
    pw_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

    # ── Insert user ──
    now = datetime.now(timezone.utc).isoformat()
    result = mongo.db.users.insert_one({
        "username":   username,
        "email":      email,
        "password":   pw_hash,
        "created_at": now,
    })

    user_id = str(result.inserted_id)
    token   = _make_token(user_id, username)

    logger.info(f"New user registered: {username} ({email})")

    return jsonify({
        "message": "Account created successfully.",
        "token":   token,
        "user": {
            "id":         user_id,
            "username":   username,
            "email":      email,
            "created_at": now,
        }
    }), 201


# ─── POST /auth/login ─────────────────────────────────────────────────────────

@auth_bp.route("/login", methods=["POST"])
def login():
    """
    Log in with email + password.
    Body: { "email": "alice@example.com", "password": "Secret123" }
    Returns a JWT token.
    """
    data     = request.get_json(silent=True) or {}
    email    = data.get("email", "").strip().lower()
    password = data.get("password", "")

    if not email or not password:
        return jsonify({"error": "Email and password are required."}), 422

    mongo = current_app.extensions["mongo"]
    user  = mongo.db.users.find_one({"email": email})

    if not user or not bcrypt.checkpw(password.encode(), user["password"].encode()):
        return jsonify({"error": "Invalid email or password."}), 401

    user_id = str(user["_id"])
    token   = _make_token(user_id, user["username"])

    logger.info(f"User logged in: {user['username']}")

    return jsonify({
        "message": "Login successful.",
        "token":   token,
        "user":    _user_dict(user),
    }), 200


# ─── GET /auth/me ─────────────────────────────────────────────────────────────

@auth_bp.route("/me", methods=["GET"])
@login_required
def me():
    """Return the currently authenticated user's profile."""
    mongo = current_app.extensions["mongo"]
    user  = mongo.db.users.find_one({"_id": ObjectId(g.user_id)})
    if not user:
        return jsonify({"error": "User not found."}), 404
    return jsonify({"user": _user_dict(user)}), 200


# ─── GET /auth/history ────────────────────────────────────────────────────────

@auth_bp.route("/history", methods=["GET"])
@login_required
def history():
    """
    Return the authenticated user's detection history, newest first.
    Query params:
        limit  (int, default 50)
        type   (str, filter by "image" | "video" | "text_detection" | "text")
    """
    mongo  = current_app.extensions["mongo"]
    limit  = min(int(request.args.get("limit", 50)), 200)
    filter_type = request.args.get("type", "")

    query = {"user_id": g.user_id}
    if filter_type:
        query["type"] = filter_type

    items = list(
        mongo.db.detections
        .find(query, {"_id": 1, "type": 1, "filename": 1, "headline": 1,
                      "prediction": 1, "confidence_score": 1, "timestamp": 1})
        .sort("timestamp", -1)
        .limit(limit)
    )

    # Convert ObjectId to string
    for item in items:
        item["id"] = str(item.pop("_id"))

    return jsonify({"history": items, "count": len(items)}), 200


# ─── DELETE /auth/history/<id> ────────────────────────────────────────────────

@auth_bp.route("/history/<item_id>", methods=["DELETE"])
@login_required
def delete_history_item(item_id: str):
    """Delete a single history entry belonging to the current user."""
    mongo  = current_app.extensions["mongo"]
    try:
        oid = ObjectId(item_id)
    except Exception:
        return jsonify({"error": "Invalid history item ID."}), 422

    result = mongo.db.detections.delete_one({"_id": oid, "user_id": g.user_id})
    if result.deleted_count == 0:
        return jsonify({"error": "Item not found or not yours."}), 404

    return jsonify({"message": "Deleted."}), 200
