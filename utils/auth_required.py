"""
authenx/utils/auth_required.py

JWT authentication decorator.
Attach to any route with @login_required to require a valid token.
User id is available inside the route as g.user_id.
"""

import os
from functools import wraps
from flask import request, jsonify, g
import jwt

JWT_SECRET = os.getenv("JWT_SECRET", "authenx-jwt-secret-change-in-prod")
JWT_ALGORITHM = "HS256"


def login_required(f):
    """Decorator: require a valid Bearer JWT token in Authorization header."""
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return jsonify({"error": "Missing or invalid Authorization header."}), 401
        token = auth_header.split(" ", 1)[1]
        try:
            payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
            g.user_id = payload["user_id"]
            g.username = payload.get("username", "")
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token expired. Please log in again."}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "Invalid token."}), 401
        return f(*args, **kwargs)
    return decorated


def optional_auth(f):
    """Decorator: parse JWT if present but don't reject if missing."""
    @wraps(f)
    def decorated(*args, **kwargs):
        g.user_id = None
        g.username = ""
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header.split(" ", 1)[1]
            try:
                payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
                g.user_id = payload["user_id"]
                g.username = payload.get("username", "")
            except jwt.InvalidTokenError:
                pass
        return f(*args, **kwargs)
    return decorated
