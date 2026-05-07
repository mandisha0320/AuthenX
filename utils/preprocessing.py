"""
authenx/utils/preprocessing.py

Shared preprocessing utilities for image and video inputs.

Provides:
    - File type validation (by extension AND magic bytes)
    - File size checking
    - Image preprocessing helpers
    - Helper to read uploaded Flask file to bytes safely
"""

import os
import io
import magic  # python-magic for MIME type detection from bytes
from flask import current_app
from werkzeug.datastructures import FileStorage
from PIL import Image


# ─── Magic Byte MIME Validation ───────────────────────────────────────────────

# Maps allowed MIME types to their category
ALLOWED_IMAGE_MIMES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/gif",
}

ALLOWED_VIDEO_MIMES = {
    "video/mp4",
    "video/x-msvideo",    # .avi
    "video/quicktime",    # .mov
    "video/x-matroska",   # .mkv
    "video/webm",
}


def get_file_extension(filename: str) -> str:
    """Return lowercase file extension without dot."""
    return os.path.splitext(filename or "")[-1].lstrip(".").lower()


def validate_image_file(file: FileStorage) -> bytes:
    """
    Validate that an uploaded file is a legitimate image.

    Checks:
        1. Filename has an allowed extension
        2. File content passes magic byte MIME detection
        3. PIL can open and decode the image

    Args:
        file: Werkzeug FileStorage object from request.files

    Returns:
        Raw bytes of the valid image.

    Raises:
        ValueError: If any validation step fails.
    """
    # Read file content
    data = file.read()
    if not data:
        raise ValueError("Uploaded file is empty.")

    # Extension check
    ext = get_file_extension(file.filename)
    allowed_exts = current_app.config.get("ALLOWED_IMAGE_EXTENSIONS", set())
    if ext not in allowed_exts:
        raise ValueError(f"File extension '.{ext}' is not allowed. Allowed: {allowed_exts}")

    # Magic byte MIME check (more reliable than extension alone)
    mime = magic.from_buffer(data, mime=True)
    if mime not in ALLOWED_IMAGE_MIMES:
        raise ValueError(f"File MIME type '{mime}' is not a supported image format.")

    # PIL decode check
    try:
        img = Image.open(io.BytesIO(data))
        img.verify()  # Verifies integrity without fully loading
    except Exception as e:
        raise ValueError(f"Image file appears corrupted: {e}")

    return data


def validate_video_file(file: FileStorage) -> tuple[bytes, str]:
    """
    Validate that an uploaded file is a legitimate video.

    Args:
        file: Werkzeug FileStorage object from request.files

    Returns:
        Tuple of (raw bytes, original filename).

    Raises:
        ValueError: If any validation step fails.
    """
    # Read file content
    data = file.read()
    if not data:
        raise ValueError("Uploaded video file is empty.")

    # Extension check
    ext = get_file_extension(file.filename)
    allowed_exts = current_app.config.get("ALLOWED_VIDEO_EXTENSIONS", set())
    if ext not in allowed_exts:
        raise ValueError(f"File extension '.{ext}' is not allowed. Allowed: {allowed_exts}")

    # Magic byte MIME check
    mime = magic.from_buffer(data, mime=True)
    if mime not in ALLOWED_VIDEO_MIMES:
        raise ValueError(f"File MIME type '{mime}' is not a supported video format.")

    return data, file.filename or f"video.{ext}"


def validate_text_input(text: str, max_len: int = 1000) -> str:
    """
    Validate and sanitize a text input field.

    Args:
        text:    Raw string from request JSON.
        max_len: Maximum allowed character length.

    Returns:
        Stripped string.

    Raises:
        ValueError: If text is empty or too long.
    """
    if not isinstance(text, str):
        raise ValueError("Text input must be a string.")

    text = text.strip()
    if not text:
        raise ValueError("Text input cannot be empty.")
    if len(text) > max_len:
        raise ValueError(f"Text input exceeds maximum length of {max_len} characters.")

    return text


def validate_document_file(file) -> tuple:
    """
    Validate an uploaded document file (.txt, .pdf, .docx).

    Args:
        file: Werkzeug FileStorage object from request.files

    Returns:
        Tuple of (raw bytes, filename, extension).

    Raises:
        ValueError: If validation fails.
    """
    import os
    data = file.read()
    if not data:
        raise ValueError("Uploaded document is empty.")

    ext = os.path.splitext(file.filename or "")[-1].lower()
    allowed = {".txt", ".pdf", ".docx"}
    if ext not in allowed:
        raise ValueError(f"File type '{ext}' not supported. Allowed: .txt, .pdf, .docx")

    # Size guard: 20 MB max for documents
    max_bytes = 20 * 1024 * 1024
    if len(data) > max_bytes:
        raise ValueError("Document exceeds 20 MB limit.")

    return data, file.filename or f"document{ext}", ext
