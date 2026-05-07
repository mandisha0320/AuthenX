# ─────────────────────────────────────────────────────────────
#  AuthenX Dockerfile
#  Build: docker build -t authenx .
#  Run:   docker run -p 5000:5000 --env-file .env authenx
#  GPU:   docker run --gpus all -p 5000:5000 --env-file .env authenx
# ─────────────────────────────────────────────────────────────

FROM python:3.11-slim

# System dependencies (libmagic for python-magic, libgl for OpenCV)
RUN apt-get update && apt-get install -y \
    libmagic1 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create checkpoints directory (models mounted via volume in prod)
RUN mkdir -p checkpoints

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
  CMD curl -f http://localhost:5000/health || exit 1

# Run with gunicorn (4 workers)
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "--timeout", "120", "app:app"]
