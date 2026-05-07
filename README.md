# AuthenX 🔍

**Production-Ready AI Content Detection & News Verification Backend**

AuthenX is a modular Flask backend that detects AI-generated images and videos, and verifies news headlines using web search + BERT embeddings. Every result includes a calibrated confidence score (0–100%).

---

## Architecture

```
authenx/
├── app.py                          # Flask app factory, blueprint registration
├── .env.example                    # Environment variable template
├── requirements.txt
│
├── routes/
│   ├── image_routes.py             # POST /detect/image
│   ├── video_routes.py             # POST /detect/video
│   └── text_routes.py              # POST /verify/headline
│
├── models/
│   ├── image_model.py              # EfficientNet-B0 deepfake detector
│   └── video_model.py              # Frame extraction + per-frame scoring
│
├── services/
│   ├── verification_service.py     # Headline ↔ search results BERT similarity
│   └── search_service.py           # DuckDuckGo / SerpAPI web search
│
├── utils/
│   ├── preprocessing.py            # File validation (magic bytes, extension, PIL)
│   └── scoring.py                  # Softmax, cosine→%, aggregation helpers
│
└── scripts/
    └── setup_db.py                 # MongoDB index creation
```

---

## How Each Module Works

### 1. Image Detection (`POST /detect/image`)
- **Model**: EfficientNet-B0 (torchvision, ImageNet pretrained)
- **Head**: Custom `Linear(1280, 2)` for binary Real/Fake classification
- **Confidence**: `softmax(logits)[fake_class] × 100`
- **Production**: Replace `checkpoints/image_deepfake.pth` with a model fine-tuned on [FaceForensics++](https://github.com/ondyari/FaceForensics) or similar

### 2. Video Detection (`POST /detect/video`)
- Extracts **1 frame/second** using OpenCV
- Each frame is processed through the image model in configurable batches
- Final score = **mean fake probability** across all frames
- Graduated thresholds: Highly Likely → Likely → Uncertain → Likely Authentic

### 3. Headline Verification (`POST /verify/headline`)
- Fetches **top 5 search results** via DuckDuckGo (or SerpAPI)
- Encodes headline + snippets with **all-MiniLM-L6-v2** (sentence-transformers)
- **Cosine similarity** between headline and each snippet
- Average similarity mapped to: Likely Real / Unverified / Likely Misinformation

### 4. Confidence Score Logic
| Source        | Method                          | Scale          |
|---------------|---------------------------------|----------------|
| Image/Video   | Softmax probability             | 0–100%         |
| Text/Headline | Cosine similarity (normalized)  | 0–100%         |

---

## Quick Start

### Prerequisites

```bash
# System deps (Ubuntu/Debian)
sudo apt-get install -y libmagic1 python3-pip

# Python 3.10+
python --version
```

### 1. Clone and install

```bash
git clone https://github.com/yourorg/authenx.git
cd authenx

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — at minimum set MONGO_URI if MongoDB is not local
```

### 3. Start MongoDB (Docker)

```bash
docker run -d --name authenx-mongo \
  -p 27017:27017 \
  -v authenx_data:/data/db \
  mongo:7
```

### 4. Create MongoDB indexes

```bash
python scripts/setup_db.py
```

### 5. Run the server

```bash
# Development
python app.py

# Production (4 workers)
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

The API is now available at `http://localhost:5000`

---

## API Reference

### Health Check

```
GET /health
```

Response:
```json
{"status": "ok", "service": "AuthenX"}
```

---

### POST /detect/image

Detect whether an image is AI-generated or authentic.

**Request**: `multipart/form-data`

| Field | Type   | Required | Description          |
|-------|--------|----------|----------------------|
| file  | binary | ✅       | Image (jpg/png/webp) |

**Response**:
```json
{
  "prediction": "AI Generated",
  "confidence_score": 82.34,
  "filename": "photo.jpg",
  "raw_probs": {
    "authentic": 17.66,
    "ai_generated": 82.34
  },
  "timestamp": "2024-01-15T10:30:00+00:00"
}
```

---

### POST /detect/video

Detect whether a video is AI-generated or authentic.

**Request**: `multipart/form-data`

| Field | Type   | Required | Description              |
|-------|--------|----------|--------------------------|
| file  | binary | ✅       | Video (mp4/avi/mov/mkv)  |

**Response**:
```json
{
  "prediction": "Likely AI Generated",
  "confidence_score": 76.45,
  "frames_analyzed": 32,
  "per_frame_scores": [
    {"frame": 1, "score": 80.12},
    {"frame": 2, "score": 71.34}
  ],
  "filename": "clip.mp4",
  "timestamp": "2024-01-15T10:30:00+00:00"
}
```

---

### POST /verify/headline

Verify whether a news headline is real or misinformation.

**Request**: `application/json`

```json
{ "headline": "Scientists discover water on Mars" }
```

**Response**:
```json
{
  "prediction": "Likely Real News",
  "confidence_score": 84.5,
  "headline": "Scientists discover water on Mars",
  "avg_similarity": 0.7812,
  "sources_checked": [
    {
      "title": "NASA confirms water ice on Mars",
      "snippet": "Researchers at NASA's Jet Propulsion Lab confirmed...",
      "url": "https://nasa.gov/...",
      "similarity_score": 88.3
    }
  ],
  "timestamp": "2024-01-15T10:30:00+00:00"
}
```

---

## Example curl Commands

### Detect Image
```bash
curl -X POST http://localhost:5000/detect/image \
  -F "file=@/path/to/image.jpg"
```

### Detect Video
```bash
curl -X POST http://localhost:5000/detect/video \
  -F "file=@/path/to/video.mp4"
```

### Verify Headline
```bash
curl -X POST http://localhost:5000/verify/headline \
  -H "Content-Type: application/json" \
  -d '{"headline": "Scientists discover water on Mars"}'
```

### Test misinformation detection
```bash
curl -X POST http://localhost:5000/verify/headline \
  -H "Content-Type: application/json" \
  -d '{"headline": "The moon is made of cheese, scientists confirm"}'
```

### Health check
```bash
curl http://localhost:5000/health
```

---

## GPU Support

AuthenX auto-detects CUDA. If a GPU is available, models run on it automatically.

To verify:
```python
import torch
print(torch.cuda.is_available())   # True → GPU in use
print(torch.cuda.get_device_name(0))
```

For GPU Docker deployment:
```bash
docker run --gpus all -p 5000:5000 authenx:latest
```

---

## Fine-tuning the Image Model

For production accuracy, fine-tune on a deepfake dataset:

```python
from models.image_model import DeepfakeImageModel
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

model = DeepfakeImageModel(pretrained=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# Training loop (simplified)
for epoch in range(num_epochs):
    for images, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Save checkpoint
torch.save(model.state_dict(), "checkpoints/image_deepfake.pth")
```

Recommended datasets:
- [FaceForensics++](https://github.com/ondyari/FaceForensics)
- [DeepFake Detection Challenge (DFDC)](https://ai.facebook.com/datasets/dfdc/)
- [Celeb-DF](https://github.com/yuezunli/celeb-deepfakeforensics)

---

## MongoDB Schema

Collection: `detections`

```json
{
  "_id": "ObjectId",
  "type": "image | video | text",
  "filename": "photo.jpg",           // image/video only
  "headline": "...",                 // text only
  "prediction": "AI Generated",
  "confidence_score": 82.34,
  "frames_analyzed": 32,            // video only
  "avg_similarity": 0.78,           // text only
  "sources_count": 5,               // text only
  "timestamp": "2024-01-15T10:30:00+00:00"
}
```

---

## Security Notes

- **File type validation**: Both extension AND magic bytes (libmagic)
- **PIL decode check**: Ensures image is not corrupted/malicious
- **Text sanitization**: HTML entity decoding, tag stripping, length cap
- **Max upload size**: Configurable via `MAX_UPLOAD_MB` (default 500 MB)
- **No eval() / exec()**: No dynamic code execution anywhere in the pipeline
- **Production**: Add authentication middleware (JWT/API key) before deploying publicly

---

## Environment Variables Reference

| Variable                | Default                              | Description                          |
|-------------------------|--------------------------------------|--------------------------------------|
| `PORT`                  | `5000`                               | Server port                          |
| `FLASK_DEBUG`           | `false`                              | Enable debug mode                    |
| `SECRET_KEY`            | `dev-secret-...`                     | Flask secret (change in prod!)       |
| `MONGO_URI`             | `mongodb://localhost:27017/authenx`  | MongoDB connection string            |
| `MAX_UPLOAD_MB`         | `500`                                | Max upload file size in MB           |
| `SEARCH_BACKEND`        | `duckduckgo`                         | `duckduckgo` or `serpapi`            |
| `SERPAPI_KEY`           | *(empty)*                            | SerpAPI key (if using serpapi)       |
| `IMAGE_MODEL_CHECKPOINT`| `checkpoints/image_deepfake.pth`     | Path to fine-tuned model weights     |
| `EMBED_MODEL`           | `all-MiniLM-L6-v2`                   | sentence-transformers model name     |
| `VIDEO_BATCH_SIZE`      | `8`                                  | Frames per GPU batch                 |
| `VIDEO_MAX_FRAMES`      | `120`                                | Max frames sampled per video         |
