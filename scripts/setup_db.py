"""
authenx/scripts/setup_db.py

One-time script to create MongoDB indexes for the authenx database.

Run: python scripts/setup_db.py
"""

import os
from pymongo import MongoClient, ASCENDING, DESCENDING
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/authenx")

client = MongoClient(MONGO_URI)
db = client.get_database()

# ── Index on detections collection ────────────────────────────
detections = db["detections"]

# Index for filtering by type (image/video/text)
detections.create_index([("type", ASCENDING)], name="idx_type")

# Index for recent results (dashboard queries)
detections.create_index([("timestamp", DESCENDING)], name="idx_timestamp_desc")

# Compound index for type + time (common query pattern)
detections.create_index(
    [("type", ASCENDING), ("timestamp", DESCENDING)],
    name="idx_type_timestamp"
)

print("✅ MongoDB indexes created successfully.")
print(f"   Database: {db.name}")
print(f"   Collection: detections")
print(f"   Indexes: {list(detections.index_information().keys())}")
