import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
SECRET_KEY = os.getenv("SECRET_KEY", "fallback-secret-key-for-dev")
UPLOAD_DIR = "uploads"
HEATMAP_DIR = "heatmaps"
REPORT_DIR = "reports"
CHECKPOINT_DIR = "checkpoints"

# Ensure directories exist
for d in [UPLOAD_DIR, HEATMAP_DIR, REPORT_DIR, CHECKPOINT_DIR]:
    os.makedirs(d, exist_ok=True)
