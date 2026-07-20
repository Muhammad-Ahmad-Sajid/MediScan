import os
from pathlib import Path
from dotenv import load_dotenv

# Base directory of the project (parent of src/)
BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env file from the project base directory
dotenv_path = BASE_DIR / ".env"
load_dotenv(dotenv_path=dotenv_path)

# ------------------------------------------------------------------------------
# DATABASE SETTINGS
# ------------------------------------------------------------------------------
# PostgreSQL database connection URL (managed via pgAdmin 4)
DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/bone_fracture_db")

# ------------------------------------------------------------------------------
# PATHS AND FOLDERS
# ------------------------------------------------------------------------------
# Upload folder for input X-ray images
UPLOAD_FOLDER_RAW: str = os.getenv("UPLOAD_FOLDER", "uploads/")
UPLOAD_FOLDER: Path = BASE_DIR / UPLOAD_FOLDER_RAW

# File path to the saved PyTorch model weights (.pth)
MODEL_CHECKPOINT_PATH_RAW: str = os.getenv("MODEL_CHECKPOINT_PATH", "checkpoints/best_model.pth")
MODEL_CHECKPOINT_PATH: Path = BASE_DIR / MODEL_CHECKPOINT_PATH_RAW

# Folder for outputting Grad-CAM heatmap visualization images
HEATMAP_OUTPUT_FOLDER_RAW: str = os.getenv("HEATMAP_OUTPUT_FOLDER", "heatmaps/")
HEATMAP_OUTPUT_FOLDER: Path = BASE_DIR / HEATMAP_OUTPUT_FOLDER_RAW

# ------------------------------------------------------------------------------
# APPLICATION CONFIG
# ------------------------------------------------------------------------------
# Debug mode flag
DEBUG: bool = os.getenv("DEBUG", "True").lower() in ("true", "1", "t", "yes")

# ------------------------------------------------------------------------------
# DIRECTORY INITIALIZATION
# ------------------------------------------------------------------------------
# Automatically ensure runtime directories exist
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
HEATMAP_OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
MODEL_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
