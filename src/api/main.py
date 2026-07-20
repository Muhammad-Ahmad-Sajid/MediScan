import sys
from pathlib import Path
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Add project root to python path to resolve src imports
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.config import (
    UPLOAD_FOLDER,
    HEATMAP_OUTPUT_FOLDER,
    MODEL_CHECKPOINT_PATH,
    DEBUG,
)
from src.model_training.model import FractureModel
from src.api.routes import auth, records, detection, prognosis

# Initialize FastAPI App
app = FastAPI(
    title="Bone Fracture Detection & Prognosis API",
    description="Backend API serving PyTorch fracture classification, Grad-CAM overlays, and prognosis rules.",
    version="1.0.0",
)

# ------------------------------------------------------------------------------
# Middleware Configurations
# ------------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permits requests from dashboard
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------------------
# PyTorch Model Lifecycle Handling
# ------------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FractureModel(pretrained=False)


# Load weights on startup
@app.on_event("startup")
def load_ml_model():
    """Loads saved PyTorch weights into memory at startup to speed up inference."""
    try:
        if MODEL_CHECKPOINT_PATH.exists():
            checkpoint = torch.load(MODEL_CHECKPOINT_PATH, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"[*] PyTorch Model loaded successfully from: {MODEL_CHECKPOINT_PATH}")
        else:
            print(f"[!] Warning: Pretrained checkpoint not found at: {MODEL_CHECKPOINT_PATH}")
            print("[*] Running API with randomly initialized weights for debugging.")

        model.to(device)
        model.eval()  # Crucial: set model to evaluation mode

        # Save model and device in app state for access in endpoints
        app.state.model = model
        app.state.device = device
        print(f"[*] Model successfully bound to device: {device}")
    except Exception as e:
        print(f"[X] CRITICAL: Failed to load PyTorch model weights: {e}")


# ------------------------------------------------------------------------------
# Static Files & Directory Mounting
# ------------------------------------------------------------------------------
# Ensure directories exist
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
HEATMAP_OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# Mount asset directories so files can be accessed via URL (e.g. http://localhost:8000/uploads/scan.png)
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_FOLDER)), name="uploads")
app.mount("/heatmaps", StaticFiles(directory=str(HEATMAP_OUTPUT_FOLDER)), name="heatmaps")

# Mount frontend static folder (css/js) if it exists
static_dir = Path("src/frontend/static")
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# ------------------------------------------------------------------------------
# Routers Registration
# ------------------------------------------------------------------------------
app.include_router(auth.router)
app.include_router(records.router)
app.include_router(detection.router)
app.include_router(prognosis.router)


# ------------------------------------------------------------------------------
# Frontend Integration Endpoint
# ------------------------------------------------------------------------------
@app.get("/")
def serve_dashboard():
    """Serves the plain HTML dashboard at the root URL."""
    html_path = Path("src/frontend/templates/index.html")
    if html_path.exists():
        return FileResponse(html_path)
    return {
        "message": "Welcome to the Bone Fracture Detection API. Frontend templates folder not found.",
        "docs_url": "/docs",
    }


if __name__ == "__main__":
    import uvicorn

    # Start uvicorn server on port 8000
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=DEBUG)
