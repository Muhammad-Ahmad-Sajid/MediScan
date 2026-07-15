import os
import cv2
import time
import uuid
import logging
from dataclasses import dataclass
from typing import Optional
from datetime import datetime
import numpy as np

import torch
import torch.nn as nn
from torchvision import models, transforms
import mlflow

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ==============================================================================
# CONFIGURATION & LOGGING
# ==============================================================================
BASE_DIR = "d:/X-ray ML Model"
CHECKPOINT_PATH = os.path.join(BASE_DIR, "checkpoints", "osteoporosis_best.pth")
HEATMAP_DIR = os.path.join(BASE_DIR, "heatmaps")
os.makedirs(HEATMAP_DIR, exist_ok=True)

MODEL_VERSION = "osteoporosis_v1"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ==============================================================================
# DATACLASSES
# ==============================================================================
@dataclass
class OsteoporosisResult:
    has_osteoporosis: bool
    confidence: float
    confidence_flag: str
    heatmap_path: Optional[str]
    clinical_recommendation: str
    model_version: str
    prediction_time_ms: float
    message: str

# ==============================================================================
# CLINICAL RECOMMENDATIONS
# ==============================================================================
RECS = {
    0: "Bone density appears normal. Maintain calcium and vitamin D intake. Routine screening in 2 years.",
    1: "Low bone density detected. Urgent referral to endocrinologist or rheumatologist recommended. DEXA scan confirmation advised. Fall prevention measures should be implemented immediately."
}

# ==============================================================================
# MODEL INITIALIZATION (Module Level)
# ==============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_model():
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 1)
    )
    return model

logger.info(f"Loading Osteoporosis model from {CHECKPOINT_PATH} to {device}")
try:
    model = create_model()
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    logger.info("Osteoporosis model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load Osteoporosis model: {e}")
    model = None

# ==============================================================================
# PREPROCESSING
# ==============================================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_image(image_path):
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise ValueError(f"Failed to load image: {image_path}")

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_gray)
    img_rgb = cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2RGB)
    
    from PIL import Image
    img_pil = Image.fromarray(img_rgb)
    input_tensor = transform(img_pil).unsqueeze(0).to(device)
    
    return input_tensor, img_rgb

# ==============================================================================
# GRAD-CAM GENERATION
# ==============================================================================
def generate_heatmap(model, input_tensor, original_rgb):
    try:
        target_layers = [model.layer4[-1]]
        targets = [ClassifierOutputTarget(0)]

        with GradCAM(model=model, target_layers=target_layers) as cam:
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
            
            rgb_resized = cv2.resize(original_rgb, (224, 224))
            rgb_norm = np.float32(rgb_resized) / 255.0
            
            heatmap_img = show_cam_on_image(rgb_norm, grayscale_cam, use_rgb=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = uuid.uuid4().hex[:8]
            filename = f"osteoporosis_{unique_id}_{timestamp}.png"
            out_path = os.path.join(HEATMAP_DIR, filename)
            
            cv2.imwrite(out_path, cv2.cvtColor(heatmap_img, cv2.COLOR_RGB2BGR))
            return os.path.join("heatmaps", filename).replace("\\", "/")
    except Exception as e:
        logger.warning(f"Failed to generate heatmap: {e}")
        return None

# ==============================================================================
# INFERENCE PIPELINE
# ==============================================================================
def run_inference(image_path: str) -> OsteoporosisResult:
    if model is None:
        raise RuntimeError("Model is not loaded.")
        
    start_time = time.time()
    
    try:
        input_tensor, original_rgb = preprocess_image(image_path)
        
        with torch.no_grad():
            output = model(input_tensor)
            prob = torch.sigmoid(output).item()
            
        confidence_pct = prob * 100.0 if prob >= 0.5 else (1.0 - prob) * 100.0
        predicted_class = 1 if prob >= 0.5 else 0
        has_osteo = (predicted_class == 1)
        
        # Confidence logic
        if confidence_pct >= 75.0:
            confidence_flag = "clear"
        elif 55.0 <= confidence_pct < 75.0:
            confidence_flag = "low_confidence"
        else:
            confidence_flag = "inconclusive"
            
        # Message logic
        if confidence_flag == "clear":
            msg = f"{'Osteoporosis' if has_osteo else 'Normal'} detected with high confidence ({confidence_pct:.1f}%)."
        elif confidence_flag == "low_confidence":
            msg = f"Possible {'Osteoporosis' if has_osteo else 'Normal'} but confidence is moderate ({confidence_pct:.1f}%). Review advised."
        else:
            msg = "Inconclusive results. Radiologist review required."
            
        heatmap_path = None
        if confidence_flag != "inconclusive":
            heatmap_path = generate_heatmap(model, input_tensor, original_rgb)
            
        pred_time_ms = (time.time() - start_time) * 1000.0
        
        # MLflow
        try:
            mlflow.set_experiment("osteoporosis_inference")
            run_name = f"osteoporosis_inf_{int(time.time())}"
            with mlflow.start_run(run_name=run_name):
                mlflow.log_metrics({
                    "confidence": confidence_pct,
                    "prediction_time_ms": pred_time_ms
                })
                mlflow.log_params({
                    "prediction": "Osteoporosis" if has_osteo else "Normal",
                    "confidence_flag": confidence_flag,
                    "model_version": MODEL_VERSION
                })
        except Exception as e:
            logger.warning(f"Failed to log to MLflow: {e}")
            
        return OsteoporosisResult(
            has_osteoporosis=has_osteo,
            confidence=round(confidence_pct, 2),
            confidence_flag=confidence_flag,
            heatmap_path=heatmap_path,
            clinical_recommendation=RECS[predicted_class],
            model_version=MODEL_VERSION,
            prediction_time_ms=round(pred_time_ms, 2),
            message=msg
        )
            
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        raise

if __name__ == "__main__":
    test_dir = os.path.join(BASE_DIR, "Mediscan", "osteoporosis", "osteoporosis")
    if os.path.exists(test_dir):
        files = os.listdir(test_dir)
        if files:
            test_image = os.path.join(test_dir, files[0])
            logger.info(f"Testing inference on: {test_image}")
            res = run_inference(test_image)
            print(res)
        else:
            logger.warning("No files found in test directory.")
    else:
        logger.warning(f"Test directory not found: {test_dir}")
