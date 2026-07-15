import os
import cv2
import time
import uuid
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
import logging
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import mlflow

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Base path for Windows machine
BASE_DIR = r"d:\X-ray ML Model"
CHECKPOINT_PATH = os.path.join(BASE_DIR, "checkpoints", "arthritis_best.pth")
HEATMAP_DIR = os.path.join(BASE_DIR, "heatmaps")
os.makedirs(HEATMAP_DIR, exist_ok=True)

MODEL_VERSION = "v1.0-arthritis-epoch9"

# Global device and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = None

GRADES = {
    0: {"name": "Normal",   "description": "No signs of osteoarthritis"},
    1: {"name": "Doubtful", "description": "Doubtful joint space narrowing"},
    2: {"name": "Mild",     "description": "Definite joint space narrowing"},
    3: {"name": "Moderate", "description": "Moderate joint space narrowing with sclerosis"},
    4: {"name": "Severe",   "description": "Large osteophytes, severe joint space narrowing"}
}

CLINICAL_RECS = {
    0: "No treatment required. Routine monitoring recommended.",
    1: "Conservative management. Lifestyle modifications advised. Follow-up in 12 months.",
    2: "Physical therapy and pain management recommended. Anti-inflammatory medication may be considered. Follow-up in 6 months.",
    3: "Orthopedic consultation recommended. Consider corticosteroid injections. Follow-up in 3 months.",
    4: "Urgent orthopedic referral. Joint replacement surgery may be indicated."
}

@dataclass
class ArthritisResult:
    grade: int
    grade_name: str
    grade_description: str
    confidence: float
    confidence_flag: str
    heatmap_path: Optional[str]
    clinical_recommendation: str
    model_version: str
    prediction_time_ms: float
    message: str


def load_model():
    """Loads the model exactly once globally."""
    global _model
    if _model is not None:
        return _model
        
    logger.info(f"Loading Arthritis model from {CHECKPOINT_PATH} to {device}")
    
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Missing checkpoint file: {CHECKPOINT_PATH}")
        
    try:
        model = models.resnet50(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 5)
        )
        
        # Need to use weights_only=False since saving full dictionary
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint) # In case it was saved directly
            
        model.to(device)
        model.eval()
        _model = model
        logger.info("Arthritis model loaded successfully.")
        return _model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def preprocess_image(image_path: str):
    """
    Applies CLAHE, replicates to 3-channel RGB, resizes, and normalizes
    to match the exact training preprocessing pipeline.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to load image or not a valid image file: {image_path}")
        
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img)
    
    # Convert to 3-channel RGB
    img_rgb = cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2RGB)
    
    # Resize to 224x224
    img_resized = cv2.resize(img_rgb, (224, 224))
    
    # Normalize with ImageNet mean and std
    img_norm = img_resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_norm = (img_norm - mean) / std
    
    # Convert to tensor (B, C, H, W)
    tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).float()
    
    return tensor, img_resized


def generate_heatmap(model, input_tensor, original_resized, target_grade):
    """
    Generates a Grad-CAM heatmap over the original resized image.
    """
    target_layers = [model.layer4[-1]]
    
    with GradCAM(model=model, target_layers=target_layers) as cam:
        targets = [ClassifierOutputTarget(target_grade)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
        
        rgb_img_float = original_resized.astype(np.float32) / 255.0
        cam_image = show_cam_on_image(rgb_img_float, grayscale_cam, use_rgb=True, colormap=cv2.COLORMAP_JET)
        
        timestamp = int(time.time())
        uid = uuid.uuid4().hex[:8]
        filename = f"arthritis_{uid}_{timestamp}.png"
        heatmap_path = os.path.join(HEATMAP_DIR, filename)
        
        cv2.imwrite(heatmap_path, cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))
        return heatmap_path


def analyze_arthritis(image_path: str) -> ArthritisResult:
    """
    Runs the full inference pipeline on an X-ray image for Arthritis grading.
    """
    try:
        model = load_model()
        
        start_time = time.time()
        
        # Preprocess
        input_tensor, original_resized = preprocess_image(image_path)
        input_tensor = input_tensor.to(device)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            
        # Get prediction
        confidence_val, predicted_class = torch.max(probabilities, dim=0)
        grade = int(predicted_class.item())
        confidence = float(confidence_val.item())
        confidence_pct = confidence * 100.0
        
        # Confidence Thresholding
        if confidence_pct >= 70:
            confidence_flag = "clear"
        elif confidence_pct >= 50:
            confidence_flag = "low_confidence"
        else:
            confidence_flag = "inconclusive"
            
        grade_info = GRADES[grade]
        grade_name = grade_info["name"]
        grade_desc = grade_info["description"]
        
        # Message Formatting
        if confidence_flag == "clear":
            message = f"Grade {grade} ({grade_name}) detected with {confidence_pct:.1f}% confidence. {grade_desc}"
        elif confidence_flag == "low_confidence":
            message = f"Possible Grade {grade} ({grade_name}) but confidence is moderate ({confidence_pct:.1f}%). Recommend radiologist review."
        else:
            message = "Inconclusive grading. Manual radiologist review required."
            
        # Heatmap Generation
        heatmap_path = None
        if confidence_flag != "inconclusive":
            heatmap_path = generate_heatmap(model, input_tensor, original_resized, grade)
            
        pred_time_ms = (time.time() - start_time) * 1000.0
        
        # Log to MLflow
        try:
            mlflow.set_experiment("arthritis_inference")
            run_name = f"arthritis_inference_{int(time.time())}"
            with mlflow.start_run(run_name=run_name):
                mlflow.log_metrics({
                    "confidence": confidence_pct,
                    "prediction_time_ms": pred_time_ms
                })
                mlflow.log_params({
                    "model_version": MODEL_VERSION,
                    "grade": grade,
                    "grade_name": grade_name,
                    "confidence_flag": confidence_flag,
                    "image_filename": os.path.basename(image_path)
                })
        except Exception as e:
            logger.warning(f"Failed to log to MLflow: {e}")
            
        return ArthritisResult(
            grade=grade,
            grade_name=grade_name,
            grade_description=grade_desc,
            confidence=round(confidence_pct, 4),
            confidence_flag=confidence_flag,
            heatmap_path=heatmap_path,
            clinical_recommendation=CLINICAL_RECS[grade],
            model_version=MODEL_VERSION,
            prediction_time_ms=round(pred_time_ms, 2),
            message=message
        )
        
    except Exception as e:
        logger.error(f"Error during arthritis inference: {e}")
        raise


if __name__ == "__main__":
    test_image_path = os.path.join(BASE_DIR, "Mediscan", "val", "2", "9015402R.png")
    
    if os.path.exists(test_image_path):
        print(f"\nTesting inference on: {test_image_path}")
        try:
            result = analyze_arthritis(test_image_path)
            print("\n--- INFERENCE RESULT ---")
            print(f"Grade: {result.grade} ({result.grade_name})")
            print(f"Confidence: {result.confidence}% [{result.confidence_flag}]")
            print(f"Message: {result.message}")
            print(f"Recommendation: {result.clinical_recommendation}")
            print(f"Heatmap: {result.heatmap_path}")
            print(f"Time: {result.prediction_time_ms}ms")
            print("------------------------\n")
        except Exception as e:
            print(f"Failed: {e}")
    else:
        print(f"Test image not found at {test_image_path}. Please check the path.")
