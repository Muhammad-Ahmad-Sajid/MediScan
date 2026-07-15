import os
import sys
import time
import uuid
import logging
from datetime import datetime
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
except ImportError:
    GradCAM = None

# ==============================================================================
# CONFIGURATION & LOGGING
# ==============================================================================
# Setup logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("mediscan.lung_nodule.inference")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "d:/X-ray ML Model/checkpoints/lung_nodule_best.pth"
HEATMAP_DIR = "d:/X-ray ML Model/heatmaps"
os.makedirs(HEATMAP_DIR, exist_ok=True)

# Global Lazy Loader
_model = None

# ==============================================================================
# DATACLASS
# ==============================================================================
@dataclass
class LungNoduleResult:
    has_nodule: bool
    nodule_probability: float
    confidence: float
    confidence_flag: str
    heatmap_path: Optional[str]
    clinical_recommendation: str
    urgency: str
    prediction_time_ms: float
    model_version: str = "lung_nodule_v1"

# ==============================================================================
# LAZY MODEL LOADING
# ==============================================================================
def load_model():
    global _model
    if _model is not None:
        return _model

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found at {CHECKPOINT_PATH}. Train the model first.")

    logger.info("Loading Lung Nodule Detection model...")
    # Build architecture matching training
    model = models.resnet50()
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 1)  # Binary classification head from train_lung_nodule.py
    )
    
    # Load weights
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    
    model = model.to(DEVICE)
    model.eval()
    
    _model = model
    logger.info("Model loaded successfully.")
    return _model

# ==============================================================================
# PREPROCESSING
# ==============================================================================
def preprocess_image(img_path):
    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise ValueError(f"Cannot load image: {img_path}")
        
    h, w = img_gray.shape
    if h < 50 or w < 50:
        raise ValueError(f"Image too small ({w}x{h}). Minimum required is 50x50.")

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_gray)
    
    # RGB Conversion
    img_rgb = cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2RGB)
    original_for_heatmap = img_rgb.copy()
    
    # Resize and Normalize
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_normalized = (img_normalized - mean) / std
    
    # HWC to CHW
    tensor = torch.tensor(img_normalized).permute(2, 0, 1).unsqueeze(0).float()
    return tensor, original_for_heatmap

# ==============================================================================
# GRAD-CAM HEATMAP
# ==============================================================================
def generate_gradcam(model, input_tensor, original_image):
    if GradCAM is None:
        logger.warning("pytorch-grad-cam not installed. Skipping heatmap generation.")
        return None
        
    try:
        target_layers = [model.layer4[-1]]
        cam = GradCAM(model=model, target_layers=target_layers)
        
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)
        grayscale_cam = grayscale_cam[0, :]
        
        # Resize cam to match original image dimensions
        h, w, _ = original_image.shape
        grayscale_cam_resized = cv2.resize(grayscale_cam, (w, h))
        
        # Normalize original image to [0, 1] for the show_cam_on_image function
        img_float = original_image.astype(np.float32) / 255.0
        
        # Blend heatmap and image
        visualization = show_cam_on_image(img_float, grayscale_cam_resized, use_rgb=True, colormap=cv2.COLORMAP_JET, image_weight=0.4)
        
        # Convert back to BGR for saving with cv2
        visualization_bgr = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
        
        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8]
        heatmap_filename = f"lung_nodule_{unique_id}_{timestamp}.png"
        heatmap_path = os.path.join(HEATMAP_DIR, heatmap_filename)
        
        cv2.imwrite(heatmap_path, visualization_bgr)
        return heatmap_path
        
    except Exception as e:
        logger.warning(f"Grad-CAM generation failed: {e}")
        return None

# ==============================================================================
# INFERENCE PIPELINE
# ==============================================================================
def get_clinical_recommendation(has_nodule, confidence_flag):
    if has_nodule and confidence_flag == "clear":
        return ("Pulmonary nodule detected with high confidence.\n"
                "Recommend CT chest with contrast for detailed characterization.\n"
                "Refer to pulmonologist for evaluation.\n"
                "Follow-up imaging in 3 months to assess growth.")
    elif has_nodule and confidence_flag == "probable":
        return ("Findings suggest possible pulmonary nodule.\n"
                "Recommend high-resolution CT scan for confirmation.\n"
                "Clinical correlation with patient history advised.\n"
                "Follow-up imaging in 3-6 months.")
    elif confidence_flag == "borderline":
        return ("Findings are inconclusive for pulmonary nodule.\n"
                "Recommend repeat imaging in 4-6 weeks.\n"
                "Consider CT chest if clinical suspicion is high.\n"
                "Correlate with symptoms and risk factors (smoking history, age).")
    elif confidence_flag == "likely_normal":
        return ("No significant pulmonary nodule detected.\n"
                "Lung fields appear within normal limits.\n"
                "Routine follow-up as clinically indicated.\n"
                "Annual screening recommended for high-risk patients.")
    elif not has_nodule and confidence_flag == "clear":
        return ("No pulmonary nodule detected.\n"
                "CT scan appears normal.\n"
                "Continue routine screening per guidelines\n"
                "(annual LDCT for high-risk patients aged 50-80).")
    return "No recommendation available."

def get_urgency(has_nodule, confidence_flag):
    if not has_nodule:
        return "routine"
    if confidence_flag in ["clear", "probable"]:
        return "urgent"
    if confidence_flag == "borderline":
        return "routine"
    return "routine"

def run_lung_nodule_inference(image_path: str) -> LungNoduleResult:
    start_time = time.time()
    
    logger.info(f"Starting inference for: {image_path}")
    model = load_model()
    
    # Preprocess
    input_tensor, original_image = preprocess_image(image_path)
    input_tensor = input_tensor.to(DEVICE)
    logger.info("Preprocessing complete.")
    
    # Inference
    with torch.no_grad():
        output = model(input_tensor)
        # Using BCEWithLogitsLoss during training means output is raw logits.
        # We need sigmoid to get probability for class 1 (Nodule).
        nodule_probability = torch.sigmoid(output).item()
        
    logger.info("Inference complete.")
    
    # Thresholds
    has_nodule = nodule_probability >= 0.55
    
    # Confidence calculation
    # Since it's binary, confidence is how far the probability is from 0.5
    # If nodule_prob is 0.9, confidence is 90%. If it's 0.1, confidence is 90% (that it's not a nodule)
    confidence = nodule_probability if nodule_probability >= 0.5 else (1.0 - nodule_probability)
    
    if nodule_probability >= 0.85 or nodule_probability <= 0.15:
        confidence_flag = "clear"
    elif 0.70 <= nodule_probability < 0.85:
        confidence_flag = "probable"
    elif 0.45 <= nodule_probability < 0.70:
        confidence_flag = "borderline"
    else: # 0.15 < prob < 0.45
        confidence_flag = "likely_normal"
        
    urgency = get_urgency(has_nodule, confidence_flag)
    recommendation = get_clinical_recommendation(has_nodule, confidence_flag)
    
    # Generate Heatmap
    heatmap_path = generate_gradcam(model, input_tensor, original_image)
    if heatmap_path:
        logger.info(f"Heatmap saved to: {heatmap_path}")
    else:
        logger.info("Heatmap generation failed or skipped.")
        
    end_time = time.time()
    prediction_time_ms = (end_time - start_time) * 1000.0
    
    return LungNoduleResult(
        has_nodule=has_nodule,
        nodule_probability=nodule_probability,
        confidence=confidence,
        confidence_flag=confidence_flag,
        heatmap_path=heatmap_path,
        clinical_recommendation=recommendation,
        urgency=urgency,
        prediction_time_ms=prediction_time_ms
    )

# ==============================================================================
# STANDALONE TEST
# ==============================================================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python lung_nodule_inference.py <image_path>")
        sys.exit(1)
        
    image_path = sys.argv[1]
    
    try:
        # Initial call to get full prediction time including load
        result = run_lung_nodule_inference(image_path)
        
        # Second call to test cached model speed
        start_cache = time.time()
        cached_result = run_lung_nodule_inference(image_path)
        cached_time = (time.time() - start_cache) * 1000.0
        
        # Override prediction time for display with cached time
        result.prediction_time_ms = cached_time
        
        print("\n" + "="*50)
        print("LUNG NODULE SCREENING RESULT")
        print("="*50)
        print(f"Nodule Detected   : {result.has_nodule}")
        print(f"Nodule Probability: {result.nodule_probability * 100:.2f}%")
        print(f"Confidence        : {result.confidence * 100:.2f}%")
        print(f"Confidence Flag   : {result.confidence_flag}")
        print(f"Urgency           : {result.urgency}")
        print(f"Heatmap Path      : {result.heatmap_path}")
        
        short_rec = result.clinical_recommendation.replace('\n', ' ')
        if len(short_rec) > 80:
            short_rec = short_rec[:77] + "..."
        print(f"Recommendation    : {short_rec}")
        print(f"Prediction Time   : {result.prediction_time_ms:.1f}ms")
        print(f"Model Version     : {result.model_version}")
        print("="*50 + "\n")
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        sys.exit(1)
