"""
TB Inference Module for MediScan AI Platform
=============================================
Tuberculosis screening from chest X-ray images using a trained ResNet-50 model.

Called by the FastAPI backend when a doctor uploads a chest X-ray for TB screening.
Model is loaded once at module level for efficient repeated inference.
"""

import os
import sys
import cv2
import time
import uuid
import logging
import numpy as np
from dataclasses import dataclass
from typing import Optional
from datetime import datetime

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BASE_DIR = "d:/X-ray ML Model"
CHECKPOINT_PATH = os.path.join(BASE_DIR, "checkpoints", "tb_best.pth")
HEATMAP_DIR = os.path.join(BASE_DIR, "heatmaps")
os.makedirs(HEATMAP_DIR, exist_ok=True)

MODEL_VERSION = "tb_v1"

# ==============================================================================
# LOGGING
# ==============================================================================
logger = logging.getLogger("mediscan.tb.inference")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)


# ==============================================================================
# OUTPUT DATACLASS
# ==============================================================================
@dataclass
class TBInferenceResult:
    has_tb: bool
    tb_probability: float
    confidence: float
    confidence_flag: str
    heatmap_path: Optional[str]
    clinical_recommendation: str
    prediction_time_ms: float
    model_version: str


# ==============================================================================
# CLINICAL RECOMMENDATIONS
# ==============================================================================
RECOMMENDATIONS = {
    ("clear", True): (
        "Chest X-ray findings strongly suggest tuberculosis. "
        "Immediate referral to pulmonology recommended. "
        "Confirm with sputum smear microscopy and GeneXpert MTB/RIF."
    ),
    ("probable", True): (
        "Chest X-ray findings are consistent with possible tuberculosis. "
        "Clinical correlation advised. "
        "Recommend sputum testing and Mantoux/TST or IGRA blood test."
    ),
    ("borderline", None): (
        "Findings are inconclusive for tuberculosis. "
        "Recommend repeat imaging in 2-4 weeks or additional testing "
        "(sputum culture, CT chest) based on clinical suspicion."
    ),
    ("likely_normal", None): (
        "Chest X-ray appears within normal limits. "
        "Low suspicion for active tuberculosis. "
        "If clinical symptoms persist, consider repeat imaging or sputum test."
    ),
    ("clear", False): (
        "No radiographic evidence of tuberculosis detected. "
        "Chest X-ray appears normal. "
        "Routine follow-up as clinically indicated."
    ),
}

# ==============================================================================
# PREPROCESSING (must match train_tb.py exactly)
# ==============================================================================
val_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def preprocess_image(image_path: str):
    """
    Preprocess a chest X-ray image identically to the training pipeline.
    Returns the input tensor and the original RGB image (for heatmap overlay).
    """
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise ValueError(f"Cannot load image: {image_path}")

    if img_gray.shape[0] < 50 or img_gray.shape[1] < 50:
        raise ValueError(
            f"Image too small: {img_gray.shape[1]}x{img_gray.shape[0]} pixels"
        )

    # CLAHE with clipLimit=3.0 (higher for chest X-rays)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_gray)

    # Convert to 3-channel RGB by repeating grayscale channel
    img_rgb = cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2RGB)

    logger.info("Preprocessing complete: CLAHE applied, converted to RGB")

    img_pil = Image.fromarray(img_rgb)
    input_tensor = val_transform(img_pil).unsqueeze(0)

    return input_tensor, img_rgb


# ==============================================================================
# MODEL CREATION & LOADING (Module Level — Lazy Singleton)
# ==============================================================================
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = None


def _create_model():
    """Create ResNet-50 with custom head matching train_tb.py exactly."""
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    # Architecture must match train_tb.py: Linear(2048,512)->ReLU->Dropout(0.5)->Linear(512,1)
    # Single output node was used with BCEWithLogitsLoss during training.
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, 1)
    )
    return model


def get_model():
    """Lazy loader: loads model on first call, reuses on subsequent calls."""
    global _model
    if _model is not None:
        return _model

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"TB model checkpoint not found at: {CHECKPOINT_PATH}. "
            "Please run train_tb.py first to generate the checkpoint."
        )

    logger.info(f"Loading TB model from {CHECKPOINT_PATH} onto {_device}")
    model = _create_model()

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=_device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(_device)
    model.eval()

    logger.info(
        f"TB model loaded successfully (trained epoch {checkpoint.get('epoch', '?')}, "
        f"val_acc={checkpoint.get('val_acc', '?')}, sensitivity={checkpoint.get('sensitivity', '?')})"
    )

    _model = model
    return _model


# ==============================================================================
# GRAD-CAM HEATMAP GENERATION
# ==============================================================================
def generate_heatmap(model, input_tensor, original_rgb):
    """
    Generate Grad-CAM heatmap overlay on the original X-ray.
    Uses cv2.applyColorMap with COLORMAP_JET and 0.6/0.4 blending.
    """
    try:
        target_layers = [model.layer4[-1]]
        targets = [ClassifierOutputTarget(0)]

        with GradCAM(model=model, target_layers=target_layers) as cam:
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]

        # Resize heatmap to match original image dimensions
        h, w = original_rgb.shape[:2]
        heatmap_resized = cv2.resize(grayscale_cam, (w, h))

        # Apply JET colormap
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
        )

        # Blend: 0.6 * heatmap + 0.4 * original
        original_bgr = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)
        blended = cv2.addWeighted(heatmap_colored, 0.6, original_bgr, 0.4, 0)

        # Save with unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8]
        filename = f"tb_{unique_id}_{timestamp}.png"
        out_path = os.path.join(HEATMAP_DIR, filename)

        cv2.imwrite(out_path, blended)
        rel_path = os.path.join("heatmaps", filename).replace("\\", "/")
        logger.info(f"Heatmap saved to {out_path}")
        return rel_path

    except Exception as e:
        logger.warning(f"Grad-CAM heatmap generation failed: {e}")
        return None


# ==============================================================================
# CONFIDENCE FLAG & RECOMMENDATION LOGIC
# ==============================================================================
def _get_confidence_flag(tb_prob: float) -> str:
    """
    Determine clinical confidence flag from raw TB probability.
    - tb_prob >= 0.85 OR tb_prob <= 0.15  -> "clear"
    - 0.60 <= tb_prob < 0.85              -> "probable"
    - 0.40 <= tb_prob < 0.60              -> "borderline"
    - 0.15 < tb_prob < 0.40              -> "likely_normal"
    """
    if tb_prob >= 0.85 or tb_prob <= 0.15:
        return "clear"
    elif tb_prob >= 0.60:
        return "probable"
    elif tb_prob >= 0.40:
        return "borderline"
    else:
        return "likely_normal"


def _get_recommendation(confidence_flag: str, has_tb: bool) -> str:
    """Look up clinical recommendation based on flag and prediction."""
    if confidence_flag == "borderline":
        return RECOMMENDATIONS[("borderline", None)]
    elif confidence_flag == "likely_normal":
        return RECOMMENDATIONS[("likely_normal", None)]
    else:
        return RECOMMENDATIONS[(confidence_flag, has_tb)]


# ==============================================================================
# MAIN INFERENCE FUNCTION
# ==============================================================================
def run_tb_inference(image_path: str) -> TBInferenceResult:
    """
    Full TB screening inference pipeline.

    Args:
        image_path: Path to chest X-ray image (PNG, JPG, JPEG)

    Returns:
        TBInferenceResult with prediction, heatmap, and recommendation
    """
    start_time = time.time()

    model = get_model()
    logger.info(f"Image loaded: {image_path}")

    # Preprocess
    input_tensor, original_rgb = preprocess_image(image_path)
    input_tensor = input_tensor.to(_device)

    # Inference
    with torch.no_grad():
        logit = model(input_tensor)
        tb_probability = torch.sigmoid(logit).item()

    logger.info("Inference complete")

    # Classification (threshold at 0.60)
    has_tb = tb_probability >= 0.60

    # Confidence = how sure the model is (max of prob, 1-prob)
    confidence = max(tb_probability, 1.0 - tb_probability)

    # Confidence flag
    confidence_flag = _get_confidence_flag(tb_probability)

    # Clinical recommendation
    recommendation = _get_recommendation(confidence_flag, has_tb)

    logger.info(
        f"Prediction: {'TB' if has_tb else 'Normal'} | "
        f"TB Prob: {tb_probability:.4f} | Confidence: {confidence:.4f} | "
        f"Flag: {confidence_flag}"
    )

    # Grad-CAM heatmap
    heatmap_path = generate_heatmap(model, input_tensor, original_rgb)

    prediction_time_ms = (time.time() - start_time) * 1000.0

    return TBInferenceResult(
        has_tb=has_tb,
        tb_probability=round(tb_probability * 100, 2),
        confidence=round(confidence * 100, 2),
        confidence_flag=confidence_flag,
        heatmap_path=heatmap_path,
        clinical_recommendation=recommendation,
        prediction_time_ms=round(prediction_time_ms, 2),
        model_version=MODEL_VERSION,
    )


# ==============================================================================
# STANDALONE TEST
# ==============================================================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        logger.error("Usage: python tb_inference.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    result = run_tb_inference(image_path)

    # Clean summary output
    summary = (
        "==================================================\n"
        "TB SCREENING RESULT\n"
        "==================================================\n"
        f"Has Tuberculosis  : {result.has_tb}\n"
        f"TB Probability    : {result.tb_probability:.2f}%\n"
        f"Confidence        : {result.confidence:.2f}%\n"
        f"Confidence Flag   : {result.confidence_flag}\n"
        f"Heatmap Path      : {result.heatmap_path}\n"
        f"Recommendation    : {result.clinical_recommendation[:80]}...\n"
        f"Prediction Time   : {result.prediction_time_ms:.1f}ms\n"
        f"Model Version     : {result.model_version}\n"
        "=================================================="
    )
    logger.info(f"\n{summary}")
