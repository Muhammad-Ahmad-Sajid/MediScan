import os
import sys
import time
import uuid
import logging
import datetime
from pathlib import Path
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure project root is in path to resolve src.* packages
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.model_training.model import Stage2FractureModel

# Setup structured logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Constants
CHECKPOINT_PATH = Path("checkpoints/stage2_best.pth")
FRACTURE_CLASSES = ["not_fractured", "fractured"]
REGION_CLASSES = ["hand", "leg", "hip", "shoulder", "unknown"]
MODEL_VERSION = "v1.0-stage2-epoch8"


# Structured result representation
@dataclass
class InferenceResult:
    fracture_detected: bool
    fracture_confidence: float
    bone_region: str
    bone_confidence: float
    confidence_flag: str
    heatmap_path: str or None
    model_version: str
    prediction_time_ms: float
    message: str


# ------------------------------------------------------------------------------
# Module-level Model Load (Loads ONCE when module is imported)
# ------------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None


def load_global_model():
    global model
    if not CHECKPOINT_PATH.exists():
        err_msg = (
            f"Critical Error: Missing checkpoint file at {CHECKPOINT_PATH.resolve()}.\n"
            "Please ensure the training runs successfully and checkpoints/stage2_best.pth is generated."
        )
        logger.error(err_msg)
        raise FileNotFoundError(err_msg)

    logger.info(f"Loading Stage 2 best checkpoint model from {CHECKPOINT_PATH.resolve()}...")
    try:
        # Load Model architecture and set strictly to False for backbone, then load weights
        model = Stage2FractureModel(pretrained=False)
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        logger.info("Stage 2 model loaded successfully at module level.")
    except Exception as e:
        logger.error(f"Failed to load Stage 2 model: {e}")
        raise RuntimeError(f"Failed to initialize global model: {e}")


# Load the model once on import
load_global_model()


# ------------------------------------------------------------------------------
# Model Wrapper for pytorch-grad-cam library
# ------------------------------------------------------------------------------
class ModelWrapper(nn.Module):
    """
    Wraps model to return only the fracture_head logits so Grad-CAM
    calculates gradients specifically for the fracture detection task.
    """

    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, x):
        fracture_logits, _ = self.base_model(x)
        return fracture_logits


# ------------------------------------------------------------------------------
# Grad-CAM Heatmap Generation Helper
# ------------------------------------------------------------------------------
def generate_gradcam_heatmap(
    image_path: str, input_tensor: torch.Tensor, predicted_class_idx: int, img_resized: np.ndarray
) -> str:
    """
    Generates a Grad-CAM heatmap highlighting regions contributing to the prediction
    and overlays it on the 224x224 grayscale X-ray using cv2.COLORMAP_JET.
    """
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        from pytorch_grad_cam.utils.image import show_cam_on_image

        wrapped_model = ModelWrapper(model)
        target_layers = [wrapped_model.base_model.backbone.layer4[-1]]
        targets = [ClassifierOutputTarget(predicted_class_idx)]

        cam = GradCAM(model=wrapped_model, target_layers=target_layers)
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]

        # Overlay on the original grayscale 224x224 image
        # Extract grayscale channel, repeat it to create 3-channel RGB float [0, 1]
        img_gray_resized = img_resized[:, :, 0]
        rgb_img_float = np.float32(cv2.cvtColor(img_gray_resized, cv2.COLOR_GRAY2RGB)) / 255.0

        # show_cam_on_image overlays CAM mask on the image using COLORMAP_JET
        cam_image = show_cam_on_image(
            rgb_img_float, grayscale_cam, use_rgb=True, colormap=cv2.COLORMAP_JET
        )
        cam_image_bgr = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

        # Save to heatmaps/{uuid}_{timestamp}.png
        heatmaps_dir = Path("heatmaps")
        heatmaps_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{uuid.uuid4().hex}_{timestamp}.png"
        output_path = heatmaps_dir / output_filename

        cv2.imwrite(str(output_path), cam_image_bgr)
        logger.info(f"Grad-CAM Heatmap saved at {output_path}")
        return str(output_path).replace("\\", "/")
    except Exception as e:
        logger.error(f"Error during Grad-CAM generation: {e}")
        raise e


# ------------------------------------------------------------------------------
# MLflow Logging Helper
# ------------------------------------------------------------------------------
def log_inference_to_mlflow(
    image_path: str,
    fracture_predicted: str,
    bone_predicted: str,
    fracture_confidence: float,
    bone_confidence: float,
    confidence_flag: str,
    prediction_time_ms: float,
):
    """
    Logs metadata, metrics, and parameters of the inference run to MLflow.
    """
    try:
        import mlflow

        mlflow.set_experiment("fracture_inference")

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"inference_{timestamp}"

        with mlflow.start_run(run_name=run_name):
            # Parameters
            mlflow.log_param("model_version", MODEL_VERSION)
            mlflow.log_param("image_filename", Path(image_path).name)
            mlflow.log_param("bone_predicted", bone_predicted)
            mlflow.log_param("fracture_predicted", fracture_predicted)
            mlflow.log_param("confidence_flag", confidence_flag)

            # Metrics
            mlflow.log_metric("fracture_confidence", fracture_confidence)
            mlflow.log_metric("bone_confidence", bone_confidence)
            mlflow.log_metric("prediction_time_ms", prediction_time_ms)

            # Numeric encoding of the flag to satisfy metric standard (must be floats)
            flag_numeric = {"inconclusive": 0.0, "low_confidence": 1.0, "clear": 2.0}[
                confidence_flag
            ]
            mlflow.log_metric("confidence_flag_numeric", flag_numeric)

            logger.info("Logged inference run details to MLflow.")
    except Exception as e:
        logger.warning(f"MLflow logging bypassed/failed: {e}")


# ------------------------------------------------------------------------------
# Core Inference Function
# ------------------------------------------------------------------------------
def run_inference(image_path: str) -> InferenceResult:
    """
    Executes core inference pipeline on the input X-ray image:
    1. Loads and preprocesses image with CLAHE and normalizations.
    2. Runs forward pass through multi-task model.
    3. Handles confidence thresholding and early returns for inconclusive results.
    4. Generates explainable Grad-CAM heatmaps.
    5. Logs metadata to MLflow.

    Args:
        image_path (str): File path to input X-ray image.

    Returns:
        InferenceResult: Data object containing predictions, confidences, heatmap, and messages.
    """
    start_time = time.time()

    # 1. Image Preprocessing with Error Handling
    try:
        # Load as grayscale
        img_gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            raise ValueError(f"Failed to read image content from path: {image_path}")
    except Exception as e:
        err_msg = f"Failed to load image file: {e}"
        logger.error(err_msg)
        raise ValueError(err_msg)

    # Apply CLAHE (clipLimit=2.0, tileGridSize=(8,8))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_enhanced = clahe.apply(img_gray)

    # Convert to 3-channel RGB (repeating grayscale) and resize
    img_rgb = cv2.cvtColor(img_enhanced, cv2.COLOR_GRAY2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)

    # Normalize with ImageNet mean/std
    img_float = img_resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_normalized = (img_float - mean) / std

    # Transpose HWC -> CHW, unsqueeze batch dim, and send to device
    input_tensor = torch.from_numpy(img_normalized.transpose(2, 0, 1)).unsqueeze(0).to(device)

    # 2. Forward Pass for Fracture Head
    with torch.no_grad():
        fracture_logits, region_logits = model(input_tensor)

        # Softmax on fracture head
        fracture_probs = F.softmax(fracture_logits, dim=1)
        fracture_conf_tensor, fracture_pred_tensor = fracture_probs.max(1)

        fracture_confidence = float(fracture_conf_tensor.item())
        fracture_pred_idx = int(fracture_pred_tensor.item())
        fracture_detected = bool(fracture_pred_idx == 1)

    # 3. Confidence Thresholding
    if fracture_confidence >= 0.80:
        confidence_flag = "clear"
    elif 0.60 <= fracture_confidence < 0.80:
        confidence_flag = "low_confidence"
    else:
        confidence_flag = "inconclusive"

    # 4. Handle Inconclusive Early Exit
    if confidence_flag == "inconclusive":
        prediction_time_ms = (time.time() - start_time) * 1000.0
        msg = "Inconclusive result. Manual radiologist review required."
        logger.info(f"Inference completed with flag: {confidence_flag}. Early exiting.")

        # Log to MLflow
        log_inference_to_mlflow(
            image_path=image_path,
            fracture_predicted="inconclusive",
            bone_predicted="N/A",
            fracture_confidence=fracture_confidence,
            bone_confidence=0.0,
            confidence_flag=confidence_flag,
            prediction_time_ms=prediction_time_ms,
        )

        return InferenceResult(
            fracture_detected=False,
            fracture_confidence=round(fracture_confidence, 4),
            bone_region="unknown",
            bone_confidence=0.0,
            confidence_flag=confidence_flag,
            heatmap_path=None,
            model_version=MODEL_VERSION,
            prediction_time_ms=round(prediction_time_ms, 2),
            message=msg,
        )

    # 5. Process Body Region Head (Runs only if not inconclusive)
    with torch.no_grad():
        region_probs = F.softmax(region_logits, dim=1)
        region_conf_tensor, region_pred_tensor = region_probs.max(1)

        bone_confidence = float(region_conf_tensor.item())
        region_pred_idx = int(region_pred_tensor.item())
        bone_region = REGION_CLASSES[region_pred_idx]

    # 6. Generate Grad-CAM Heatmap
    heatmap_path = None
    try:
        heatmap_path = generate_gradcam_heatmap(
            image_path=image_path,
            input_tensor=input_tensor,
            predicted_class_idx=fracture_pred_idx,
            img_resized=img_resized,
        )
    except Exception as e:
        logger.warning(f"Grad-CAM generation failed, continuing: {e}")

    # Calculate final duration
    prediction_time_ms = (time.time() - start_time) * 1000.0

    # 7. Build Human Readable Message
    confidence_pct = f"{fracture_confidence * 100:.2f}"
    if confidence_flag == "clear":
        if fracture_detected:
            msg = f"Fracture detected in {bone_region} region with {confidence_pct}% confidence. Prognosis will be calculated separately."
        else:
            msg = f"No fracture detected with {confidence_pct}% confidence."
    else:  # low_confidence
        msg = f"Possible fracture detected but confidence is low ({confidence_pct}%). Recommend repeat scan or manual review."

    # 8. MLflow Logging
    log_inference_to_mlflow(
        image_path=image_path,
        fracture_predicted=FRACTURE_CLASSES[fracture_pred_idx],
        bone_predicted=bone_region,
        fracture_confidence=fracture_confidence,
        bone_confidence=bone_confidence,
        confidence_flag=confidence_flag,
        prediction_time_ms=prediction_time_ms,
    )

    logger.info(
        f"Inference complete: fracture_detected={fracture_detected}, bone_region={bone_region}, flag={confidence_flag}"
    )

    return InferenceResult(
        fracture_detected=fracture_detected,
        fracture_confidence=round(fracture_confidence, 4),
        bone_region=bone_region,
        bone_confidence=round(bone_confidence, 4),
        confidence_flag=confidence_flag,
        heatmap_path=heatmap_path,
        model_version=MODEL_VERSION,
        prediction_time_ms=round(prediction_time_ms, 2),
        message=msg,
    )


# ------------------------------------------------------------------------------
# Test Block
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Test file run logic
    test_image = "data/stage2/train/fracatlas_IMG0001816.jpg"
    if os.path.exists(test_image):
        logger.info(f"Testing inference module on: {test_image}")
        try:
            result = run_inference(test_image)
            print("\n" + "=" * 50)
            print("TEST INFERENCE RESULT:")
            print("=" * 50)
            for k, v in result.__dict__.items():
                print(f"{k:<25}: {v}")
            print("=" * 50)
        except Exception as err:
            logger.error(f"Inference test run failed: {err}")
    else:
        logger.info(f"Test image {test_image} not found, skipping direct run test.")
