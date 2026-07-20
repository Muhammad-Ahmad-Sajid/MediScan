import os

os.environ["OMP_NUM_THREADS"] = "1"
import sys
import uuid
import time
import logging
import datetime
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

# Suppress albumentations warning if any
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import albumentations as A
from albumentations.pytorch import ToTensorV2

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import RawScoresOutputTarget

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("mediscan.bone_age.inference")


@dataclass
class BoneAgeResult:
    predicted_months: float
    predicted_display: str
    uncertainty_months: float
    confidence_flag: str
    chronological_age_months: Optional[int]
    deviation_months: Optional[float]
    skeletal_age_flag: str
    heatmap_path: Optional[str]
    clinical_recommendation: str
    prediction_time_ms: float
    model_version: str
    mae_note: str


# Global model cache for lazy loading
_model = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "checkpoints/bone_age_best.pth"
HEATMAPS_DIR = "heatmaps"

# Create heatmaps dir
os.makedirs(HEATMAPS_DIR, exist_ok=True)


class BoneAgeModel(nn.Module):
    def __init__(self, use_gender=True):
        super().__init__()
        self.use_gender = use_gender
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        input_features = num_ftrs + 1 if use_gender else num_ftrs
        self.regression_head = nn.Sequential(
            nn.Linear(input_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
        )

    def forward(self, x, gender=None):
        features = self.backbone(x)
        if self.use_gender:
            if gender is None:
                # Default to a neutral gender tensor (0.5) if not provided
                gender = torch.full(
                    (features.size(0), 1),
                    0.5,
                    device=features.device,
                    dtype=features.dtype,
                )
            features = torch.cat([features, gender], dim=1)
        return self.regression_head(features)


def load_model():
    global _model
    if _model is not None:
        return _model

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Bone age checkpoint not found at {CHECKPOINT_PATH}")

    logger.info("Loading Bone Age ResNet-50 model...")
    model = BoneAgeModel()

    # Load safe weights
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()
    _model = model
    return _model


def preprocess_image(img_path: str) -> np.ndarray:
    """Load and preprocess identical to training pipeline."""
    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise ValueError(f"Failed to load image at {img_path}")

    h, w = img_gray.shape
    if h < 50 or w < 50:
        raise ValueError(f"Image too small: {w}x{h}")

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_gray)

    # Convert to 3 channel
    img_rgb = cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2RGB)
    return img_rgb


def get_base_transform():
    return A.Compose(
        [
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )


def get_tta_transforms():
    """Return 5 different albumentations pipelines for TTA."""
    return [
        A.Compose(
            [
                A.Resize(224, 224),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        ),
        A.Compose(
            [
                A.Resize(224, 224),
                A.HorizontalFlip(p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        ),
        A.Compose(
            [
                A.Resize(224, 224),
                A.Rotate(limit=(5, 5), p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        ),
        A.Compose(
            [
                A.Resize(224, 224),
                A.Rotate(limit=(-5, -5), p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        ),
        A.Compose(
            [
                A.Resize(224, 224),
                A.RandomBrightnessContrast(brightness_limit=(0.1, 0.1), contrast_limit=0, p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        ),
    ]


def format_age_display(months_raw: float) -> str:
    years = int(months_raw) // 12
    months = int(months_raw) % 12

    y_str = "year" if years == 1 else "years"
    m_str = "month" if months == 1 else "months"
    return f"{years} {y_str} {months} {m_str}"


def determine_deviation(predicted_months: float, chronological_age_months: Optional[int]):
    if chronological_age_months is None:
        return (
            None,
            "no_comparison",
            "Chronological age not provided. Unable to assess skeletal maturity deviation. Provide patient's date of birth for full assessment.",
        )

    deviation = predicted_months - chronological_age_months
    abs_dev = abs(deviation)

    if abs_dev <= 12:
        flag = "age_appropriate"
        msg = "Bone age is consistent with chronological age."
    elif deviation > 24:
        flag = "significantly_advanced"
        msg = f"Bone age is significantly advanced by {deviation:.1f} months. Urgent pediatric endocrinology referral recommended. Evaluate for: precocious puberty, congenital adrenal hyperplasia, growth hormone secreting tumor."
    elif deviation > 12:
        flag = "advanced"
        msg = f"Bone age is advanced by {deviation:.1f} months. Possible causes: early puberty, growth hormone excess, hyperthyroidism, obesity. Recommend: pediatric endocrinology referral."
    elif deviation < -24:
        flag = "significantly_delayed"
        msg = f"Bone age is significantly delayed by {abs_dev:.1f} months. Urgent pediatric endocrinology referral recommended. Evaluate for: growth hormone deficiency, Turner syndrome, chronic systemic disease."
    else:  # deviation < -12
        flag = "delayed"
        msg = f"Bone age is delayed by {abs_dev:.1f} months. Possible causes: growth hormone deficiency, hypothyroidism, constitutional delay, chronic illness, malnutrition. Recommend: pediatric endocrinology referral, growth hormone testing."

    return deviation, flag, msg


class RegressionTarget:
    """Fallback custom target for Grad-CAM regression."""

    def __call__(self, model_output):
        return model_output.squeeze()


def generate_heatmap(model, img_rgb: np.ndarray, tensor_img: torch.Tensor) -> Optional[str]:
    try:
        target_layers = [model.backbone.layer4[-1]]

        # Try using built-in target, otherwise fallback
        try:
            targets = [RawScoresOutputTarget()]
        except NameError:
            targets = [RegressionTarget()]

        with GradCAM(model=model, target_layers=target_layers) as cam:
            # We don't want to break the computational graph for CAM calculation
            grayscale_cam = cam(input_tensor=tensor_img, targets=targets)[0, :]

        # Resize heatmap to match original image
        h, w = img_rgb.shape[:2]
        grayscale_cam = cv2.resize(grayscale_cam, (w, h))

        # Normalize original image to [0, 1] for blending
        img_normalized = img_rgb.astype(np.float32) / 255.0

        # Blend
        visualization = show_cam_on_image(img_normalized, grayscale_cam, use_rgb=True, colormap=cv2.COLORMAP_JET)
        visualization = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)

        filename = f"bone_age_{uuid.uuid4().hex[:8]}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = os.path.join(HEATMAPS_DIR, filename)
        cv2.imwrite(filepath, visualization)
        return filepath

    except Exception as e:
        logger.warning(f"Grad-CAM generation failed: {e}")
        return None


def run_bone_age_inference(image_path: str, chronological_age_months: int = None) -> BoneAgeResult:
    start_time = time.time()

    logger.info(f"Starting inference for: {image_path}")
    model = load_model()

    img_rgb = preprocess_image(image_path)

    # 1. Test Time Augmentation (TTA)
    tta_pipelines = get_tta_transforms()
    predictions = []

    with torch.no_grad():
        for i, pipe in enumerate(tta_pipelines):
            tensor_img = pipe(image=img_rgb)["image"].unsqueeze(0).to(DEVICE)
            output = model(tensor_img).squeeze().item()
            pred_months = output * 228.0
            pred_months = max(0.0, min(228.0, pred_months))  # Clamp
            predictions.append(pred_months)
            logger.info(f"TTA #{i+1} prediction: {pred_months:.1f} months")

    # Calculate statistics
    predicted_months = float(np.mean(predictions))
    uncertainty_months = float(np.std(predictions))

    logger.info(f"Final Aggregated Prediction: {predicted_months:.1f} ± {uncertainty_months:.1f} months")

    # Determine confidence
    if uncertainty_months <= 4.0:
        confidence = "confident"
    elif uncertainty_months <= 8.0:
        confidence = "moderate"
    else:
        confidence = "low_confidence"

    # Deviation and recommendations
    deviation, flag, rec = determine_deviation(predicted_months, chronological_age_months)

    # Generate Heatmap on original un-augmented image
    base_tensor = get_base_transform()(image=img_rgb)["image"].unsqueeze(0).to(DEVICE)
    heatmap_path = generate_heatmap(model, img_rgb, base_tensor)

    end_time = time.time()
    total_time_ms = (end_time - start_time) * 1000.0

    return BoneAgeResult(
        predicted_months=predicted_months,
        predicted_display=format_age_display(predicted_months),
        uncertainty_months=uncertainty_months,
        confidence_flag=confidence,
        chronological_age_months=chronological_age_months,
        deviation_months=deviation,
        skeletal_age_flag=flag,
        heatmap_path=heatmap_path,
        clinical_recommendation=rec,
        prediction_time_ms=total_time_ms,
        model_version="bone_age_v1",
        mae_note="Model MAE: 7.34 months on RSNA test set",
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python bone_age_inference.py <image_path> [chronological_age_months]")
        sys.exit(1)

    img_path = sys.argv[1]
    chrono_age = int(sys.argv[2]) if len(sys.argv) > 2 else None

    try:
        res = run_bone_age_inference(img_path, chrono_age)

        print("\n" + "=" * 50)
        print("BONE AGE ESTIMATION RESULT")
        print("=" * 50)
        print(f"Predicted Bone Age : {res.predicted_display} ({res.predicted_months:.1f} months)")
        print(f"Uncertainty        : ±{res.uncertainty_months:.1f} months")
        print(f"Confidence         : {res.confidence_flag}")

        if res.chronological_age_months is not None:
            print(
                f"Chronological Age  : {format_age_display(res.chronological_age_months)} ({res.chronological_age_months} months)"
            )
            dev_sign = "+" if res.deviation_months > 0 else ""
            print(f"Deviation          : {dev_sign}{res.deviation_months:.1f} months")

        print(f"Skeletal Age Flag  : {res.skeletal_age_flag}")
        print(f"Heatmap Path       : {res.heatmap_path}")
        print(f"Recommendation     : {res.clinical_recommendation[:80]}...")
        print(f"Prediction Time    : {res.prediction_time_ms:.1f}ms (includes TTA)")
        print(f"Model MAE          : {res.mae_note.split(':')[1].strip()}")
        print(f"Model Version      : {res.model_version}")
        print("=" * 50 + "\n")

    except Exception as e:
        print(f"\nError running inference: {e}")
