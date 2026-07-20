import cv2
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.model_training.model import FractureModel
from src.data_preparation.preprocess import get_inference_transform
from src.config import MODEL_CHECKPOINT_PATH


class ModelWrapper(nn.Module):
    """
    Wraps the multi-task FractureModel to return only the severity logits
    so that pytorch-grad-cam can compute gradients for explainability heatmaps.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        severity_logits, _ = self.model(x)
        return severity_logits


def generate_heatmap(image_path: str, model_path: str = None) -> str:
    """
    Loads the trained multi-task FractureModel from model_path, generates a Grad-CAM
    heatmap for the predicted severity class, overlays it on the image, and saves it.

    Args:
        image_path (str): Path to the input X-ray image (PNG or JPG).
        model_path (str): Path to the model checkpoints file.

    Returns:
        str: Absolute or relative path to the saved heatmap overlay image.
    """
    if model_path is None:
        model_path = str(MODEL_CHECKPOINT_PATH)
    # 1. Error handling for missing model file
    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(
            f"Trained model checkpoint not found at: {model_file.absolute()}.\n"
            "Please run train_stage2.py first to fine-tune the model."
        )

    # 2. Setup device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 3. Initialize model and load state dictionary
    model = FractureModel(pretrained=False)
    try:
        checkpoint = torch.load(model_file, map_location=device)
        state_dict = checkpoint["model_state_dict"]

        # Check if this is the old single-head checkpoint
        is_old_checkpoint = "backbone.fc.1.weight" in state_dict and "severity_head.0.weight" not in state_dict

        if is_old_checkpoint:
            model.severity_head = torch.nn.Sequential(torch.nn.Identity(), torch.nn.Linear(2048, 4))
            new_state_dict = {}
            for k, v in state_dict.items():
                if k == "backbone.fc.1.weight":
                    new_state_dict["severity_head.1.weight"] = v
                elif k == "backbone.fc.1.bias":
                    new_state_dict["severity_head.1.bias"] = v
                else:
                    new_state_dict[k] = v
            state_dict = new_state_dict

        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()
    except Exception as e:
        raise RuntimeError(f"Failed to load weights into FractureModel: {e}")

    # 4. Preprocess input image (Grayscale -> CLAHE -> 3-channel RGB -> Resize -> Normalize)
    img_gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise FileNotFoundError(f"Failed to load input image: {image_path}")

    # Apply CLAHE contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_enhanced = clahe.apply(img_gray)

    # Convert to 3-channel RGB and resize
    img_rgb = cv2.cvtColor(img_enhanced, cv2.COLOR_GRAY2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)

    # Apply standard inference transform (normalization + ToTensorV2)
    transform = get_inference_transform()
    transformed = transform(image=img_resized)
    input_tensor = transformed["image"].unsqueeze(0).to(device)  # Shape: (1, 3, 224, 224)

    # 5. Determine predicted severity category for CAM targeting
    with torch.no_grad():
        severity_logits, _ = model(input_tensor)
        predicted_idx = int(torch.argmax(severity_logits, dim=1).item())

    # 6. Setup Grad-CAM parameters
    # Target the last layer of the last block (layer4[-1])
    wrapped_model = ModelWrapper(model)
    target_layers = [wrapped_model.model.backbone.layer4[-1]]
    targets = [ClassifierOutputTarget(predicted_idx)]

    # 7. Generate CAM mask
    cam = GradCAM(model=wrapped_model, target_layers=target_layers)
    # Generate mask of shape (224, 224)
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]

    # 8. Overlay the heatmap on the float32 RGB original image
    rgb_img_float = np.float32(img_resized) / 255.0
    # show_cam_on_image defaults to cv2.COLORMAP_JET
    cam_image = show_cam_on_image(rgb_img_float, grayscale_cam, use_rgb=True, colormap=cv2.COLORMAP_JET)

    # Convert back to BGR for saving with OpenCV
    cam_image_bgr = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

    # 9. Save file to the heatmaps/ directory with a timestamped filename
    heatmaps_dir = Path("heatmaps")
    heatmaps_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    img_stem = Path(image_path).stem
    output_filename = f"heatmap_{img_stem}_{timestamp}.png"
    output_path = heatmaps_dir / output_filename

    cv2.imwrite(str(output_path), cam_image_bgr)

    return str(output_path)
