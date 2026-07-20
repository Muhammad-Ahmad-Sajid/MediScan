from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.config import HEATMAP_OUTPUT_FOLDER
from src.data_preparation.preprocess import get_inference_transform


def generate_gradcam_heatmap(
    image_path: str,
    model: nn.Module,
    device: torch.device,
    target_category: int = None,
    output_filename: str = None,
) -> Path:
    """
    Generates a Grad-CAM heatmap overlay for a bone scan image.

    Args:
        image_path (str): Path to the input X-ray image (PNG or JPG).
        model (nn.Module): Preloaded PyTorch classification model.
        device (torch.device): Device to run inference on (cuda or cpu).
        target_category (int, optional): The target class index for Grad-CAM.
                                        If None, defaults to the highest logit class.
        output_filename (str, optional): Target filename for the output overlay.
                                         If None, a filename is derived from the input.

    Returns:
        Path: The file path to the saved heatmap overlay image.
    """
    # 1. Load the original image and apply enhancement (CLAHE)
    img_gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise FileNotFoundError(f"Failed to read image for Grad-CAM: {image_path}")

    # Apply CLAHE to enhance bone features (matches preprocessing)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_enhanced = clahe.apply(img_gray)

    # Convert to 3-channel RGB
    img_rgb = cv2.cvtColor(img_enhanced, cv2.COLOR_GRAY2RGB)

    # Resize to 224x224 (ResNet size)
    img_resized = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)

    # Create normalized input tensor for the model
    transform = get_inference_transform()
    # transform expects a numpy image and returns a dict with 'image' tensor
    transformed = transform(image=img_resized)
    input_tensor = (
        transformed["image"].unsqueeze(0).to(device)
    )  # Shape: (1, 3, 224, 224)

    # 2. Setup Grad-CAM target layers
    # For ResNet-50, the target layer is the final convolutional layer of layer4
    target_layers = [model.backbone.layer4[-1]]

    # 3. Instantiate Grad-CAM extractor
    cam = GradCAM(model=model, target_layers=target_layers)

    # 4. Determine target prediction category
    if target_category is None:
        # Run a quick forward pass to determine the highest logit class
        model.eval()
        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = outputs.max(1)
            target_category = int(predicted.item())

    targets = [ClassifierOutputTarget(target_category)]

    # 5. Generate CAM mask
    # cam() returns a batch of activation maps as float32 in [0, 1]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]

    # 6. Overlay CAM mask on the unnormalized float32 RGB image
    # Scale image to [0, 1] for show_cam_on_image
    rgb_img_float = np.float32(img_resized) / 255.0
    cam_image = show_cam_on_image(rgb_img_float, grayscale_cam, use_rgb=True)

    # Convert back to BGR (OpenCV format) for saving
    cam_image_bgr = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

    # 7. Save the heatmap overlay image
    HEATMAP_OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    if output_filename is None:
        # Construct unique name: prefix_image.png
        prefix = f"heatmap_cls_{target_category}_"
        original_name = Path(image_path).name
        output_filename = f"{prefix}{original_name}"
        if not output_filename.endswith(".png"):
            # Force save as png for clean overlay format
            output_filename = f"{Path(output_filename).stem}.png"

    output_path = HEATMAP_OUTPUT_FOLDER / output_filename
    cv2.imwrite(str(output_path), cam_image_bgr)

    print(f"Grad-CAM Heatmap overlay successfully saved to: {output_path}")
    return output_path
