import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transform():
    """
    Returns the Albumentations transform pipeline for training.
    Applies flips, rotation, brightness/contrast jitter, noise, normalization, and converts to tensor.
    """
    return A.Compose(
        [
            A.Rotate(limit=10, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )


def get_inference_transform():
    """
    Returns the Albumentations transform pipeline for validation/inference.
    Applies only normalization and converts to tensor.
    """
    return A.Compose(
        [
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )


def preprocess_xray(image_path: str, is_training: bool = True) -> torch.Tensor:
    """
    Complete preprocessing pipeline for an X-ray image (PNG or JPG).

    1. Load image as grayscale.
    2. Apply CLAHE to enhance bone contrast.
    3. Convert to 3-channel RGB (by repeating the grayscale channel).
    4. Resize to 224x224.
    5. Apply Albumentations augmentation (if is_training=True) & Normalization.
    6. Return a PyTorch tensor.
    """
    # 1. Load image as grayscale
    img_gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")

    # 2. Apply CLAHE using OpenCV (clipLimit=2.0, tileGridSize=8x8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_gray)

    # 3. Convert to 3-channel RGB by repeating the grayscale channel
    img_rgb = cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2RGB)

    # 4. Resize to 224x224
    img_resized = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)

    # 5. Apply transforms
    transform = get_train_transform() if is_training else get_inference_transform()
    augmented = transform(image=img_resized)

    return augmented["image"]
