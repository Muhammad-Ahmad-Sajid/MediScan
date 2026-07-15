import cv2
import numpy as np
import torch
import pytest
from pathlib import Path

# Add project root to sys.path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data_preparation.preprocess import preprocess_xray


@pytest.fixture
def temp_images_dir(tmp_path):
    """Provides a temporary path for creating synthetic image assets."""
    return tmp_path


def create_synthetic_grayscale_image(path, width=256, height=256):
    """Generates a random grayscale image and saves it to the given path."""
    # Create random noise image (0 to 255)
    img = np.random.randint(0, 256, (height, width), dtype=np.uint8)
    cv2.imwrite(str(path), img)
    return img


def test_png_preprocessing_shape_and_dtype(temp_images_dir):
    """Verifies that preprocessing a PNG returns a float32 torch tensor of shape (3, 224, 224)."""
    png_path = temp_images_dir / "synthetic_test.png"
    create_synthetic_grayscale_image(png_path)

    tensor = preprocess_xray(str(png_path), is_training=False)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32


def test_jpg_preprocessing_shape_and_dtype(temp_images_dir):
    """Verifies that preprocessing a JPG returns a float32 torch tensor of shape (3, 224, 224)."""
    jpg_path = temp_images_dir / "synthetic_test.jpg"
    create_synthetic_grayscale_image(jpg_path)

    tensor = preprocess_xray(str(jpg_path), is_training=False)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32


def test_imagenet_normalization_ranges(temp_images_dir):
    """Verifies that pixel values are normalized to roughly the ImageNet range, not raw 0-255."""
    png_path = temp_images_dir / "synthetic_test.png"
    create_synthetic_grayscale_image(png_path)

    tensor = preprocess_xray(str(png_path), is_training=False)

    # ImageNet normalization maps [0, 1] range to roughly [-2.2, 2.7] range.
    # We assert that the values are centered and scaled in this range.
    assert tensor.min() >= -3.0
    assert tensor.max() <= 3.0
    # Raw pixels (0-255) would have min >= 0 and max > 1.0.
    # ImageNet normalized pixels will always yield negative values for low pixels and positive for high.
    assert (tensor < 0).any()
    assert (tensor > 0).any()


def test_training_transform_augmentations_stochastic(temp_images_dir):
    """Verifies that training transform applies stochastic augmentations (different results on subsequent calls)."""
    png_path = temp_images_dir / "synthetic_test.png"
    create_synthetic_grayscale_image(png_path)

    # Run twice with training=True. Augmentations like Rotate, GaussNoise should trigger
    tensor1 = preprocess_xray(str(png_path), is_training=True)

    # Retry up to 10 times to account for the probability that no augmentations fire
    different = False
    for _ in range(10):
        tensor2 = preprocess_xray(str(png_path), is_training=True)
        if not torch.equal(tensor1, tensor2):
            different = True
            break

    assert different


def test_inference_transform_is_deterministic(temp_images_dir):
    """Verifies that inference transform is completely deterministic (same tensor on subsequent calls)."""
    png_path = temp_images_dir / "synthetic_test.png"
    create_synthetic_grayscale_image(png_path)

    tensor1 = preprocess_xray(str(png_path), is_training=False)
    tensor2 = preprocess_xray(str(png_path), is_training=False)

    # Tensors must be exactly identical
    assert torch.equal(tensor1, tensor2)
