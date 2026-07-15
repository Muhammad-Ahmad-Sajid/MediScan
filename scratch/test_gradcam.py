import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from gradcam import generate_heatmap

# Create a dummy image
import cv2
import numpy as np

dummy_img_path = "scratch/dummy_test.png"
cv2.imwrite(dummy_img_path, np.zeros((300, 300), dtype=np.uint8))

try:
    path = generate_heatmap(dummy_img_path, "checkpoints/fracatlas_best.pth")
    print(f"Success! Heatmap saved to: {path}")
    if Path(path).exists():
        print("Confirmed: Heatmap file exists on disk!")
except Exception as e:
    print(f"Error occurred calling generate_heatmap: {e}")
