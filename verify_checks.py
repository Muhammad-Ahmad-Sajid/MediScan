import os
import torch
import glob
from osteoporosis_inference import run_inference as run_osteoporosis_inference

print("Check 1 - Verify osteoporosis checkpoint exists and is valid")
checkpoint_path = "d:/X-ray ML Model/checkpoints/osteoporosis_best.pth"

# Check file exists
if os.path.exists(checkpoint_path):
    size_mb = os.path.getsize(checkpoint_path) / 1024 / 1024
    print(f"[OK] Checkpoint found: {size_mb:.1f} MB")
else:
    print("[FAIL] Checkpoint NOT found")

# Check it loads correctly
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
if "epoch" in checkpoint:
    print(f"[OK] Epoch saved: {checkpoint['epoch']}")
    print(f"[OK] Val accuracy: {checkpoint['val_acc']:.2f}%")
    print(f"[OK] Val loss: {checkpoint['val_loss']:.4f}")
    print(f"[OK] Keys in checkpoint: {list(checkpoint.keys())}")
else:
    print("[OK] Checkpoint contains raw state_dict (which is standard for inference).")
    print(f"[OK] Number of layer weights saved: {len(checkpoint.keys())}")

print("\nCheck 2 - Run a real inference test on actual osteoporosis image")
# Dynamically grab an image from the dataset
test_images = glob.glob("d:/X-ray ML Model/Mediscan/osteoporosis/osteoporosis/*.*")
if test_images:
    test_image = test_images[0]
else:
    test_image = ""

result = run_osteoporosis_inference(test_image)

print("=" * 50)
print("OSTEOPOROSIS INFERENCE CHECK")
print("=" * 50)
print(f"Has Osteoporosis  : {result.has_osteoporosis}")
# Note: confidence is already returned as a percentage from our module
print(f"Confidence        : {result.confidence:.2f}%")
print(f"Confidence Flag   : {result.confidence_flag}")
print(f"Heatmap Path      : {result.heatmap_path}")
print(f"Recommendation    : {result.clinical_recommendation[:60]}...")
print(f"Prediction Time   : {result.prediction_time_ms:.1f}ms")
print(f"Model Version     : {result.model_version}")
print("=" * 50)

# Verify heatmap was actually saved
if result.heatmap_path:
    # Resolve absolute path for checking
    abs_heatmap = os.path.join("d:/X-ray ML Model", result.heatmap_path)
    if os.path.exists(abs_heatmap):
        size_kb = os.path.getsize(abs_heatmap) / 1024
        print(f"[OK] Heatmap saved: {size_kb:.1f} KB")
    else:
        print("[WARN] Heatmap path returned, but file not found on disk")
else:
    print("[WARN] Heatmap not generated (may be inconclusive)")
