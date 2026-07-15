import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from inference import run_inference
import pandas as pd

# Load labels and find the first fractured leg scan
df = pd.read_csv("fracatlas_labels.csv")
fractured_scans = df[df["has_fracture"] is True].reset_index(drop=True)

if len(fractured_scans) == 0:
    print("[!] No fractured scans found in labels CSV!")
    sys.exit(1)

# Let's find one leg fracture scan
leg_scans = fractured_scans[fractured_scans["bone_affected"] == "leg"].reset_index(drop=True)
if len(leg_scans) > 0:
    sample_scan = leg_scans.iloc[0]
else:
    sample_scan = fractured_scans.iloc[0]

img_path = sample_scan["image_path"]
print(f"[*] Selected scan for testing: {img_path}")
print(
    f"[*] Expected bone: {sample_scan['bone_affected']}, Expected severity: {sample_scan['severity_class']}"
)

if not Path(img_path).exists():
    print(f"[!] Image path does not exist: {img_path}")
    sys.exit(1)

try:
    print("[*] Running inference...")
    res = run_inference(img_path)
    print("\n" + "=" * 60)
    print("INFERENCE VERIFICATION RESULTS")
    print("=" * 60)
    print(f"Fracture Detected:  {res.fracture_detected}")
    print(f"Predicted Severity: {res.severity} (Confidence: {res.severity_confidence:.2f})")
    print(f"Predicted Bone:     {res.bone_affected} (Confidence: {res.bone_confidence:.2f})")
    print(f"Heatmap Saved To:   {res.heatmap_path}")
    print("=" * 60)

    if res.heatmap_path and Path(res.heatmap_path).exists():
        print("[*] Success: Grad-CAM heatmap file verified on disk!")
    else:
        print("[!] Warning: Grad-CAM heatmap file was not saved successfully.")

except Exception as e:
    print(f"[!] Error running inference: {e}")
    sys.exit(1)
