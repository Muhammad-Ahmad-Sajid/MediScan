import sys
from pathlib import Path
import torch
import torch.nn as nn

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.model_training.model import FractureModel

checkpoint_path = "checkpoints/fracatlas_best.pth"
checkpoint = torch.load(checkpoint_path, map_location="cpu")
state_dict = checkpoint["model_state_dict"]

# Check if it is the old single-head checkpoint
is_old_checkpoint = "backbone.fc.1.weight" in state_dict and "severity_head.0.weight" not in state_dict
print(f"Is old single-head checkpoint: {is_old_checkpoint}")

model = FractureModel(pretrained=False)

if is_old_checkpoint:
    # Dynamically adapt severity_head to match the checkpoint's classification layer
    model.severity_head = nn.Sequential(nn.Identity(), nn.Linear(2048, 4))

    # Map the keys in state_dict
    new_state_dict = {}
    for k, v in state_dict.items():
        if k == "backbone.fc.1.weight":
            new_state_dict["severity_head.1.weight"] = v
        elif k == "backbone.fc.1.bias":
            new_state_dict["severity_head.1.bias"] = v
        else:
            new_state_dict[k] = v
    state_dict = new_state_dict

# Load the model state dict
try:
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("Success loading model!")
    print(f"Missing keys (should not contain backbone layers): {len(missing)}")
    print(f"Unexpected keys: {len(unexpected)}")

    # Verify that the severity head is loaded
    # Check if weights are not zero/random
    weight_sum = model.severity_head[1].weight.sum().item()
    print(f"Severity head weights sum: {weight_sum:.4f}")
except Exception as e:
    print(f"Error loading model: {e}")
