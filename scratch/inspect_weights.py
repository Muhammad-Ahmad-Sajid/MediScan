import torch

checkpoint_path = "checkpoints/fracatlas_best.pth"
checkpoint = torch.load(checkpoint_path, map_location="cpu")
state_dict = checkpoint["model_state_dict"]

weight = state_dict["backbone.fc.1.weight"]
bias = state_dict["backbone.fc.1.bias"]

print(f"Weight shape: {weight.shape}")
print(f"Bias shape: {bias.shape}")
