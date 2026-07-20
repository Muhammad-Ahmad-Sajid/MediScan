import torch

checkpoint_path = "checkpoints/fracatlas_best.pth"
checkpoint = torch.load(checkpoint_path, map_location="cpu")
state_dict = checkpoint["model_state_dict"]

non_layer_keys = [
    k
    for k in state_dict.keys()
    if not k.startswith("backbone.layer") and not k.startswith("backbone.conv") and not k.startswith("backbone.bn")
]
print("Non-backbone-layer keys in state_dict:")
print(non_layer_keys)
