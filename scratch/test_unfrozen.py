import os

# Force single-threaded execution
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENCV_NUM_THREADS"] = "0"
os.environ["KMP_BLOCKTIME"] = "0"

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

# Set threads
torch.set_num_threads(1)


class MuraResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet50()
        self.backbone.fc = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(p=0.4), nn.Linear(512, 2))

    def forward(self, x):
        return self.backbone(x)

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True


def main():
    print("[*] Starting test script...")
    device = torch.device("cpu")

    print("[*] Initializing model...")
    model = MuraResNet50().to(device)

    checkpoint_path = "checkpoints/mura_latest.pth"
    if os.path.exists(checkpoint_path):
        print(f"[*] Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("[*] Checkpoint loaded successfully.")

    print("[*] Unfreezing all layers...")
    model.unfreeze_all()

    print("[*] Creating optimizer...")
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()

    print("[*] Creating dummy input...")
    x = torch.randn(2, 3, 224, 224, device=device)
    y = torch.randint(0, 2, (2,), device=device)

    print("[*] Running FORWARD pass...")
    outputs = model(x)
    print("[*] Forward pass completed.")

    print("[*] Computing loss...")
    loss = criterion(outputs, y)
    print(f"[*] Loss: {loss.item():.4f}")

    print("[*] Running BACKWARD pass...")
    loss.backward()
    print("[*] Backward pass completed.")

    print("[*] Running OPTIMIZER step...")
    optimizer.step()
    print("[*] Optimizer step completed.")

    print("[*] Test completed successfully!")


if __name__ == "__main__":
    main()
