import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet50()
        self.backbone.fc = nn.Linear(2048, 2)

    def forward(self, x):
        return self.backbone(x)


def main():
    print("[*] Initializing dummy model and optimizer...")
    model = DummyModel().cpu()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    x = torch.randn(4, 3, 224, 224)
    y = torch.randint(0, 2, (4,))

    print("[*] Running forward pass under CPU BFloat16 autocast...")
    try:
        with torch.cpu.amp.autocast(dtype=torch.bfloat16):
            outputs = model(x)
            loss = criterion(outputs, y)
        print(f"[*] Forward pass successful. Loss: {loss.item()}")

        print("[*] Running backward pass...")
        loss.backward()
        print("[*] Backward pass successful.")

        print("[*] Running optimizer step...")
        optimizer.step()
        print("[*] Optimizer step successful! BFloat16 is fully supported.")
    except Exception:
        import traceback

        print("[!] Error encountered:")
        traceback.print_exc()


if __name__ == "__main__":
    main()
