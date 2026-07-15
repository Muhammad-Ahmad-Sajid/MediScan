import os
from pathlib import Path

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
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import cv2

cv2.setNumThreads(0)
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Set threads
torch.set_num_threads(1)


class MURAPretrainDataset(Dataset):
    def __init__(self, base_dir: Path, split: str, transform=None):
        self.split_dir = base_dir / split
        self.transform = transform
        self.samples = []
        for label_name, label_idx in [("normal", 0), ("abnormal", 1)]:
            class_dir = self.split_dir / label_name
            if class_dir.exists():
                for file in class_dir.iterdir():
                    if file.suffix.lower() in [
                        ".png",
                        ".jpg",
                        ".jpeg",
                    ] and not file.name.startswith("._"):
                        self.samples.append((file, label_idx))
                        if len(self.samples) >= 100:  # limit to 100 samples for quick test
                            break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            raise FileNotFoundError(f"Failed to load image: {img_path}")
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img_gray)
        img_rgb = cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2RGB)
        img_resized = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
        if self.transform:
            augmented = self.transform(image=img_resized)
            img_tensor = augmented["image"]
        else:
            img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        return img_tensor, label


class MuraResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet50()
        self.backbone.fc = nn.Sequential(
            nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(p=0.4), nn.Linear(512, 2)
        )

    def forward(self, x):
        return self.backbone(x)

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True


def main():
    print("[*] Starting dataloader test script...")
    device = torch.device("cpu")

    print("[*] Setting up transforms...")
    train_transform = A.Compose(
        [
            A.Rotate(limit=10, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )

    print("[*] Initializing dataset...")
    data_dir = Path("d:/X-ray ML Model/data/mura")
    dataset = MURAPretrainDataset(data_dir, "train", transform=train_transform)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
    print(f"[*] Dataset initialized with {len(dataset)} samples.")

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

    print("[*] Starting training loop simulation for 5 batches...")
    for idx, (images, labels) in enumerate(loader):
        print(f"\n--- Batch {idx+1} ---")
        print("  Loading batch tensors...")
        images, labels = images.to(device), labels.to(device)

        print("  Running FORWARD pass...")
        outputs = model(images)

        print("  Computing loss...")
        loss = criterion(outputs, labels)
        print(f"  Loss: {loss.item():.4f}")

        print("  Running BACKWARD pass...")
        optimizer.zero_grad()
        loss.backward()

        print("  Running OPTIMIZER step...")
        optimizer.step()

        print(f"--- Batch {idx+1} completed successfully! ---")
        if idx + 1 >= 5:
            break

    print("\n[*] Dataloader test completed successfully!")


if __name__ == "__main__":
    main()
