import os
import gc
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.models as models
from torchvision.models import ResNet50_Weights
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Force single-threaded execution for OpenMP, MKL, OpenBLAS, NumExpr, and OpenCV on CPU
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENCV_NUM_THREADS"] = "0"
cv2.setNumThreads(0)
torch.set_num_threads(1)


# ------------------------------------------------------------------------------
# MURA Dataset Class
# ------------------------------------------------------------------------------
class MURAPretrainDataset(Dataset):
    """
    Dataset class for MURA stage 1 pretraining.
    Loads files from data/mura/{split}/{normal|abnormal}/
    """

    def __init__(self, base_dir: Path, split: str, transform=None):
        self.split_dir = base_dir / split
        self.transform = transform
        self.samples = []

        if not self.split_dir.exists():
            raise FileNotFoundError(
                f"MURA directory split not found at: {self.split_dir}"
            )

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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # 1. Load grayscale
        img_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            raise FileNotFoundError(f"Failed to load image: {img_path}")

        # 2. Apply CLAHE using OpenCV (clipLimit=2.0, tileGridSize=(8,8))
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img_gray)

        # 3. Convert to 3-channel RGB
        img_rgb = cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2RGB)

        # 4. Resize to 224x224
        img_resized = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)

        # 5. Apply augmentations & normalization
        if self.transform:
            augmented = self.transform(image=img_resized)
            img_tensor = augmented["image"]
        else:
            img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0

        return img_tensor, label


# ------------------------------------------------------------------------------
# Model Definition
# ------------------------------------------------------------------------------
class MuraResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        # Load backbone with pre-trained ImageNet weights
        self.backbone = models.resnet50(weights=ResNet50_Weights.DEFAULT)

        # Replace the final fully connected layer to output a single logit for binary classification
        self.backbone.fc = nn.Sequential(
            nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(p=0.4), nn.Linear(512, 1)
        )

    def forward(self, x):
        return self.backbone(x)

    def freeze_lower_layers(self):
        """Freezes all layers except layer3, layer4, and the fc classifier head."""
        for param in self.parameters():
            param.requires_grad = False

        # Unfreeze layer3, layer4, and fc head
        for name, child in self.backbone.named_children():
            if name in ["layer3", "layer4", "fc"]:
                for param in child.parameters():
                    param.requires_grad = True
        print(
            "[*] Frozen lower layers: only layer3, layer4, and fc head are trainable."
        )

    def unfreeze_all(self):
        """Unfreezes all layers for end-to-end training."""
        for param in self.parameters():
            param.requires_grad = True
        print("[*] Unfrozen all layers: entire backbone is now trainable.")


# ------------------------------------------------------------------------------
# Training and Validation Epochs
# ------------------------------------------------------------------------------
def train_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    epoch,
    start_batch=0,
    save_freq=500,
    checkpoint_dir=None,
    best_val_loss=None,
    early_stop_counter=None,
):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for idx, (images, labels) in enumerate(loader):
        # Skip batches if resuming mid-epoch
        if idx < start_batch:
            continue

        images = images.to(device)
        labels = (
            labels.to(device).float().unsqueeze(1)
        )  # Shape: (B, 1) for BCEWithLogitsLoss

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        # Calculate binary accuracy (threshold outputs at logit 0.0)
        preds = (outputs > 0.0).float()
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()

        if (idx + 1) % 100 == 0 or (idx + 1) == len(loader):
            print(
                f"  Batch {idx+1}/{len(loader)} | Loss: {loss.item():.4f} | Acc: {100.0 * correct / total:.2f}%"
            )
            gc.collect()

        # Save mid-epoch checkpoint
        if (
            save_freq > 0
            and (idx + 1) % save_freq == 0
            and (idx + 1) < len(loader)
            and checkpoint_dir is not None
        ):
            latest_checkpoint_path = checkpoint_dir / "mura_latest.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "batch_idx": idx,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": 0.0,
                    "val_acc": 0.0,
                    "best_val_loss": (
                        best_val_loss if best_val_loss is not None else float("in")
                    ),
                    "early_stop_counter": (
                        early_stop_counter if early_stop_counter is not None else 0
                    ),
                },
                latest_checkpoint_path,
            )
            print(
                f"  [Checkpoint] Saved mid-epoch state at Batch {idx+1}/{len(loader)} to: {latest_checkpoint_path.name}"
            )
            gc.collect()

        del images, labels, outputs, loss

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def val_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = (outputs > 0.0).float()
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()
            del images, labels, outputs, loss

    gc.collect()
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


# ------------------------------------------------------------------------------
# Main Training Loop
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="MURA Stage 1 Pretraining")
    parser.add_argument(
        "--epochs", type=int, default=25, help="Total number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the last saved best checkpoint",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("STAGE 1 MURA PRETRAINING (BINARY CLASSIFICATION)")
    print(f"Device: {device}")
    print("=" * 80)

    # 1. Transform definitions using Albumentations
    train_transform = A.Compose(
        [
            A.Rotate(limit=10, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )

    val_transform = A.Compose(
        [
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )

    # 2. Datasets
    data_dir = Path("data/mura")
    train_dataset = MURAPretrainDataset(data_dir, "train", transform=train_transform)
    val_dataset = MURAPretrainDataset(data_dir, "val", transform=val_transform)

    print(f"Train size: {len(train_dataset)} | Val size: {len(val_dataset)}")

    # 3. Handle class imbalance using WeightedRandomSampler on the training set
    train_labels = [sample[1] for sample in train_dataset.samples]
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in train_labels]

    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    # Note: set num_workers=0 on Windows to prevent multiprocessing errors
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    # 4. Model initialization
    model = MuraResNet50().to(device)

    # First 5 epochs: freeze lower layers by default
    model.freeze_lower_layers()

    # 5. Define loss criterion (BCEWithLogitsLoss with pos_weight)
    train_normal_count = class_counts[0]
    train_abnormal_count = class_counts[1]
    pos_weight_value = train_normal_count / train_abnormal_count
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float, device=device)
    print(f"Applying BCEWithLogitsLoss pos_weight: {pos_weight.item():.4f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_loss = float("in")
    start_epoch = 1
    start_batch = 0
    early_stop_patience = 5
    early_stop_counter = 0

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = checkpoint_dir / "mura_pretrained.pth"
    latest_checkpoint_path = checkpoint_dir / "mura_latest.pth"

    # Setup default optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    # 6. Resume from checkpoint if requested
    if args.resume:
        checkpoint_path = None
        if latest_checkpoint_path.exists():
            checkpoint_path = latest_checkpoint_path
            print(f"[*] Resuming from latest checkpoint: {latest_checkpoint_path}")
        elif best_checkpoint_path.exists():
            checkpoint_path = best_checkpoint_path
            print(f"[*] Resuming from best checkpoint: {best_checkpoint_path}")

        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            state_dict = checkpoint["model_state_dict"]
            model_state = model.state_dict()
            filtered_state_dict = {}
            shape_mismatch_detected = False
            for k, v in state_dict.items():
                if k in model_state:
                    if v.shape == model_state[k].shape:
                        filtered_state_dict[k] = v
                    else:
                        print(
                            f"[*] Skipping key {k} due to shape mismatch: checkpoint shape {v.shape} vs model shape {model_state[k].shape}"
                        )
                        shape_mismatch_detected = True
                else:
                    print(f"[*] Skipping unexpected key {k} from checkpoint")
            model.load_state_dict(filtered_state_dict, strict=False)
            print(
                "[*] Successfully loaded model weights (transferred backbone features, skipped mismatched fc head)."
            )

            # Check if checkpoint was saved mid-epoch
            batch_idx = checkpoint.get("batch_idx", None)
            if batch_idx is not None:
                start_epoch = checkpoint["epoch"]
                start_batch = batch_idx + 1
                print(
                    f"[*] Successfully loaded mid-epoch checkpoint. Resuming from Epoch {start_epoch}, Batch {start_batch}"
                )
            else:
                start_epoch = checkpoint["epoch"] + 1
                start_batch = 0
                print(
                    f"[*] Successfully loaded checkpoint. Resuming from Epoch {start_epoch}"
                )

            best_val_loss = checkpoint.get(
                "best_val_loss", checkpoint.get("val_loss", float("in"))
            )
            early_stop_counter = checkpoint.get("early_stop_counter", 0)

            # Align optimizer/scheduler state
            if "optimizer_state_dict" in checkpoint and not shape_mismatch_detected:
                try:
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                    print("[*] Successfully loaded optimizer state.")
                except Exception as e:
                    print(f"[!] Warning: Could not load optimizer state: {e}")
            else:
                print(
                    "[*] Optimizer state loading skipped to prevent shape mismatch runtime errors."
                )
        else:
            print(
                "[!] Warning: No checkpoint found to resume from. Starting from scratch."
            )

    # 7. Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")

        # Freezing/Unfreezing rules
        if epoch >= 6:
            model.unfreeze_all()
            # If we transitioned to epoch >= 6, lower the learning rate as per training plan
            for param_group in optimizer.param_groups:
                param_group["lr"] = args.lr * 0.1
        else:
            model.freeze_lower_layers()
            # Maintain default learning rate for frozen stage
            for param_group in optimizer.param_groups:
                param_group["lr"] = args.lr

        # Determine start batch (only for the very first epoch after resuming)
        current_start_batch = start_batch if epoch == start_epoch else 0

        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch=epoch,
            start_batch=current_start_batch,
            save_freq=500,
            checkpoint_dir=checkpoint_dir,
            best_val_loss=best_val_loss,
            early_stop_counter=early_stop_counter,
        )
        print(f"  -> Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100.0:.2f}%")

        val_loss, val_acc = val_epoch(model, val_loader, criterion, device)
        print(f"  -> Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc*100.0:.2f}%")

        # Step the ReduceLROnPlateau scheduler based on val loss
        scheduler.step(val_loss)

        # Save latest model checkpoint at the end of every epoch
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "best_val_loss": best_val_loss,
                "early_stop_counter": early_stop_counter,
            },
            latest_checkpoint_path,
        )
        print(f"  Saved latest model state to: {latest_checkpoint_path.name}")

        # Save best model checkpoint & manage early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "best_val_loss": best_val_loss,
                },
                best_checkpoint_path,
            )
            print(
                f"  *** Val Loss improved! Saved best stage 1 model to: {best_checkpoint_path.name} ***"
            )
        else:
            early_stop_counter += 1
            print(
                f"  Early stopping counter: {early_stop_counter}/{early_stop_patience}"
            )

        if early_stop_counter >= early_stop_patience:
            print(
                f"\n[!] Early stopping triggered. Validation loss has not improved for {early_stop_patience} epochs."
            )
            break

    print("\n" + "=" * 80)
    print("STAGE 1 PRETRAINING COMPLETED!")
    print(f"Saved Checkpoint path: {best_checkpoint_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
