import argparse
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Add project root to python path to resolve src imports
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.config import MODEL_CHECKPOINT_PATH
from src.data_preparation.dataset import get_dataloaders
from src.model_training.model import BoneFractureClassifier


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Runs a single training epoch across all batches."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for idx, batch in enumerate(loader):
        # batch[0] is always the preprocessed image tensor
        # batch[1] is the target label (binary classification for MURA, severity for FracAtlas)
        tensors = batch[0].to(device)
        labels = batch[1].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(tensors)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track statistics
        running_loss += loss.item() * tensors.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Print progress logs every 15 batches (or last batch)
        if (idx + 1) % 15 == 0 or (idx + 1) == len(loader):
            current_acc = 100.0 * correct / total
            print(
                f"  Batch {idx+1}/{len(loader)} - Loss: {loss.item():.4f} | Acc: {current_acc:.2f}%"
            )

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    """Evaluates the model performance on the validation set."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            tensors = batch[0].to(device)
            labels = batch[1].to(device)

            outputs = model(tensors)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * tensors.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc


def main():
    parser = argparse.ArgumentParser(description="Train Bone Fracture Detection Models")
    parser.add_argument(
        "--dataset",
        type=str,
        default="fracatlas",
        choices=["mura", "fracatlas"],
        help="Select the target dataset to train on ('mura' or 'fracatlas')",
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for loaders"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
    args = parser.parse_args()

    # Configure device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training Environment Initialized.")
    print(f"Device: {device}")

    # Load loaders (set num_workers=0 on Windows to prevent multiprocessing DLL spawn errors)
    import os

    num_workers = 0 if os.name == "nt" else 4
    loaders = get_dataloaders(batch_size=args.batch_size, num_workers=num_workers)

    if args.dataset == "mura":
        train_loader = loaders["mura_train"]
        val_loader = loaders["mura_val"]
        num_classes = 2
        checkpoint_name = "mura_best.pth"
    else:
        train_loader = loaders["frac_train"]
        val_loader = loaders["frac_val"]
        num_classes = 4
        checkpoint_name = "fracatlas_best.pth"

    print("\nModel Configuration:")
    print(f"  Dataset: {args.dataset.upper()}")
    print(f"  Target Classes: {num_classes}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.lr}")

    # Initialize classification model
    model = BoneFractureClassifier(num_classes=num_classes, pretrained=True).to(device)

    # Define optimization criterion and optimizer (AdamW matches ResNet well)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Plateau learning rate decay
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, threshold=0.001
    )

    # Verify save folder path
    checkpoint_dir = MODEL_CHECKPOINT_PATH.parent
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = checkpoint_dir / checkpoint_name

    best_val_loss = float("inf")

    # Epoch Loop
    for epoch in range(1, args.epochs + 1):
        print("\n--- Epoch {epoch}/{args.epochs} ---")

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        print(f"  -> Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100.0:.2f}%")

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"  -> Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc*100.0:.2f}%")

        # Update scheduler
        scheduler.step(val_loss)

        # Check validation improvement to save checkpoint weights
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "dataset": args.dataset,
                    "num_classes": num_classes,
                },
                best_checkpoint_path,
            )
            print(
                f"  *** Val Loss improved! Saved checkpoint to: {best_checkpoint_path.name} ***"
            )

    print("\n" + "=" * 50)
    print(f"TRAINING COMPLETED FOR {args.dataset.upper()}!")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
