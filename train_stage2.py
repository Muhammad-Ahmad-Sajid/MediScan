import os
import sys
import gc
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
from sklearn.metrics import classification_report, confusion_matrix

# 1. Windows CPU Multithreading Optimization to prevent c10.dll / OpenMP crashes
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
torch.set_num_threads(1)

# Ensure root path is in sys.path to resolve src.* packages
sys.path.append(str(Path(__file__).resolve().parent))

from src.data_preparation.dataset import get_stage2_dataloaders
from src.model_training.model import Stage2FractureModel

FRACTURE_CLASSES = ["not_fractured", "fractured"]
REGION_CLASSES = ["hand", "leg", "hip", "shoulder", "unknown"]


def train_one_epoch(
    model,
    loader,
    criterion_frac,
    criterion_region,
    optimizer,
    device,
    epoch,
    start_batch=0,
    save_freq=300,
    checkpoint_dir=None,
    best_val_loss=float("inf"),
    early_stop_counter=0,
    scheduler_state_dict=None,
):
    model.train()
    running_loss = 0.0
    correct_frac = 0
    correct_region = 0
    total = 0

    for idx, batch in enumerate(loader):
        # Skip batches if resuming mid-epoch
        if idx < start_batch:
            continue

        tensors = batch[0].to(device)
        labels_frac = batch[1].to(device)  # fracture target (0 or 1)
        labels_region = batch[2].to(device)  # region target index (0-4)

        optimizer.zero_grad()

        # Forward pass: (fracture_logits, region_logits)
        outputs_frac, outputs_region = model(tensors)

        # Calculate individual losses
        loss_frac = criterion_frac(outputs_frac, labels_frac)
        loss_region = criterion_region(outputs_region, labels_region)

        # Combined multi-task loss (0.6 * fracture + 0.4 * region)
        total_loss = 0.6 * loss_frac + 0.4 * loss_region
        total_loss.backward()
        optimizer.step()

        # Accumulate metrics
        running_loss += total_loss.item() * tensors.size(0)
        _, predicted_frac = outputs_frac.max(1)
        _, predicted_region = outputs_region.max(1)

        total += labels_frac.size(0)
        correct_frac += predicted_frac.eq(labels_frac).sum().item()
        correct_region += predicted_region.eq(labels_region).sum().item()

        # Print progress
        if (idx + 1) % 10 == 0 or (idx + 1) == len(loader):
            print(
                f"  Batch {idx+1}/{len(loader)} - Loss: {total_loss.item():.4f} | Frac Acc: {100.0*correct_frac/total:.2f}% | Region Acc: {100.0*correct_region/total:.2f}%"
            )
            gc.collect()

        # Save mid-epoch checkpoint
        if (
            save_freq > 0
            and (idx + 1) % save_freq == 0
            and (idx + 1) < len(loader)
            and checkpoint_dir is not None
        ):
            latest_checkpoint_path = checkpoint_dir / "stage2_latest.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "batch_idx": idx,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler_state_dict,
                    "val_loss": 0.0,
                    "best_val_loss": best_val_loss,
                    "early_stop_counter": early_stop_counter,
                },
                latest_checkpoint_path,
            )
            print(
                f"  [Checkpoint] Saved mid-epoch state at Batch {idx+1}/{len(loader)} to: {latest_checkpoint_path.name}"
            )
            gc.collect()

        # Clean memory variables
        del (
            tensors,
            labels_frac,
            labels_region,
            outputs_frac,
            outputs_region,
            total_loss,
        )

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc_frac = correct_frac / total if total > 0 else 0.0
    epoch_acc_region = correct_region / total if total > 0 else 0.0

    return epoch_loss, epoch_acc_frac, epoch_acc_region


def validate(model, loader, criterion_frac, criterion_region, device):
    model.eval()
    running_loss = 0.0
    correct_frac = 0
    correct_region = 0
    total = 0

    all_true_frac = []
    all_pred_frac = []
    all_true_region = []
    all_pred_region = []

    with torch.no_grad():
        for batch in loader:
            tensors = batch[0].to(device)
            labels_frac = batch[1].to(device)
            labels_region = batch[2].to(device)

            outputs_frac, outputs_region = model(tensors)

            loss_frac = criterion_frac(outputs_frac, labels_frac)
            loss_region = criterion_region(outputs_region, labels_region)
            total_loss = 0.6 * loss_frac + 0.4 * loss_region

            running_loss += total_loss.item() * tensors.size(0)
            _, predicted_frac = outputs_frac.max(1)
            _, predicted_region = outputs_region.max(1)

            total += labels_frac.size(0)
            correct_frac += predicted_frac.eq(labels_frac).sum().item()
            correct_region += predicted_region.eq(labels_region).sum().item()

            all_true_frac.extend(labels_frac.cpu().numpy())
            all_pred_frac.extend(predicted_frac.cpu().numpy())
            all_true_region.extend(labels_region.cpu().numpy())
            all_pred_region.extend(predicted_region.cpu().numpy())

            del (
                tensors,
                labels_frac,
                labels_region,
                outputs_frac,
                outputs_region,
                total_loss,
            )

    val_loss = running_loss / total if total > 0 else 0.0
    val_acc_frac = correct_frac / total if total > 0 else 0.0
    val_acc_region = correct_region / total if total > 0 else 0.0

    report_data = {
        "true_frac": all_true_frac,
        "pred_frac": all_pred_frac,
        "true_region": all_true_region,
        "pred_region": all_pred_region,
    }

    gc.collect()
    return val_loss, val_acc_frac, val_acc_region, report_data


def main():
    parser = argparse.ArgumentParser(description="Stage 2 Fine-Tuning Training Script")
    parser.add_argument(
        "--epochs", type=int, default=8, help="Number of fine-tuning epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for training"
    )
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument(
        "--csv_path",
        type=str,
        default="d:/X-ray ML Model/stage2_labels.csv",
        help="Path to Stage 2 CSV labels",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the last saved latest checkpoint",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("STAGE 2 FINE-TUNING ON UNIFIED DATASET (MULTI-HEAD)")
    print(f"Device: {device}")
    print("=" * 80)

    # Load loaders (set num_workers=0 on Windows CPU)
    train_loader, val_loader = get_stage2_dataloaders(
        csv_path=args.csv_path, batch_size=args.batch_size, num_workers=0
    )

    # Initialize Stage 2 model
    # Check if pre-trained MURA backbone weights are available
    pretrained_mura_path = Path("checkpoints/mura_pretrained.pth")
    has_pretrained_mura = pretrained_mura_path.exists()

    # Load backbone with random or Imagenet weights first, then load MURA if available
    model = Stage2FractureModel(pretrained=not has_pretrained_mura)

    if has_pretrained_mura:
        print(
            f"[*] Loading pre-trained MURA backbone weights from {pretrained_mura_path}"
        )
        checkpoint = torch.load(pretrained_mura_path, map_location=device)
        # Load backbone weights with strict=False (ignores classification layer difference)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    else:
        print(
            f"[!] Warning: MURA pre-trained checkpoint not found at {pretrained_mura_path}."
        )

    model.unfreeze_all()
    model.to(device)

    # Setup loss functions and optimizer
    criterion_frac = nn.CrossEntropyLoss()
    criterion_region = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = checkpoint_dir / "stage2_best.pth"
    latest_checkpoint_path = checkpoint_dir / "stage2_latest.pth"

    best_val_loss = float("inf")
    start_epoch = 1
    start_batch = 0
    early_stop_patience = 5
    early_stop_counter = 0

    epoch_history = []
    best_epoch = 1
    best_fracture_acc = 0.0
    best_bone_acc = 0.0

    # Resume from checkpoint if requested
    if args.resume:
        if latest_checkpoint_path.exists():
            print(f"[*] Resuming from latest checkpoint: {latest_checkpoint_path}")
            checkpoint = torch.load(latest_checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])

            # Load optimizer state
            if "optimizer_state_dict" in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                    print("[*] Successfully loaded optimizer state.")
                except Exception as e:
                    print(f"[!] Warning: Could not load optimizer state: {e}")

            # Load scheduler state
            if (
                "scheduler_state_dict" in checkpoint
                and checkpoint["scheduler_state_dict"] is not None
            ):
                try:
                    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                    print("[*] Successfully loaded scheduler state.")
                except Exception as e:
                    print(f"[!] Warning: Could not load scheduler state: {e}")

            best_val_loss = checkpoint.get("best_val_loss", float("inf"))
            early_stop_counter = checkpoint.get("early_stop_counter", 0)
            best_epoch = checkpoint.get("best_epoch", checkpoint.get("epoch", 1))
            best_fracture_acc = checkpoint.get("best_fracture_acc", 0.0)
            best_bone_acc = checkpoint.get("best_bone_acc", 0.0)
            epoch_history = checkpoint.get("epoch_history", [])

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
                    f"[*] Successfully loaded epoch-end checkpoint. Resuming from Epoch {start_epoch}"
                )
        else:
            print(
                "[!] Warning: No checkpoint found to resume from. Starting from scratch."
            )

    start_time = time.time()

    # Training Loop
    for epoch in range(start_epoch, args.epochs + 1):

        print(f"\n--- Epoch {epoch}/{args.epochs} ---")

        current_start_batch = start_batch if epoch == start_epoch else 0
        if current_start_batch > 0:
            print(
                f"[*] Skipping first {current_start_batch} batches to resume training mid-epoch..."
            )

        # Train one epoch
        train_loss, train_acc_frac, train_acc_region = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion_frac=criterion_frac,
            criterion_region=criterion_region,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            start_batch=current_start_batch,
            save_freq=300,  # Mid-epoch save frequency set to 300 batches
            checkpoint_dir=checkpoint_dir,
            best_val_loss=best_val_loss,
            early_stop_counter=early_stop_counter,
            scheduler_state_dict=scheduler.state_dict(),
        )
        print(
            f"  -> Train Loss: {train_loss:.4f} | Frac Acc: {train_acc_frac*100.0:.2f}% | Region Acc: {train_acc_region*100.0:.2f}%"
        )

        # Validate
        val_loss, val_acc_frac, val_acc_region, reports = validate(
            model=model,
            loader=val_loader,
            criterion_frac=criterion_frac,
            criterion_region=criterion_region,
            device=device,
        )
        print(
            f"  -> Val Loss:   {val_loss:.4f} | Frac Acc: {val_acc_frac*100.0:.2f}% | Region Acc: {val_acc_region*100.0:.2f}%"
        )

        # Step LR scheduler (CosineAnnealingLR steps on epoch level)
        scheduler.step()

        # Get current learning rate
        current_lr = optimizer.param_groups[0]["lr"]

        # Append to history
        epoch_history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc_frac": train_acc_frac * 100.0,
                "train_acc_region": train_acc_region * 100.0,
                "val_loss": val_loss,
                "val_acc_frac": val_acc_frac * 100.0,
                "val_acc_region": val_acc_region * 100.0,
                "lr": current_lr,
            }
        )

        # Save best model checkpoint & manage early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_fracture_acc = val_acc_frac * 100.0
            best_bone_acc = val_acc_region * 100.0
            early_stop_counter = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_loss": val_loss,
                    "val_acc_frac": val_acc_frac,
                    "val_acc_region": val_acc_region,
                },
                best_checkpoint_path,
            )
            print(
                f"  *** Val Loss improved! Saved best model to: {best_checkpoint_path.name} ***"
            )
        else:
            early_stop_counter += 1
            print(
                f"  Early stopping counter: {early_stop_counter}/{early_stop_patience}"
            )

        # Save latest checkpoint at epoch end
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": val_loss,
                "best_val_loss": best_val_loss,
                "early_stop_counter": early_stop_counter,
                "best_epoch": best_epoch,
                "best_fracture_acc": best_fracture_acc,
                "best_bone_acc": best_bone_acc,
                "epoch_history": epoch_history,
            },
            latest_checkpoint_path,
        )
        print(f"  Saved latest model state to: {latest_checkpoint_path.name}")

        if early_stop_counter >= early_stop_patience:
            print(
                f"\n[!] Early stopping triggered. Validation loss has not improved for {early_stop_patience} epochs."
            )
            break

    # 1. Print the per-epoch history table
    if len(epoch_history) > 0:
        print("\n" + "=" * 88)
        print("STAGE 2 TRAINING HISTORY PER EPOCH")
        print("=" * 88)
        print(
            f"{'Epoch':<6} | {'Train Loss':<10} | {'Frac Acc':<10} | {'Region Acc':<10} | {'Val Loss':<10} | {'Val Frac':<10} | {'Val Region':<10} | {'LR':<8}"
        )
        print("-" * 88)
        for h in epoch_history:
            print(
                f"{h['epoch']:<6} | {h['train_loss']:<10.4f} | {h['train_acc_frac']:<10.2f}% | {h['train_acc_region']:<10.2f}% | {h['val_loss']:<10.4f} | {h['val_acc_frac']:<10.2f}% | {h['val_acc_region']:<10.2f}% | {h['lr']:<8.2e}"
            )
        print("=" * 88)

    # Calculate total training time
    total_training_time = time.time() - start_time
    hours = int(total_training_time // 3600)
    minutes = int((total_training_time % 3600) // 60)
    seconds = int(total_training_time % 60)
    training_time_str = (
        f"{hours}h {minutes}m {seconds}s" if hours > 0 else f"{minutes}m {seconds}s"
    )

    # 2. Run Final Evaluation using the best model saved
    if best_checkpoint_path.exists():
        print(f"\n[*] Loading best model for final evaluation: {best_checkpoint_path}")
        checkpoint = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        model.eval()
        all_severity_preds = []
        all_severity_labels = []
        all_bone_preds = []
        all_bone_labels = []

        with torch.no_grad():
            for batch in val_loader:
                images = batch[0].to(device)
                severity_labels = batch[1].to(device)
                bone_labels = batch[2].to(device)

                severity_logits, bone_logits = model(images)

                severity_preds = torch.argmax(severity_logits, dim=1).cpu().numpy()
                bone_preds = torch.argmax(bone_logits, dim=1).cpu().numpy()

                all_severity_preds.extend(severity_preds)
                all_severity_labels.extend(severity_labels.cpu().numpy())
                all_bone_preds.extend(bone_preds)
                all_bone_labels.extend(bone_labels.cpu().numpy())

        severity_classes = FRACTURE_CLASSES
        bone_classes = REGION_CLASSES

        print("\n" + "=" * 60)
        print("STAGE 2 FINAL EVALUATION REPORT")
        print("=" * 60)

        print("\n--- FRACTURE DETECTION HEAD ---")
        print(
            classification_report(
                all_severity_labels,
                all_severity_preds,
                target_names=severity_classes,
                zero_division=0,
            )
        )

        print("\n--- BODY REGION HEAD ---")
        print(
            classification_report(
                all_bone_labels,
                all_bone_preds,
                target_names=bone_classes,
                zero_division=0,
            )
        )

        print("\n--- CONFUSION MATRIX (Fracture Detection) ---")
        print(confusion_matrix(all_severity_labels, all_severity_preds))

        print("\n--- CONFUSION MATRIX (Body Region) ---")
        print(confusion_matrix(all_bone_labels, all_bone_preds))

        print("\n--- TRAINING SUMMARY ---")
        print(f"Best Fracture Detection Accuracy : {best_fracture_acc:.2f}%")
        print(f"Best Body Region Accuracy        : {best_bone_acc:.2f}%")
        print(f"Best Epoch                       : {best_epoch}")
        print(f"Best Val Loss                    : {best_val_loss:.4f}")
        print(f"Checkpoint saved to              : {best_checkpoint_path}")
        print(f"Total training time              : {training_time_str}")
        print("=" * 60)
    else:
        print(
            f"[!] Warning: Best checkpoint not found at {best_checkpoint_path} for final evaluation."
        )


if __name__ == "__main__":
    main()
