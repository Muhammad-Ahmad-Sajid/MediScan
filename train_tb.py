import os
import cv2
import json
import time
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    recall_score,
)
import mlflow

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BASE_DIR = "d:/X-ray ML Model"
DATA_DIR = os.path.join(BASE_DIR, "Mediscan", "tb", "TB_Chest_X-ray_Database")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

BATCH_SIZE = 32
NUM_WORKERS = 0
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
EPOCHS = 30
PATIENCE_LR = 3
PATIENCE_EARLY_STOP = 5
FACTOR_LR = 0.5

# Class Mapping (Binary: 0=Normal, 1=Tuberculosis)
CLASSES = {"Normal": 0, "Tuberculosis": 1}
CLASS_NAMES = {0: "Normal", 1: "Tuberculosis"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# ==============================================================================
# DATASET & PREPROCESSING
# ==============================================================================
class GaussianNoise(object):
    def __init__(self, p=0.2, mean=0.0, std=0.1):
        self.p = p
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        if torch.rand(1).item() < self.p:
            noise = torch.randn(tensor.size()) * self.std + self.mean
            return tensor + noise
        return tensor


train_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(degrees=5),
        transforms.ColorJitter(brightness=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        GaussianNoise(p=0.2, std=0.05),
    ]
)

val_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class TBDataset(Dataset):
    def __init__(self, filepaths, labels, transform=None):
        self.filepaths = filepaths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img_path = self.filepaths[idx]
        label = self.labels[idx]

        # Load as grayscale
        img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            raise ValueError(f"Failed to load image: {img_path}")

        # Apply CLAHE (clipLimit=3.0, tileGridSize=8x8 for Chest X-ray)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img_gray)

        # Convert to 3-channel RGB
        img_rgb = cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2RGB)

        from PIL import Image

        img_pil = Image.fromarray(img_rgb)

        if self.transform:
            img_tensor = self.transform(img_pil)
        else:
            img_tensor = transforms.ToTensor()(img_pil)

        return img_tensor, torch.tensor(label, dtype=torch.float32)


# ==============================================================================
# MODEL DEFINITION
# ==============================================================================
def create_model():
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    # Note: Linear(512, 1) is used instead of Linear(512, 2) because BCEWithLogitsLoss
    # intrinsically requires a single output node for binary classification to work with pos_weight.
    model.fc = nn.Sequential(nn.Linear(num_ftrs, 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, 1))
    return model


# ==============================================================================
# TRAINING LOOP
# ==============================================================================
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Check directory structure
    if (
        not os.path.exists(DATA_DIR)
        or not os.path.exists(os.path.join(DATA_DIR, "Normal"))
        or not os.path.exists(os.path.join(DATA_DIR, "Tuberculosis"))
    ):
        logger.error(f"Expected dataset structure not found at {DATA_DIR}")
        logger.error(f"Actual contents of {os.path.dirname(DATA_DIR)}:")
        parent_dir = os.path.dirname(DATA_DIR)
        if os.path.exists(parent_dir):
            for item in os.listdir(parent_dir):
                logger.error(f"  - {item}")
        else:
            logger.error("  Parent directory does not exist either.")
        logger.error("Please download and extract the dataset properly.")
        return

    all_filepaths = []
    all_labels = []

    for class_name, class_idx in CLASSES.items():
        class_dir = os.path.join(DATA_DIR, class_name)
        for file in os.listdir(class_dir):
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                all_filepaths.append(os.path.join(class_dir, file))
                all_labels.append(class_idx)

    # 80/10/10 Stratified Split
    X_temp, X_test, y_temp, y_test = train_test_split(
        all_filepaths, all_labels, test_size=0.10, stratify=all_labels, random_state=42
    )
    # Remaining 90% goes into 80/10 (which is 88.88% train and 11.11% val of the temp split)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=(0.10 / 0.90), stratify=y_temp, random_state=42
    )

    num_normal_train = y_train.count(0)
    num_tb_train = y_train.count(1)

    logger.info("=== Dataset Split Counts ===")
    logger.info(f"Total images: {len(all_labels)}")
    logger.info(f"Train : {len(y_train)} (Normal: {num_normal_train}, TB: {num_tb_train})")
    logger.info(f"Val   : {len(y_val)} (Normal: {y_val.count(0)}, TB: {y_val.count(1)})")
    logger.info(f"Test  : {len(y_test)} (Normal: {y_test.count(0)}, TB: {y_test.count(1)})")

    # WeightedRandomSampler for class imbalance
    class_sample_counts = [num_normal_train, num_tb_train]
    weights = [1.0 / count if count > 0 else 0 for count in class_sample_counts]
    sample_weights = [weights[label] for label in y_train]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_dataset = TBDataset(X_train, y_train, transform=train_transform)
    val_dataset = TBDataset(X_val, y_val, transform=val_transform)
    test_dataset = TBDataset(X_test, y_test, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = create_model().to(device)

    # Class Weights for Loss
    pos_weight_val = num_normal_train / num_tb_train if num_tb_train > 0 else 1.0
    pos_weight = torch.tensor([pos_weight_val]).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=PATIENCE_LR, factor=FACTOR_LR)

    best_val_acc = 0.0
    best_epoch = 0
    float("inf")
    best_sensitivity = 0.0
    epochs_no_improve = 0
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "tb_sensitivity": [],
    }

    start_time = time.time()

    mlflow.set_experiment("tb_screening")
    with mlflow.start_run(run_name="tb_v1"):
        mlflow.log_params(
            {
                "model": "ResNet50",
                "dataset": "TB_Chest_Xray",
                "pos_weight": pos_weight_val,
                "batch_size": BATCH_SIZE,
                "lr": LEARNING_RATE,
            }
        )

        logger.info("\nEpoch | Train Loss | Train Acc | Val Loss | Val Acc | Sensitivity | LR")
        logger.info("-" * 75)

        for epoch in range(1, EPOCHS + 1):
            if epoch == 6:
                logger.info("Epoch 6 reached: Unfreezing layer3, layer4, and fc")
                for name, param in model.named_parameters():
                    if "layer3" in name or "layer4" in name or "fc" in name:
                        param.requires_grad = True

            # Train
            model.train()
            train_loss = 0.0
            train_preds, train_targets = [], []

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * inputs.size(0)
                preds = torch.sigmoid(outputs) >= 0.5
                train_preds.extend(preds.cpu().numpy())
                train_targets.extend(labels.cpu().numpy())

            epoch_train_loss = train_loss / len(train_dataset)
            epoch_train_acc = accuracy_score(train_targets, train_preds)

            # Validate
            model.eval()
            val_loss = 0.0
            val_preds, val_targets = [], []

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * inputs.size(0)
                    preds = torch.sigmoid(outputs) >= 0.5
                    val_preds.extend(preds.cpu().numpy())
                    val_targets.extend(labels.cpu().numpy())

            epoch_val_loss = val_loss / len(val_dataset)
            epoch_val_acc = accuracy_score(val_targets, val_preds)
            epoch_tb_sensitivity = recall_score(val_targets, val_preds, pos_label=1, zero_division=0)

            current_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(epoch_val_loss)

            # Table row logging
            logger.info(
                f"{epoch:5d} | {epoch_train_loss:10.4f} | {epoch_train_acc:9.4f} | {epoch_val_loss:8.4f} | {epoch_val_acc:7.4f} | {epoch_tb_sensitivity:11.4f} | {current_lr:.1e}"
            )
            logger.info(f"TB Sensitivity: {epoch_tb_sensitivity*100:.2f}% — target above 85%")

            mlflow.log_metrics(
                {
                    "train_loss": epoch_train_loss,
                    "train_acc": epoch_train_acc,
                    "val_loss": epoch_val_loss,
                    "val_acc": epoch_val_acc,
                    "tb_sensitivity": epoch_tb_sensitivity,
                },
                step=epoch,
            )

            history["train_loss"].append(epoch_train_loss)
            history["val_loss"].append(epoch_val_loss)
            history["train_acc"].append(epoch_train_acc)
            history["val_acc"].append(epoch_val_acc)
            history["tb_sensitivity"].append(epoch_tb_sensitivity)

            # Warnings Checks
            if epoch == 8 and epoch_val_acc <= 0.70:
                logger.warning("WARNING: Validation accuracy has not exceeded 70% by epoch 8.")
            if epoch > 10 and epoch_tb_sensitivity < 0.70:
                logger.warning(f"WARNING: TB Sensitivity dropped below 70% at epoch {epoch}!")

            # Checkpointing
            if epoch_val_acc > best_val_acc:
                best_val_acc = epoch_val_acc
                best_epoch = epoch
                best_sensitivity = epoch_tb_sensitivity
                epochs_no_improve = 0

                best_model_path = os.path.join(CHECKPOINT_DIR, "tb_best.pth")
                save_dict = {
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": epoch_val_loss,
                    "val_acc": epoch_val_acc,
                    "sensitivity": epoch_tb_sensitivity,
                }
                torch.save(save_dict, best_model_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= PATIENCE_EARLY_STOP:
                    logger.info(f"Early stopping triggered after {epoch} epochs.")
                    break

        total_time = time.time() - start_time
        logger.info("\n=== TRAINING SUMMARY ===")
        logger.info(f"Best Epoch: {best_epoch}")
        logger.info(f"Best Val Acc: {best_val_acc:.4f}")
        logger.info(f"Best Sensitivity: {best_sensitivity:.4f}")
        logger.info(f"Total Time: {total_time/60:.2f} minutes")

        # Load best model for TEST evaluation
        checkpoint = torch.load(os.path.join(CHECKPOINT_DIR, "tb_best.pth"))
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        test_preds, test_targets = [], []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
                outputs = model(inputs)
                preds = torch.sigmoid(outputs) >= 0.5
                test_preds.extend(preds.cpu().numpy())
                test_targets.extend(labels.cpu().numpy())

        cm = confusion_matrix(test_targets, test_preds)
        report = classification_report(test_targets, test_preds, target_names=["Normal", "Tuberculosis"])

        tn, fp, fn, tp = cm.ravel()
        test_sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # TB class
        test_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # Normal class

        logger.info("\n=== FINAL TEST EVALUATION ===")
        logger.info(f"\nConfusion Matrix:\n{cm}")
        logger.info(f"\nClassification Report:\n{report}")
        logger.info(f"*** TB Sensitivity (Recall): {test_sensitivity*100:.2f}% ***")
        logger.info(f"TB Specificity (Normal Recall): {test_specificity*100:.2f}%")

        if test_sensitivity < 0.85:
            logger.warning(
                "⚠️ TB sensitivity below clinical threshold. Consider lowering classification threshold or retraining."
            )

        history_path = os.path.join(CHECKPOINT_DIR, "tb_training_history.json")
        with open(history_path, "w") as f:
            json.dump(history, f, indent=4)
        logger.info(f"Saved training history to {history_path}")


if __name__ == "__main__":
    train_model()
