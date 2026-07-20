import os
import sys
import json
import time
import logging
from collections import Counter

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import mlflow

# ==============================================================================
# CONFIGURATION & LOGGING
# ==============================================================================
BASE_DIR = "d:/X-ray ML Model/Mediscan/brain_tumor"
CHECKPOINT_DIR = "d:/X-ray ML Model/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "brain_tumor_best.pth")
HISTORY_JSON_PATH = os.path.join(CHECKPOINT_DIR, "brain_tumor_training_history.json")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("mediscan.brain_tumor.train")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Note: Lowered batch size to 8 to ensure stability on the RTX 4050 6GB VRAM when all layers unfreeze.
BATCH_SIZE = 8
NUM_WORKERS = 0
EPOCHS = 40
EARLY_STOPPING_PATIENCE = 7
LR = 1e-4
WEIGHT_DECAY = 1e-5

CLASS_NAMES = ["notumor", "glioma", "meningioma", "pituitary"]
CLASS_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}


# ==============================================================================
# DATASET DISCOVERY
# ==============================================================================
def discover_dataset():
    if not os.path.exists(BASE_DIR):
        logger.error(f"Dataset folder does not exist: {BASE_DIR}")
        logger.error("Please download and extract the dataset first.")
        sys.exit(1)

    logger.info("=== DATASET DISCOVERY ===")
    total_images = 0
    expected_structure_found = True

    for root, dirs, files in os.walk(BASE_DIR):
        level = root.replace(BASE_DIR, "").count(os.sep)
        indent = "  " * level
        folder_name = os.path.basename(root)
        if not folder_name:
            folder_name = "brain_tumor (root)"

        img_files = [f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        print(f"{indent}{folder_name}/  [{len(img_files)} images]")

        if level > 0 and len(img_files) > 0:
            total_images += len(img_files)

    print(f"\nTotal image count across all folders: {total_images}")

    # Check if expected folders exist
    if not os.path.exists(os.path.join(BASE_DIR, "Training")) or not os.path.exists(
        os.path.join(BASE_DIR, "Testing")
    ):
        expected_structure_found = False

    if not expected_structure_found:
        logger.error("⚠️ Dataset structure differs from expected.")
        logger.error(
            "Expected 'Training' and 'Testing' folders with class subdirectories."
        )
        logger.error("Printed folder tree above. Adjust the script accordingly.")
        sys.exit(1)

    if total_images == 0:
        logger.error("Dataset folder is empty or contains no images. Exiting.")
        sys.exit(1)


# ==============================================================================
# PREPROCESSING & AUGMENTATION
# ==============================================================================
train_transforms = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=0, scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

val_transforms = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def apply_noise(tensor, p=0.2):
    if torch.rand(1).item() < p:
        noise = torch.randn(tensor.size()) * 0.05
        return tensor + noise
    return tensor


# Create CLAHE object globally to prevent OpenCV memory leaks in DataLoader
clahe_obj = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))


def preprocess_image(img_path):
    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        img_gray = np.zeros((224, 224), dtype=np.uint8)

    # MRI specific CLAHE
    img_clahe = clahe_obj.apply(img_gray)
    img_rgb = cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2RGB)
    return img_rgb


class BrainTumorDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, is_train=False):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.is_train = is_train

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        img_rgb = preprocess_image(img_path)

        if self.transform:
            tensor = self.transform(img_rgb)

        if self.is_train:
            tensor = apply_noise(tensor)

        return tensor, label


def load_data(split_folder):
    paths = []
    labels = []
    split_dir = os.path.join(BASE_DIR, split_folder)
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(split_dir, class_name)
        if os.path.exists(class_dir):
            for img in os.listdir(class_dir):
                if img.lower().endswith((".png", ".jpg", ".jpeg")):
                    paths.append(os.path.join(class_dir, img))
                    labels.append(CLASS_MAP[class_name])
    return paths, labels


# ==============================================================================
# MODEL & TRAINING UTILS
# ==============================================================================
def build_model():
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)

    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512), nn.ReLU(), nn.Dropout(0.4), nn.Linear(512, 4)
    )

    # Freeze all except layer3, layer4, fc
    for name, param in model.named_parameters():
        if not any(n in name for n in ["layer3", "layer4", "fc"]):
            param.requires_grad = False

    return model.to(DEVICE)


def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True
    logger.info("Unfrozen all layers for fine-tuning.")


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss, total, correct = 0.0, 0, 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data).item()
        total += labels.size(0)

    return running_loss / total, correct / total


def evaluate(model, loader, criterion):
    model.eval()
    running_loss, total, correct = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data).item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2, 3])
    cm_diag = cm.diagonal()
    cm_sums = cm.sum(axis=1)

    recalls = {
        CLASS_NAMES[i]: (cm_diag[i] / cm_sums[i] if cm_sums[i] > 0 else 0)
        for i in range(4)
    }

    return epoch_loss, epoch_acc, recalls, all_preds, all_labels


# ==============================================================================
# MAIN WORKFLOW
# ==============================================================================
def main():
    discover_dataset()

    X_train_full, y_train_full = load_data("Training")
    X_test, y_test = load_data("Testing")

    # 90/10 Split for Train/Val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.10,
        stratify=y_train_full,
        random_state=42,
    )

    train_counts = Counter(y_train)
    val_counts = Counter(y_val)
    test_counts = Counter(y_test)

    print("\nSplit | No Tumor | Glioma | Meningioma | Pituitary | Total")
    print(
        f"train | {train_counts[0]:8} | {train_counts[1]:6} | {train_counts[2]:10} | {train_counts[3]:9} | {len(y_train)}"
    )
    print(
        f"val   | {val_counts[0]:8} | {val_counts[1]:6} | {val_counts[2]:10} | {val_counts[3]:9} | {len(y_val)}"
    )
    print(
        f"test  | {test_counts[0]:8} | {test_counts[1]:6} | {test_counts[2]:10} | {test_counts[3]:9} | {len(y_test)}\n"
    )

    # Manual class weights to heavily penalize glioma misclassification (Change 1)
    class_weights = [0.8, 2.5, 1.0, 0.9]
    weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

    # Glioma Oversampling via WeightedRandomSampler (Change 2)
    from torch.utils.data import WeightedRandomSampler

    sample_weights = [2.5 if label == 1 else 1.0 for label in y_train]
    sampler = WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True
    )

    train_ds = BrainTumorDataset(
        X_train, y_train, transform=train_transforms, is_train=True
    )
    val_ds = BrainTumorDataset(X_val, y_val, transform=val_transforms)
    test_ds = BrainTumorDataset(X_test, y_test, transform=val_transforms)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    # Model & Optim
    model = build_model()
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5
    )

    mlflow.set_experiment("brain_tumor_detection")
    best_val_loss = float("in")
    epochs_no_improve = 0
    best_epoch = 0
    best_acc = 0.0
    history = []

    start_time = time.time()

    with mlflow.start_run(run_name="brain_tumor_v1"):
        mlflow.log_params(
            {
                "model": "ResNet50",
                "dataset": "Brain_Tumor_MRI",
                "num_classes": 4,
                "batch_size": BATCH_SIZE,
                "lr": LR,
            }
        )

        print(
            "\nEpoch | Train Loss | Train Acc | Val Loss | Val Acc | Glioma Recall | Meningioma Recall | Pituitary Recall | LR"
        )
        print("-" * 115)

        for epoch in range(1, EPOCHS + 1):
            if epoch == 6:
                unfreeze_all(model)
                optimizer = optim.Adam(
                    model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
                )
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min", patience=3, factor=0.5
                )

            current_lr = optimizer.param_groups[0]["lr"]

            t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer)
            v_loss, v_acc, recalls, _, _ = evaluate(model, val_loader, criterion)

            scheduler.step(v_loss)

            print(
                f"{epoch:5} | {t_loss:10.4f} | {t_acc:9.4f} | {v_loss:8.4f} | {v_acc:7.4f} | {recalls['glioma']:13.4f} | {recalls['meningioma']:17.4f} | {recalls['pituitary']:16.4f} | {current_lr:.2e}"
            )
            print(
                f"Glioma Recall: {recalls['glioma']*100:.2f}% — target above 85% (aggressive cancer)"
            )
            print(f"Meningioma Recall: {recalls['meningioma']*100:.2f}%")
            print(f"Pituitary Recall: {recalls['pituitary']*100:.2f}%")

            if epoch >= 8 and v_acc < 0.70:
                logger.warning("Validation accuracy has not exceeded 70% by epoch 8.")
            if epoch >= 10 and recalls["glioma"] < 0.75:
                logger.warning("Glioma recall dropped below 75%.")

            history.append(
                {
                    "epoch": epoch,
                    "train_loss": t_loss,
                    "val_loss": v_loss,
                    "train_acc": t_acc,
                    "val_acc": v_acc,
                    "recalls": recalls,
                }
            )

            mlflow.log_metrics(
                {
                    "train_loss": t_loss,
                    "val_loss": v_loss,
                    "train_acc": t_acc,
                    "val_acc": v_acc,
                    "glioma_recall": recalls["glioma"],
                    "meningioma_recall": recalls["meningioma"],
                    "pituitary_recall": recalls["pituitary"],
                    "notumor_recall": recalls["notumor"],
                },
                step=epoch,
            )

            if v_loss < best_val_loss:
                best_val_loss = v_loss
                best_epoch = epoch
                best_acc = v_acc
                epochs_no_improve = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "state_dict": model.state_dict(),
                        "val_loss": v_loss,
                        "val_acc": v_acc,
                        "per_class_recall": recalls,
                    },
                    BEST_MODEL_PATH,
                )
            else:
                epochs_no_improve += 1

            with open(HISTORY_JSON_PATH, "w") as f:
                json.dump(history, f, indent=4)

            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    total_time = time.time() - start_time

    # TEST EVALUATION
    logger.info("=== FINAL EVALUATION ON TEST SET ===")
    checkpoint = torch.load(BEST_MODEL_PATH)
    model.load_state_dict(checkpoint["state_dict"])

    _, test_acc, test_recalls, test_preds, test_labels = evaluate(
        model, test_loader, criterion
    )

    print("\nConfusion Matrix:")
    print(confusion_matrix(test_labels, test_preds, labels=[0, 1, 2, 3]))
    print("\nClassification Report:")
    print(
        classification_report(
            test_labels, test_preds, target_names=CLASS_NAMES, zero_division=0
        )
    )

    print("\nPer-class recall breakdown:")
    print(f"No Tumor recall:   {test_recalls['notumor']*100:.2f}%")
    print(f"Glioma recall:     {test_recalls['glioma']*100:.2f}% <-- HIGHLIGHT THIS")
    print(f"Meningioma recall: {test_recalls['meningioma']*100:.2f}%")
    print(f"Pituitary recall:  {test_recalls['pituitary']*100:.2f}%")
    print(f"\nOverall accuracy: {test_acc*100:.2f}%")

    if test_recalls["glioma"] < 0.85:
        print("\n⚠️ Glioma recall below clinical threshold.")
        print("Gliomas are aggressive cancers — missing them is dangerous.")
        print(
            "Consider: (1) increasing glioma class weight, (2) lowering classification threshold for glioma, (3) adding more augmentation."
        )

    for cls in CLASS_NAMES:
        if test_recalls[cls] < 0.70:
            print(
                f"⚠️ Warning: {cls} recall is below 70% ({test_recalls[cls]*100:.2f}%)"
            )

    if all(test_recalls[cls] >= 0.95 for cls in CLASS_NAMES):
        print(
            "\nNote: Very high recall across all classes. Consider testing on external MRI data to verify generalization."
        )

    print(
        f"\nTraining summary: best epoch={best_epoch}, best val acc={best_acc:.4f}, total training time={total_time/60:.2f} mins"
    )


if __name__ == "__main__":
    main()
