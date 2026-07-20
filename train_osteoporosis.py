import os
import cv2
import json
import time
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)
import mlflow

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BASE_DIR = "d:/X-ray ML Model"
DATA_DIR = os.path.join(BASE_DIR, "Mediscan", "osteoporosis")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Training Settings
BATCH_SIZE = 32
NUM_WORKERS = 0
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
EPOCHS = 30
PATIENCE_LR = 3
PATIENCE_EARLY_STOP = 5
FACTOR_LR = 0.5

# Class Mapping
CLASSES = {"normal": 0, "osteoporosis": 1}
CLASS_NAMES = {0: "Normal", 1: "Osteoporosis"}

# Logging Setup
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
    def __init__(self, p=0.3, mean=0.0, std=0.1):
        self.p = p
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        if torch.rand(1).item() < self.p:
            noise = torch.randn(tensor.size()) * self.std + self.mean
            return tensor + noise
        return tensor

    def __repr__(self):
        return self.__class__.__name__ + f"(p={self.p}, mean={self.mean}, std={self.std})"


class OsteoporosisDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.filepaths = []
        self.labels = []

        logger.info(f"Scanning dataset in {data_dir}...")
        for class_name, class_idx in CLASSES.items():
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.exists(class_dir):
                logger.warning(f"Directory not found: {class_dir}")
                continue

            for file in os.listdir(class_dir):
                if file.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.filepaths.append(os.path.join(class_dir, file))
                    self.labels.append(class_idx)

        self.num_normal = self.labels.count(0)
        self.num_osteo = self.labels.count(1)
        logger.info(f"Found {len(self.filepaths)} images: {self.num_normal} Normal, {self.num_osteo} Osteoporosis")

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img_path = self.filepaths[idx]
        label = self.labels[idx]

        # Load grayscale
        img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            raise ValueError(f"Failed to load image: {img_path}")

        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img_gray)

        # Convert to 3-channel RGB
        img_rgb = cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2RGB)

        # Convert to PIL Image for torchvision transforms
        from PIL import Image

        img_pil = Image.fromarray(img_rgb)

        if self.transform:
            img_tensor = self.transform(img_pil)
        else:
            img_tensor = transforms.ToTensor()(img_pil)

        return img_tensor, torch.tensor(label, dtype=torch.float32)


# Transforms
train_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        GaussianNoise(p=0.3, std=0.05),
    ]
)

val_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# ==============================================================================
# MODEL DEFINITION
# ==============================================================================
def create_model():
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

    # Freeze all layers initially
    for param in model.parameters():
        param.requires_grad = False

    # Replace final FC layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 1),  # 1 output for BCEWithLogitsLoss
    )

    return model


# ==============================================================================
# TRAINING LOOP
# ==============================================================================
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Prepare Data
    dataset = OsteoporosisDataset(DATA_DIR, transform=None)
    if len(dataset) == 0:
        logger.error(
            "Dataset is empty. Please ensure the Kaggle dataset is downloaded and extracted to d:/X-ray ML Model/Mediscan/osteoporosis/"
        )
        return

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Apply proper transforms
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # Initialize Model
    model = create_model().to(device)

    # Calculate pos_weight for class imbalance
    num_normal = dataset.num_normal
    num_osteo = dataset.num_osteo
    pos_weight_val = num_normal / num_osteo if num_osteo > 0 else 1.0
    pos_weight = torch.tensor([pos_weight_val]).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=PATIENCE_LR, factor=FACTOR_LR)

    best_val_acc = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    start_time = time.time()

    # MLflow Setup
    mlflow.set_experiment("osteoporosis_screening")

    with mlflow.start_run(run_name="osteoporosis_v1"):
        mlflow.log_params(
            {
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "epochs": EPOCHS,
                "pos_weight": pos_weight_val,
                "optimizer": "Adam",
                "scheduler": "ReduceLROnPlateau",
            }
        )

        for epoch in range(1, EPOCHS + 1):
            # Unfreeze layers after epoch 5
            if epoch == 6:
                logger.info("Epoch 6 reached: Unfreezing layer3 and layer4")
                for name, param in model.named_parameters():
                    if "layer3" in name or "layer4" in name or "fc" in name:
                        param.requires_grad = True

            # Train phase
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

            epoch_train_loss = train_loss / len(train_loader.dataset)
            epoch_train_acc = accuracy_score(train_targets, train_preds)

            # Validation phase
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

            epoch_val_loss = val_loss / len(val_loader.dataset)
            epoch_val_acc = accuracy_score(val_targets, val_preds)

            # Step scheduler
            scheduler.step(epoch_val_loss)

            # Logging
            logger.info(
                f"Epoch {epoch}/{EPOCHS} - "
                f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | "
                f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}"
            )

            mlflow.log_metrics(
                {
                    "train_loss": epoch_train_loss,
                    "train_acc": epoch_train_acc,
                    "val_loss": epoch_val_loss,
                    "val_acc": epoch_val_acc,
                },
                step=epoch,
            )

            history["train_loss"].append(epoch_train_loss)
            history["val_loss"].append(epoch_val_loss)
            history["train_acc"].append(epoch_train_acc)
            history["val_acc"].append(epoch_val_acc)

            # Early Stopping and Checkpointing
            if epoch_val_acc > best_val_acc:
                best_val_acc = epoch_val_acc
                best_epoch = epoch
                epochs_no_improve = 0
                best_model_path = os.path.join(CHECKPOINT_DIR, "osteoporosis_best.pth")
                torch.save(model.state_dict(), best_model_path)
                logger.info(f"--> Saved new best model to {best_model_path}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= PATIENCE_EARLY_STOP:
                    logger.info(f"Early stopping triggered after {epoch} epochs.")
                    break

        # Final Evaluation
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time/60:.2f} minutes.")
        logger.info(f"Best Val Accuracy: {best_val_acc:.4f} at Epoch {best_epoch}")

        # Load best model for final evaluation
        model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "osteoporosis_best.pth")))
        model.eval()

        final_preds, final_targets = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
                outputs = model(inputs)
                preds = torch.sigmoid(outputs) >= 0.5
                final_preds.extend(preds.cpu().numpy())
                final_targets.extend(labels.cpu().numpy())

        cm = confusion_matrix(final_targets, final_preds)
        report = classification_report(final_targets, final_preds, target_names=["Normal", "Osteoporosis"])

        # Sensitivity and Specificity
        # cm structure: [[TN, FP], [FN, TP]]
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Recall on Class 1 (Osteoporosis)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # Recall on Class 0 (Normal)

        logger.info("\n=== FINAL EVALUATION ===")
        logger.info(f"\nConfusion Matrix:\n{cm}")
        logger.info(f"\nClassification Report:\n{report}")
        logger.info(f"Sensitivity (Osteoporosis Recall): {sensitivity:.4f}")
        logger.info(f"Specificity (Normal Recall): {specificity:.4f}")

        mlflow.log_metrics(
            {
                "best_epoch": best_epoch,
                "best_val_acc": best_val_acc,
                "sensitivity": sensitivity,
                "specificity": specificity,
            }
        )

        # Save History
        history_path = os.path.join(CHECKPOINT_DIR, "osteoporosis_training_history.json")
        with open(history_path, "w") as f:
            json.dump(history, f, indent=4)
        logger.info(f"Saved training history to {history_path}")


if __name__ == "__main__":
    train_model()
