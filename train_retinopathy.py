print("STARTING SCRIPT", flush=True)
import os

os.environ["OMP_NUM_THREADS"] = "1"
import sys
import time
import json
import logging
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    cohen_kappa_score,
    classification_report,
    confusion_matrix,
    recall_score,
)
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import cv2

cv2.setNumThreads(0)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("mediscan.retinopathy.train")

BASE_DIR = "d:/X-ray ML Model/Mediscan/retinopathy"
CHECKPOINT_DIR = "d:/X-ray ML Model/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "retinopathy_best.pth")
HISTORY_PATH = os.path.join(CHECKPOINT_DIR, "retinopathy_training_history.json")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_dataset_discovery():
    logger.info("=== DATASET DISCOVERY ===")
    img_folders = []
    csv_file = None

    for root, dirs, files in os.walk(BASE_DIR):
        level = root.replace(BASE_DIR, "").count(os.sep)
        indent = "  " * level
        img_count = len(
            [f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        )
        logger.info(f"{indent}{os.path.basename(root)}/  [{img_count} images]")

        for f in files:
            if f.lower().endswith(".csv"):
                csv_file = os.path.join(root, f)
                logger.info(f"\nFound: {csv_file}")
                df = pd.read_csv(csv_file)
                logger.info(f"Columns: {df.columns.tolist()}")
                logger.info(f"\nHead:\n{df.head(10)}")
                for col in df.columns:
                    if any(
                        kw in col.lower()
                        for kw in ["label", "level", "diagnosis", "grade"]
                    ):
                        logger.info(
                            f"\n{col} distribution:\n{df[col].value_counts().sort_index()}"
                        )

    return csv_file


class RetinopathyDataset(Dataset):
    def __init__(self, df, base_dir, is_train=False):
        self.df = df.reset_index(drop=True)
        self.base_dir = base_dir
        self.is_train = is_train

        # Determine image id column (either 'id_code' or 'image')
        self.id_col = "id_code" if "id_code" in self.df.columns else self.df.columns[0]
        self.label_col = (
            "diagnosis" if "diagnosis" in self.df.columns else self.df.columns[1]
        )

        # Build an absolute path dictionary for lightning fast lookups
        self.image_paths = {}
        for root, _, files in os.walk(self.base_dir):
            for f in files:
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    name_no_ext = os.path.splitext(f)[0]
                    self.image_paths[name_no_ext] = os.path.join(root, f)
                    self.image_paths[f] = os.path.join(root, f)

        # Torchvision transforms (applied after OpenCV preprocessing)
        if self.is_train:
            self.transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.RandomRotation(
                        180
                    ),  # 360 is effectively handled by 180 (±180)
                    transforms.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = str(self.df.iloc[idx][self.id_col])
        name_no_ext = os.path.splitext(img_id)[0]

        if name_no_ext not in self.image_paths and img_id not in self.image_paths:
            raise ValueError(
                f"Failed to find image: {img_id} anywhere in {self.base_dir}"
            )

        img_path = self.image_paths.get(img_id, self.image_paths.get(name_no_ext))

        label = int(self.df.iloc[idx][self.label_col])

        # Step 1: Load COLOR
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Step 2: Green channel enhancement
        enhanced = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), 30), -4, 128)

        # Step 3: Resize (just to be safe)
        enhanced = cv2.resize(enhanced, (224, 224))

        # Apply torchvision transforms (includes normalization & noise)
        tensor_img = self.transform(enhanced)

        # Add random Gaussian noise for training
        if self.is_train and np.random.rand() < 0.2:
            noise = torch.randn_like(tensor_img) * 0.05
            tensor_img = tensor_img + noise

        return tensor_img, label


def get_dataloaders(csv_file):
    df = pd.read_csv(csv_file)
    label_col = "diagnosis" if "diagnosis" in df.columns else df.columns[1]

    # 80/10/10 split
    train_df, temp_df = train_test_split(
        df, test_size=0.2, stratify=df[label_col], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df[label_col], random_state=42
    )

    # Print distribution
    logger.info("\nClass Distribution:")
    logger.info(f"Split | Grade 0 | Grade 1 | Grade 2 | Grade 3 | Grade 4 | Total")
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        counts = split_df[label_col].value_counts()
        c0, c1, c2, c3, c4 = (
            counts.get(0, 0),
            counts.get(1, 0),
            counts.get(2, 0),
            counts.get(3, 0),
            counts.get(4, 0),
        )
        logger.info(
            f"{name:5} | {c0:7} | {c1:7} | {c2:7} | {c3:7} | {c4:7} | {len(split_df)}"
        )

    img_dir = os.path.join(os.path.dirname(csv_file), "train_images")
    if not os.path.exists(img_dir):
        img_dir = os.path.dirname(csv_file)  # fallback flat struct

    train_ds = RetinopathyDataset(train_df, img_dir, is_train=True)
    val_ds = RetinopathyDataset(val_df, img_dir, is_train=False)
    test_ds = RetinopathyDataset(test_df, img_dir, is_train=False)

    # Compute class weights
    class_counts = train_df[label_col].value_counts().sort_index().values
    total_samples = len(train_df)
    class_weights = total_samples / (5.0 * class_counts)
    class_weights_tensor = torch.FloatTensor(class_weights).to(DEVICE)
    logger.info(f"\nComputed Class Weights: {class_weights}")

    # WeightedRandomSampler
    sample_weights = [class_weights[label] for label in train_df[label_col].values]
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    train_loader = DataLoader(train_ds, batch_size=16, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader, class_weights_tensor, test_df


class RetinopathyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512), nn.ReLU(), nn.Dropout(0.4), nn.Linear(512, 5)
        )

    def forward(self, x):
        return self.backbone(x)


def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True


def train_model():
    csv_file = run_dataset_discovery()
    if not csv_file:
        logger.error("No CSV found! Exiting.")
        return

    train_loader, val_loader, test_loader, class_weights, test_df = get_dataloaders(
        csv_file
    )

    model = RetinopathyModel().to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,
        weight_decay=1e-5,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=3, factor=0.5
    )

    best_qwk = -1.0
    patience_counter = 0
    history = []
    start_epoch = 1

    if os.path.exists(CHECKPOINT_PATH):
        logger.info(f"Resuming training from {CHECKPOINT_PATH}...")
        try:
            checkpoint = torch.load(CHECKPOINT_PATH)
            model.load_state_dict(checkpoint["state_dict"])

            # If we are past epoch 5, we need to unfreeze all layers so the optimizer gets all params
            if checkpoint["epoch"] >= 5:
                unfreeze_all(model)
                optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="max", patience=3, factor=0.5
                )

            if "optimizer" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer"])
            if "scheduler" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler"])

            start_epoch = checkpoint["epoch"] + 1
            best_qwk = checkpoint.get("val_qwk", -1.0)

            # Load history if exists
            if os.path.exists(HISTORY_PATH):
                with open(HISTORY_PATH, "r") as f:
                    history = json.load(f)

            logger.info(
                f"Successfully resumed at Epoch {start_epoch} with Best QWK {best_qwk:.4f}"
            )
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")

    # Freeze all except layer3, layer4, fc (if starting fresh or before epoch 6)
    if start_epoch <= 5:
        for name, param in model.named_parameters():
            if not any(k in name for k in ["layer3", "layer4", "fc"]):
                param.requires_grad = False
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-4,
            weight_decay=1e-5,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=3, factor=0.5
        )

    # mlflow.set_experiment removed

    logger.info(
        f"\nEpoch | Train Loss | Train Acc | Val Loss | Val Acc | Val QWK | Grade4 Recall | LR"
    )
    logger.info("-" * 85)

    start_time = time.time()

    # MLFlow removed
    for epoch in range(start_epoch, 31):
        if epoch == 6:
            logger.info("Unfreezing all layers...")
            unfreeze_all(model)
            optimizer = optim.Adam(
                model.parameters(),
                lr=scheduler.optimizer.param_groups[0]["lr"],
                weight_decay=1e-5,
            )
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", patience=3, factor=0.5
            )

        # Train
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels).item()
            total += labels.size(0)

        train_loss = train_loss / total
        train_acc = correct / total

        # Val
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * imgs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += torch.sum(preds == labels).item()
                total += labels.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss = val_loss / total
        val_acc = correct / total
        qwk = cohen_kappa_score(all_labels, all_preds, weights="quadratic")

        # Recalls
        recalls = recall_score(
            all_labels, all_preds, average=None, labels=[0, 1, 2, 3, 4], zero_division=0
        )
        g4_recall = recalls[4] * 100.0 if len(recalls) > 4 else 0.0

        lr = optimizer.param_groups[0]["lr"]
        logger.info(
            f"{epoch:5} | {train_loss:10.4f} | {train_acc*100:8.1f}% | {val_loss:8.4f} | {val_acc*100:6.1f}% | {qwk:7.4f} | {g4_recall:12.1f}% | {lr:.2e}"
        )

        logger.info(f"QWK: {qwk:.4f} — target above 0.70")
        logger.info(
            f"Grade 4 (Proliferative) Recall: {g4_recall:.2f}% — target above 70%"
        )

        if epoch >= 10:
            if qwk < 0.50:
                logger.warning("WARNING: QWK below 0.50 after epoch 10")
            if g4_recall < 50.0:
                logger.warning("WARNING: Grade 4 recall below 50% after epoch 10")

        # Removed MLFlow to bypass Windows DLL deadlock

        history.append({"epoch": epoch, "qwk": qwk, "val_loss": val_loss})

        scheduler.step(qwk)  # Maximize QWK

        if qwk > best_qwk:
            best_qwk = qwk
            patience_counter = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_qwk": qwk,
                    "per_class_recall": recalls.tolist(),
                },
                CHECKPOINT_PATH,
            )
        else:
            patience_counter += 1

        if patience_counter >= 5:
            logger.info(f"Early stopping triggered at epoch {epoch}")
            break

    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f)

    total_time = (time.time() - start_time) / 60.0

    # Final Evaluation
    logger.info("\n" + "=" * 50)
    logger.info("FINAL EVALUATION ON TEST SET")
    logger.info("=" * 50)

    checkpoint = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            outputs = model(imgs.to(DEVICE))
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    qwk_test = cohen_kappa_score(all_labels, all_preds, weights="quadratic")
    conf_matrix = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, digits=4, zero_division=0)

    recalls_test = recall_score(
        all_labels, all_preds, average=None, labels=[0, 1, 2, 3, 4], zero_division=0
    )

    logger.info(f"\nClassification Report:\n{report}")
    logger.info(f"\nConfusion Matrix (5x5):\n{conf_matrix}")
    logger.info(f"\n>>> QUADRATIC WEIGHTED KAPPA: {qwk_test:.4f} <<<")

    logger.info("\nPer-class recall:")
    for i in range(5):
        if i == 4:
            logger.info(
                f"Grade 4 (Proliferative): {recalls_test[i]*100:.2f}% <- HIGHLIGHT"
            )
        else:
            names = ["No DR", "Mild", "Moderate", "Severe"]
            logger.info(f"Grade {i} ({names[i]}): {recalls_test[i]*100:.2f}%")

    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    logger.info(f"Overall Accuracy: {acc*100:.2f}%")

    # Combined G3+G4 recall
    g34_true = [1 if l in [3, 4] else 0 for l in all_labels]
    g34_pred = [1 if p in [3, 4] else 0 for p in all_preds]
    combined_recall = recall_score(g34_true, g34_pred, pos_label=1, zero_division=0)
    logger.info(
        f"Clinically Critical Accuracy (G3+G4 combined recall): {combined_recall*100:.2f}%"
    )

    # Ordinal Error Analysis
    diffs = np.abs(np.array(all_labels) - np.array(all_preds))
    avg_err = np.mean(diffs)
    off_by_1 = np.mean(diffs <= 1) * 100
    dangerous_err = np.mean(diffs >= 2) * 100

    missed_severe = sum(
        1 for p, t in zip(all_preds, all_labels) if p == 0 and t in [3, 4]
    )

    logger.info(f"\nOrdinal Error Analysis:")
    logger.info(f"Average prediction error: {avg_err:.2f} grades")
    logger.info(f"Predictions off by <= 1 grade: {off_by_1:.2f}%")
    logger.info(f"Dangerous errors (off by >= 2): {dangerous_err:.2f}%")
    logger.info(f"Count of Grade 0 predictions when truth was 3/4: {missed_severe}")

    logger.info(
        f"\nTraining Summary: Best Epoch: {checkpoint['epoch']} | Best Val QWK: {checkpoint['val_qwk']:.4f} | Time: {total_time:.1f} min"
    )

    # Warnings
    if qwk_test < 0.70:
        logger.warning(
            "⚠️ QWK below target. Consider: (1) ordinal regression loss, (2) label smoothing, (3) focal loss for rare classes."
        )
    if recalls_test[4] < 0.70:
        logger.warning(
            "⚠️ Proliferative DR recall below threshold. These patients risk blindness without treatment. Increase Grade 4 class weight."
        )
    if combined_recall < 0.75:
        logger.warning("⚠️ Severe DR detection is inadequate for clinical use.")
    if dangerous_err > 5.0:
        logger.warning(
            f"⚠️ > 5% of predictions are dangerous errors ({dangerous_err:.1f}%)."
        )


if __name__ == "__main__":
    train_model()
