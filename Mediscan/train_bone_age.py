import os
import cv2
import time
import torch
import logging
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from sklearn.model_selection import train_test_split

BASE_DIR = "d:/X-ray ML Model/Mediscan/bone_age"
CHECKPOINT_PATH = "checkpoints/bone_age_best.pth"
LATEST_PATH = "checkpoints/bone_age_latest.pth"
HISTORY_PATH = "checkpoints/bone_age_training_history.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
NUM_WORKERS = 0
EPOCHS = 30
PATIENCE = 5
MAX_AGE_MONTHS = 228.0

logger = logging.getLogger("mediscan.bone_age")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
if not logger.handlers:
    logger.addHandler(ch)


class AddGaussianNoise:
    def __init__(self, mean=0.0, std=1.0, p=0.2):
        self.std = std
        self.mean = mean
        self.p = p

    def __call__(self, tensor):
        if torch.rand(1).item() < self.p:
            return tensor + torch.randn(tensor.size()) * self.std + self.mean
        return tensor


def get_train_transforms():
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.15),
            transforms.RandomAffine(degrees=0, scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            AddGaussianNoise(p=0.2),
        ]
    )


def get_val_transforms():
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


clahe_obj = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))


def preprocess_image(img_path):
    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        img_gray = np.zeros((224, 224), dtype=np.uint8)
    img_clahe = clahe_obj.apply(img_gray)
    img_rgb = cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2RGB)
    return img_rgb


class BoneAgeDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, use_gender=False):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.use_gender = use_gender

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        age = row["age_months"]

        img_rgb = preprocess_image(img_path)
        tensor = self.transform(img_rgb)

        norm_age = age / MAX_AGE_MONTHS

        if self.use_gender:
            gender = row["gender"]
            return (
                tensor,
                torch.tensor([gender], dtype=torch.float32),
                torch.tensor([norm_age], dtype=torch.float32),
                age,
            )
        return tensor, torch.tensor([norm_age], dtype=torch.float32), age


class BoneAgeModel(nn.Module):
    def __init__(self, use_gender=False):
        super().__init__()
        self.use_gender = use_gender
        self.backbone = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        input_features = num_ftrs + 1 if use_gender else num_ftrs
        self.regression_head = nn.Sequential(
            nn.Linear(input_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
        )

    def forward(self, x, gender=None):
        features = self.backbone(x)
        if self.use_gender and gender is not None:
            features = torch.cat([features, gender], dim=1)
        return self.regression_head(features)


def perform_discovery():
    logger.info("=== DATASET DISCOVERY ===")
    total = 0
    img_dir = None
    for root, dirs, files in os.walk(BASE_DIR):
        level = root.replace(BASE_DIR, "").count(os.sep)
        indent = "  " * level
        img_count = len(
            [f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        )
        other = len([f for f in files if f.lower().endswith((".csv", ".json", ".txt"))])
        logger.info(
            f"{indent}{os.path.basename(root)}/  [{img_count} images, {other} metadata files]"
        )
        if img_count > 0:
            if "training" in root.lower() or img_dir is None:
                img_dir = root
        total += img_count

    csv_path = None
    for root, dirs, files in os.walk(BASE_DIR):
        for f in files:
            if f.lower().endswith(".csv"):
                found_csv = os.path.join(root, f)
                logger.info(f"\nFound: {found_csv}")
                if "training" in f.lower() or csv_path is None:
                    csv_path = found_csv
                try:
                    df = pd.read_csv(csv_path)
                    logger.info(f"Shape: {df.shape}")
                    logger.info(f"Columns: {df.columns.tolist()}")
                    logger.info(f"Head:\n{df.head(10)}")
                    for col in df.columns:
                        if "age" in col.lower() or "month" in col.lower():
                            logger.info(
                                f"\nAge column '{col}' stats:\n{df[col].describe()}"
                            )
                except Exception as e:
                    logger.error(f"Error reading CSV: {e}")

    if csv_path is not None:
        logger.info(f"\nSelected metadata: {csv_path}")
    return img_dir, csv_path


def prepare_data(csv_path, img_dir):
    df = pd.read_csv(csv_path)

    age_col = None
    for col in df.columns:
        if "age" in col.lower() or "month" in col.lower():
            age_col = col
            break

    if age_col is None:
        raise ValueError("Could not find age column in CSV")

    df["age_months"] = df[age_col]

    use_gender = False
    gender_col = None
    for col in df.columns:
        if col.lower() in ["gender", "sex", "male"]:
            gender_col = col
            break

    if gender_col:
        logger.info("Gender feature: included")
        use_gender = True
        df["gender"] = df[gender_col].apply(
            lambda x: 0 if str(x).lower() in ["m", "male", "true", "1"] else 1
        )
    else:
        logger.info("Gender feature: not available")

    id_col = df.columns[0]

    valid_data = []
    for _, row in df.iterrows():
        img_id = str(row[id_col])
        for ext in [".png", ".jpg", ".jpeg"]:
            path = os.path.join(img_dir, img_id + ext)
            if os.path.exists(path):
                row_dict = row.to_dict()
                row_dict["image_path"] = path
                valid_data.append(row_dict)
                break

    df_valid = pd.DataFrame(valid_data)
    logger.info(f"Found {len(df_valid)} valid images matching CSV records")

    if len(df_valid) == 0:
        return None, None, None, use_gender

    bins = [0, 24, 60, 120, 180, 228, float("in")]
    labels = ["0-24", "25-60", "61-120", "121-180", "181-228", "228+"]
    df_valid["age_bin"] = pd.cut(
        df_valid["age_months"], bins=bins, labels=labels, right=True
    )

    train_df, test_val_df = train_test_split(
        df_valid, test_size=0.2, stratify=df_valid["age_bin"], random_state=42
    )
    val_df, test_df = train_test_split(
        test_val_df, test_size=0.5, stratify=test_val_df["age_bin"], random_state=42
    )

    logger.info("  Split | Count | Age Range | Mean Age | Std Age")
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        logger.info(
            f"  {name:5s} | {len(split_df):5d} | {split_df['age_months'].min():.0f}-{split_df['age_months'].max():.0f}     | {split_df['age_months'].mean():.1f}    | {split_df['age_months'].std():.1f}"
        )

    return train_df, val_df, test_df, use_gender


def train_one_epoch(model, dataloader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    all_preds, all_actuals = [], []

    for batch in dataloader:
        if len(batch) == 4:
            inputs, genders, labels, actuals = batch
            inputs, genders, labels = (
                inputs.to(DEVICE),
                genders.to(DEVICE),
                labels.to(DEVICE),
            )
            outputs = model(inputs, genders)
        else:
            inputs, labels, actuals = batch
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)

        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        preds_months = outputs.detach().cpu().numpy() * MAX_AGE_MONTHS
        all_preds.extend(preds_months.flatten())
        all_actuals.extend(actuals.numpy().flatten())

    epoch_loss = running_loss / len(dataloader.dataset)
    mae = np.mean(np.abs(np.array(all_preds) - np.array(all_actuals)))
    return epoch_loss, mae


def evaluate(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    all_preds, all_actuals = [], []

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 4:
                inputs, genders, labels, actuals = batch
                inputs, genders, labels = (
                    inputs.to(DEVICE),
                    genders.to(DEVICE),
                    labels.to(DEVICE),
                )
                outputs = model(inputs, genders)
            else:
                inputs, labels, actuals = batch
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)

            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

            preds_months = outputs.cpu().numpy() * MAX_AGE_MONTHS
            all_preds.extend(preds_months.flatten())
            all_actuals.extend(actuals.numpy().flatten())

    epoch_loss = running_loss / len(dataloader.dataset)

    preds_arr = np.array(all_preds)
    actuals_arr = np.array(all_actuals)

    mae = np.mean(np.abs(preds_arr - actuals_arr))
    rmse = np.sqrt(np.mean((preds_arr - actuals_arr) ** 2))
    within_12 = np.mean(np.abs(preds_arr - actuals_arr) <= 12) * 100

    return epoch_loss, mae, rmse, within_12, preds_arr, actuals_arr


def main():
    if not os.path.exists(BASE_DIR):
        logger.info(f"Directory {BASE_DIR} does not exist. Please download dataset.")
        return

    img_dir, csv_path = perform_discovery()

    if img_dir is None or csv_path is None:
        logger.info("Dataset structure incomplete. Exiting.")
        return

    train_df, val_df, test_df, use_gender = prepare_data(csv_path, img_dir)
    if train_df is None:
        logger.info("No valid data constructed. Exiting.")
        return

    train_ds = BoneAgeDataset(train_df, img_dir, get_train_transforms(), use_gender)
    val_ds = BoneAgeDataset(val_df, img_dir, get_val_transforms(), use_gender)
    test_ds = BoneAgeDataset(test_df, img_dir, get_val_transforms(), use_gender)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    model = BoneAgeModel(use_gender).to(DEVICE)

    for name, param in model.named_parameters():
        if not ("layer3" in name or "layer4" in name or "regression_head" in name):
            param.requires_grad = False

    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,
        weight_decay=1e-5,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5
    )

    best_val_mae = float("inf")
    patience_counter = 0
    best_epoch = 0
    best_rmse = 0

    logger.info(
        "Epoch | Train Loss | Train MAE | Val Loss | Val MAE (months) | Val RMSE | Within ±12mo | LR"
    )
    logger.info("-" * 95)

    start_time = time.time()

    start_epoch = 1
    if os.path.exists(LATEST_PATH):
        logger.info(f"Resuming from checkpoint: {LATEST_PATH}")
        checkpoint = torch.load(LATEST_PATH, map_location=DEVICE, weights_only=False)
        start_epoch = checkpoint["epoch"] + 1
        best_val_mae = checkpoint["best_val_mae"]
        best_rmse = checkpoint["best_rmse"]
        best_epoch = checkpoint["best_epoch"]
        patience_counter = checkpoint["patience_counter"]
        model.load_state_dict(checkpoint["state_dict"])

        if start_epoch > 6:
            for param in model.parameters():
                param.requires_grad = True
            optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", patience=3, factor=0.5
            )

        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    for epoch in range(start_epoch, EPOCHS + 1):
        if epoch == 6:
            for param in model.parameters():
                param.requires_grad = True
            optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", patience=3, factor=0.5
            )

        train_loss, train_mae = train_one_epoch(
            model, train_loader, optimizer, criterion
        )
        val_loss, val_mae, val_rmse, within_12, _, _ = evaluate(
            model, val_loader, criterion
        )

        lr = optimizer.param_groups[0]["lr"]
        logger.info(
            f"{epoch:5d} | {train_loss:10.4f} | {train_mae:9.2f} | {val_loss:8.4f} | {val_mae:16.2f} | {val_rmse:8.2f} | {within_12:11.1f}% | {lr:.2e}"
        )
        logger.info(f"Bone Age MAE: {val_mae:.1f} months — target below 12 months")

        if epoch >= 10 and val_mae > 20.0:
            logger.warning(
                "⚠️ MAE still above 20 months. Model may not be converging properly. Check: (1) age normalization, (2) learning rate, (3) data loading."
            )

        scheduler.step(val_mae)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_rmse = val_rmse
            best_epoch = epoch
            patience_counter = 0
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "val_mae": val_mae,
                    "val_rmse": val_rmse,
                    "age_normalization_factor": MAX_AGE_MONTHS,
                },
                CHECKPOINT_PATH,
            )
        else:
            patience_counter += 1

        torch.save(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_mae": best_val_mae,
                "best_rmse": best_rmse,
                "best_epoch": best_epoch,
                "patience_counter": patience_counter,
            },
            LATEST_PATH,
        )

        if patience_counter >= PATIENCE:
            logger.info(f"Early stopping triggered at epoch {epoch}")
            break

    total_time = time.time() - start_time

    model.load_state_dict(torch.load(CHECKPOINT_PATH)["state_dict"])
    _, test_mae, test_rmse, test_within_12, preds, actuals = evaluate(
        model, test_loader, criterion
    )

    print("\n" + "=" * 50)
    print("FINAL EVALUATION")
    print("=" * 50)
    print(f"MAE (months)         : {test_mae:.2f}")
    print(f"RMSE (months)        : {test_rmse:.2f}")

    within_6 = np.mean(np.abs(preds - actuals) <= 6) * 100
    within_24 = np.mean(np.abs(preds - actuals) <= 24) * 100
    print(f"Within ±6 months     : {within_6:.1f}%")
    print(f"Within ±12 months    : {test_within_12:.1f}%")
    print(f"Within ±24 months    : {within_24:.1f}%")

    print("\nAge Group Breakdown:")
    print("Age Group      | Count | MAE (months) | Within ±12mo")

    bins = [0, 24, 60, 120, 180, 228]
    labels = [
        "0-24 months",
        "25-60 months",
        "61-120 months",
        "121-180 months",
        "181-228 months",
    ]

    for i in range(len(bins) - 1):
        mask = (actuals > bins[i]) & (actuals <= bins[i + 1])
        if np.sum(mask) > 0:
            g_mae = np.mean(np.abs(preds[mask] - actuals[mask]))
            g_w12 = np.mean(np.abs(preds[mask] - actuals[mask]) <= 12) * 100
            print(
                f"{labels[i]:14s} | {np.sum(mask):5d} | {g_mae:12.1f} | {g_w12:11.1f}%"
            )
            if g_mae > 24:
                print(f"  ⚠️ Warning: MAE for {labels[i]} exceeds 24 months")

    print("\nFirst 20 pairs (Actual, Predicted):")
    for a, p in zip(actuals[:20], preds[:20]):
        print(f"Actual: {a:5.1f} | Predicted: {p:5.1f}")

    print("\nTraining Summary:")
    print(
        f"Best Epoch: {best_epoch} | Best MAE: {best_val_mae:.2f} | Best RMSE: {best_rmse:.2f} | Time: {total_time/60:.1f} min"
    )

    if test_mae > 12:
        print(
            "\n⚠️ Bone age MAE exceeds clinical target of 12 months. Consider: (1) longer training, (2) including gender as input feature, (3) attention mechanism on hand/wrist region."
        )
    elif test_mae < 6:
        print("\n✅ Excellent bone age estimation — MAE below 6 months.")

    if test_within_12 < 70:
        print("⚠️ Warning: Less than 70% of predictions are within ±12 months.")


if __name__ == "__main__":
    main()
