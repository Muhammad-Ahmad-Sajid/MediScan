from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# Import the preprocess function
from src.data_preparation.preprocess import preprocess_xray


class MURADataset(Dataset):
    """
    PyTorch Dataset for the MURA dataset.
    Loads images from data/mura/{split}/{normal|abnormal}
    Returns: (tensor, binary_label, file_path)
    """

    def __init__(self, base_dir: str = "data/mura", split: str = "train"):
        self.split = split
        self.base_dir = Path(base_dir) / split
        self.image_paths = []
        self.labels = []

        if not self.base_dir.exists():
            print(f"Warning: MURA directory not found at {self.base_dir}")
            return

        # Scan for normal and abnormal files
        for label_name, label_idx in [("normal", 0), ("abnormal", 1)]:
            class_dir = self.base_dir / label_name
            if class_dir.exists():
                for file in class_dir.iterdir():
                    if file.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                        self.image_paths.append(file)
                        self.labels.append(label_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Determine if augmentations should be applied (only during training)
        is_training = self.split == "train"

        # Preprocess and augment
        tensor = preprocess_xray(str(img_path), is_training=is_training)

        return tensor, label, str(img_path)


class FracAtlasDataset(Dataset):
    """
    PyTorch Dataset for the FracAtlas dataset.
    Loads images based on fracatlas_labels.csv
    Returns: (tensor, severity_label, bone_label, file_path)
    Filters to keep only fractured images to ensure 4 severity classes:
    ['hairline', 'simple', 'displaced', 'comminuted']
    """

    SEVERITY_MAP = {"hairline": 0, "simple": 1, "displaced": 2, "comminuted": 3}

    BONE_MAP = {
        "hand": 0,  # -> distal_radius (0)
        "shoulder": 1,  # -> clavicle (1)
        "leg": 2,  # -> ankle (2)
        "hip": 3,  # -> femur (3)
        "mixed": 4,  # -> humerus (4)
        "other": 5,  # -> metatarsal (5)
    }

    def __init__(self, csv_path: str = "fracatlas_labels.csv", split: str = "train"):
        self.split = split
        self.csv_path = Path(csv_path)

        if not self.csv_path.exists():
            print(f"Warning: FracAtlas labels CSV not found at {self.csv_path}")
            self.df = pd.DataFrame()
            return

        # Load labels and filter by split and fracture status
        df = pd.read_csv(self.csv_path)

        # Keep only records belonging to the current split (train/val)
        # Check image_path containing "/train/" or "/val/"
        split_pattern = f"/{split}/"
        df_split = df[df["image_path"].str.contains(split_pattern, case=False, na=False)]

        # Keep only fractured scans to map exactly to the 4 severity classes
        self.df = df_split[df_split["has_fracture"] is True].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # The image_path in CSV is relative, resolve it to absolute path
        # CSV path format: data/fracatlas/{train|val}/{class}/IMG_xxxx.jpg
        img_path = Path("d:/X-ray ML Model") / row["image_path"]

        severity_class = row["severity_class"]
        bone_affected = row["bone_affected"]

        severity_idx = self.SEVERITY_MAP.get(severity_class, 0)
        bone_idx = self.BONE_MAP.get(bone_affected, 5)

        # Determine if augmentations should be applied (only during training)
        is_training = self.split == "train"

        # Preprocess and augment
        tensor = preprocess_xray(str(img_path), is_training=is_training)

        return tensor, severity_idx, bone_idx, str(img_path)


def get_dataloaders(
    mura_dir: str = "data/mura",
    fracatlas_csv: str = "fracatlas_labels.csv",
    batch_size: int = 32,
    num_workers: int = 4,
):
    """
    Returns separate training and validation DataLoaders for both MURA and FracAtlas datasets.
    """
    print("\nInitializing PyTorch Datasets...")

    # 1. Create Datasets
    mura_train_ds = MURADataset(base_dir=mura_dir, split="train")
    mura_val_ds = MURADataset(base_dir=mura_dir, split="val")

    frac_train_ds = FracAtlasDataset(csv_path=fracatlas_csv, split="train")
    frac_val_ds = FracAtlasDataset(csv_path=fracatlas_csv, split="val")

    # Print dataset sizes
    print("=" * 60)
    print("DATASET SIZES")
    print("=" * 60)
    print(f"MURA Train size:      {len(mura_train_ds)} images")
    print(f"MURA Val size:        {len(mura_val_ds)} images")
    print(f"FracAtlas Train size: {len(frac_train_ds)} images")
    print(f"FracAtlas Val size:   {len(frac_val_ds)} images")
    print("=" * 60)

    # 2. Create DataLoaders
    # Note: Pin memory is set to True to optimize GPU transfer speed
    mura_train_loader = DataLoader(
        mura_train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    mura_val_loader = DataLoader(
        mura_val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    frac_train_loader = DataLoader(
        frac_train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    frac_val_loader = DataLoader(
        frac_val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return {
        "mura_train": mura_train_loader,
        "mura_val": mura_val_loader,
        "frac_train": frac_train_loader,
        "frac_val": frac_val_loader,
    }


class Stage2Dataset(Dataset):
    """
    PyTorch Dataset for the unified Stage 2 dataset.
    Loads images and labels based on stage2_labels.csv.
    Returns: (tensor, fracture_label, region_label, file_path)
    """

    REGION_MAP = {"hand": 0, "leg": 1, "hip": 2, "shoulder": 3, "unknown": 4}

    def __init__(self, csv_path: str = "d:/X-ray ML Model/stage2_labels.csv", split: str = "train"):
        self.split = split
        self.csv_path = Path(csv_path)

        if not self.csv_path.exists():
            print(f"Warning: Stage 2 labels CSV not found at {self.csv_path}")
            self.df = pd.DataFrame()
            return

        df_all = pd.read_csv(self.csv_path)
        # Split is determined by the image_path starting folder:
        # e.g., 'data/stage2/train/' or 'data/stage2/val/'
        split_pattern = f"data/stage2/{split}/"
        self.df = df_all[df_all["image_path"].str.startswith(split_pattern, na=False)].reset_index(
            drop=True
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # image_path in CSV is relative to the root workspace, e.g. data/stage2/train/fracatlas_...
        img_path = Path("d:/X-ray ML Model") / row["image_path"]

        fractured = int(row["fractured"])  # 0 or 1
        body_region = row["body_region"]
        region_idx = self.REGION_MAP.get(body_region, 4)

        # Determine if augmentations should be applied (only during training)
        is_training = self.split == "train"

        # Preprocess and augment
        tensor = preprocess_xray(str(img_path), is_training=is_training)

        return tensor, fractured, region_idx, str(img_path)


def get_stage2_dataloaders(
    csv_path: str = "d:/X-ray ML Model/stage2_labels.csv",
    batch_size: int = 16,
    num_workers: int = 0,
):
    """
    Returns training and validation dataloaders for Stage 2 fine-tuning.
    """
    print("\nInitializing Stage 2 PyTorch Datasets...")
    train_ds = Stage2Dataset(csv_path=csv_path, split="train")
    val_ds = Stage2Dataset(csv_path=csv_path, split="val")

    print("=" * 60)
    print("STAGE 2 DATASET SIZES")
    print("=" * 60)
    print(f"Stage 2 Train size: {len(train_ds)} images")
    print(f"Stage 2 Val size:   {len(val_ds)} images")
    print("=" * 60)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader
