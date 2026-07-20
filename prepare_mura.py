import shutil
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    print("=" * 80)
    print("MURA DATASET ORGANIZER & SPLITTER")
    print("=" * 80)

    # Define paths
    metadata_csv_path = Path("d:/X-ray ML Model/mura_metadata.csv")
    dest_dir = Path("d:/X-ray ML Model/data/mura")

    # 1. Read metadata
    if not metadata_csv_path.exists():
        print(f"[!] Metadata CSV not found at {metadata_csv_path}")
        return

    df = pd.read_csv(metadata_csv_path)
    print(f"Loaded {len(df)} records from metadata CSV.")

    # 2. Filter out macOS resource files starting with ._ in path or name
    original_len = len(df)

    # We also check if the file path ends with a filename starting with ._
    df = df[~df["file_path"].apply(lambda x: Path(x).name.startswith("._"))].reset_index(drop=True)
    skipped_resource_files = original_len - len(df)
    print(f"Filtered out {skipped_resource_files} macOS resource files.")

    # 3. Create stratified 85/15 train/val split based on study_type
    # We also include label in stratification if we want both study_type and class to be balanced
    df["stratify_col"] = df["study_type"] + "_" + df["label"]

    train_df, val_df = train_test_split(df, test_size=0.15, random_state=42, stratify=df["stratify_col"])

    print(f"Stratified split completed: Train size = {len(train_df)}, Val size = {len(val_df)}")

    # 4. Copy files to data/mura/
    # Target folders structure:
    #   data/mura/train/normal/
    #   data/mura/train/abnormal/
    #   data/mura/val/normal/
    #   data/mura/val/abnormal/

    # Clean output directories if they exist
    if dest_dir.exists():
        print(f"Cleaning existing destination directory {dest_dir}...")
        shutil.rmtree(dest_dir)

    for split in ["train", "val"]:
        for label in ["normal", "abnormal"]:
            (dest_dir / split / label).mkdir(parents=True, exist_ok=True)

    print("Copying and renaming images to destination directories...")

    counts = {
        ("train", "normal"): 0,
        ("train", "abnormal"): 0,
        ("val", "normal"): 0,
        ("val", "abnormal"): 0,
    }

    # Perform copying for train split
    for _, row in train_df.iterrows():
        src_path = Path(row["file_path"])
        label = row["label"]
        study_type = row["study_type"]

        # To avoid name collisions (since MURA images are named image1.png, image2.png etc.),
        # we extract patient ID and study ID from the path.
        # Example path: .../XR_WRIST/patient07840/study1_negative/image1.png
        parts = src_path.parts
        patient_id = "patient"
        study_id = "study"
        for part in parts:
            if "patient" in part:
                patient_id = part
            if "study" in part:
                study_id = part

        # Unique target name: {study_type}_{patient_id}_{study_id}_{original_filename}
        unique_name = f"{study_type}_{patient_id}_{study_id}_{src_path.name}"
        dest_path = dest_dir / "train" / label / unique_name

        shutil.copy2(src_path, dest_path)
        counts[("train", label)] += 1

    # Perform copying for val split
    for _, row in val_df.iterrows():
        src_path = Path(row["file_path"])
        label = row["label"]
        study_type = row["study_type"]

        parts = src_path.parts
        patient_id = "patient"
        study_id = "study"
        for part in parts:
            if "patient" in part:
                patient_id = part
            if "study" in part:
                study_id = part

        unique_name = f"{study_type}_{patient_id}_{study_id}_{src_path.name}"
        dest_path = dest_dir / "val" / label / unique_name

        shutil.copy2(src_path, dest_path)
        counts[("val", label)] += 1

    total_copied = sum(counts.values())

    # 5. Output Summary Table
    print("\n" + "=" * 50)
    print(f"{'Split':<10} | {'Class':<12} | {'Count':<10}")
    print("-" * 50)
    print(f"{'train':<10} | {'normal':<12} | {counts[('train', 'normal')]:<10}")
    print(f"{'train':<10} | {'abnormal':<12} | {counts[('train', 'abnormal')]:<10}")
    print(f"{'val':<10} | {'normal':<12} | {counts[('val', 'normal')]:<10}")
    print(f"{'val':<10} | {'abnormal':<12} | {counts[('val', 'abnormal')]:<10}")
    print("-" * 50)
    print(f"Total copied: {total_copied}")
    print(f"Skipped (._): {skipped_resource_files}")
    print("=" * 50)


if __name__ == "__main__":
    main()
