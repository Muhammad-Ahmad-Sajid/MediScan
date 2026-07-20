import os
import sys
import shutil
import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def list_directory_contents(dir_path):
    print(f"\n--- Directory contents of '{dir_path}' ---")
    try:
        path = Path(dir_path)
        if path.exists() and path.is_dir():
            for item in sorted(os.listdir(path)):
                item_path = path / item
                prefix = "[DIR] " if item_path.is_dir() else "[FILE]"
                size_str = f" ({item_path.stat().st_size} bytes)" if item_path.is_file() else ""
                print(f"  {prefix} {item}{size_str}")
        else:
            print(f"  Path '{dir_path}' does not exist or is not a directory.")
    except Exception as e:
        print(f"  Error reading directory: {e}")
    print("----------------------------------------")


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare Stage 2 Unified Dataset")
    parser.add_argument(
        "--fracatlas_path",
        type=str,
        default="d:/X-ray ML Model/fracatlas",
        help="Path to the FracAtlas dataset root",
    )
    parser.add_argument(
        "--archive_path",
        type=str,
        default="d:/X-ray ML Model/archive/Bone_Fracture_Binary_Classification/Bone_Fracture_Binary_Classification",
        help="Path to the Archive dataset root",
    )
    parser.add_argument(
        "--dest_path",
        type=str,
        default="d:/X-ray ML Model/data/stage2",
        help="Destination directory for Stage 2 data",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="d:/X-ray ML Model/stage2_labels.csv",
        help="Path to output the unified master CSV file",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    frac_root = Path(args.fracatlas_path)
    frac_images = frac_root / "images"
    frac_csv = frac_root / "dataset.csv"

    print("=" * 80)
    print("STAGE 2 DATA PREPARATION DAEMON")
    print("=" * 80)
    print(f"Checking FracAtlas images directory at: {frac_images}")

    # 1. Early exit check if FracAtlas images folder is missing
    if not frac_images.exists():
        print(f"\n[!] ALERT: FracAtlas images folder is missing or wrong at: {frac_images}")
        # Print actual contents of d:/X-ray ML Model/fracatlas/
        list_directory_contents("d:/X-ray ML Model/fracatlas")
        print("\nExiting cleanly as requested.")
        sys.exit(0)

    print("[*] FracAtlas images folder found. Proceeding with dataset preparation...")

    archive_root = Path(args.archive_path)
    dest_root = Path(args.dest_path)
    train_dest = dest_root / "train"
    val_dest = dest_root / "val"

    # Create output directories
    train_dest.mkdir(parents=True, exist_ok=True)
    val_dest.mkdir(parents=True, exist_ok=True)

    records = []  # Stores metadata of copied files to build CSV and print summary

    # ==========================================
    # DATASET 1: FracAtlas
    # ==========================================
    print("\n--- Processing FracAtlas ---")
    if not frac_csv.exists():
        print(f"Error: dataset.csv not found at {frac_csv}")
        sys.exit(1)

    df_raw = pd.read_csv(frac_csv)
    print(f"Total rows in raw dataset.csv: {len(df_raw)}")

    # Filtering & body region determination
    filtered_frac_rows = []
    for _, row in df_raw.iterrows():
        # Skip if mixed is 1
        if row["mixed"] == 1:
            continue

        # Determine body region
        active_regions = []
        for col in ["hand", "leg", "hip", "shoulder"]:
            if row[col] == 1:
                active_regions.append(col)

        # Skip if none or multiple regions are active
        if len(active_regions) != 1:
            continue

        region = active_regions[0]
        fractured_val = int(row["fractured"])  # 1 or 0

        filtered_frac_rows.append(
            {
                "image_id": row["image_id"],
                "body_region": region,
                "fractured": fractured_val,
            }
        )

    df_frac = pd.DataFrame(filtered_frac_rows)
    print(f"Usable FracAtlas images after filtering: {len(df_frac)}")

    # Stratified split: combine body_region and fractured status
    df_frac["stratify_col"] = df_frac["body_region"] + "_" + df_frac["fractured"].astype(str)

    frac_train_df, frac_val_df = train_test_split(
        df_frac, test_size=0.2, random_state=42, stratify=df_frac["stratify_col"]
    )

    print(f"FracAtlas Split: Train={len(frac_train_df)}, Val={len(frac_val_df)}")

    # Copy and rename function helper to avoid name collisions
    def copy_file_safe(src_path, dest_dir, prefix):
        src_path = Path(src_path)
        dest_dir = Path(dest_dir)
        original_name = src_path.name
        dest_file = dest_dir / f"{prefix}_{original_name}"

        # Handle collision
        if dest_file.exists():
            stem = dest_file.stem
            suffix = dest_file.suffix
            counter = 1
            while True:
                new_name = f"{stem}_{counter}{suffix}"
                new_dest = dest_dir / new_name
                if not new_dest.exists():
                    dest_file = new_dest
                    break
                counter += 1

        shutil.copy2(src_path, dest_file)
        return dest_file

    # Helper to find FracAtlas image
    def find_fracatlas_image(image_id):
        # Check subfolders "Fractured" and "Non_fractured"
        p_frac = frac_images / "Fractured" / image_id
        if p_frac.exists():
            return p_frac
        p_non = frac_images / "Non_fractured" / image_id
        if p_non.exists():
            return p_non
        p_flat = frac_images / image_id
        if p_flat.exists():
            return p_flat
        return None

    # Copy FracAtlas Train
    print("Copying FracAtlas train images...")
    for _, row in frac_train_df.iterrows():
        image_id = row["image_id"]
        src_img = find_fracatlas_image(image_id)
        if src_img is None:
            print(f"Warning: Image {image_id} not found in {frac_images}")
            continue

        copied_path = copy_file_safe(src_img, train_dest, "fracatlas")
        # Relative path using forward slashes
        rel_path = "data/stage2/train/" + copied_path.name
        records.append(
            {
                "source": "FracAtlas",
                "split": "train",
                "fractured": "yes" if row["fractured"] == 1 else "no",
                "region": row["body_region"],
                "image_path": rel_path,
            }
        )

    # Copy FracAtlas Val
    print("Copying FracAtlas val images...")
    for _, row in frac_val_df.iterrows():
        image_id = row["image_id"]
        src_img = find_fracatlas_image(image_id)
        if src_img is None:
            print(f"Warning: Image {image_id} not found in {frac_images}")
            continue

        copied_path = copy_file_safe(src_img, val_dest, "fracatlas")
        rel_path = "data/stage2/val/" + copied_path.name
        records.append(
            {
                "source": "FracAtlas",
                "split": "val",
                "fractured": "yes" if row["fractured"] == 1 else "no",
                "region": row["body_region"],
                "image_path": rel_path,
            }
        )

    # ==========================================
    # DATASET 2: Archive
    # ==========================================
    print("\n--- Processing Archive ---")
    if not archive_root.exists():
        print(f"Error: Archive directory not found at {archive_root}")
        sys.exit(1)

    # Read files in train/ and test/ to merge them
    archive_train_test_list = []

    for split_dir in ["train", "test"]:
        for category in ["fractured", "not fractured"]:
            cat_dir = archive_root / split_dir / category
            if not cat_dir.exists():
                continue

            for file in cat_dir.iterdir():
                if file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    archive_train_test_list.append(
                        {
                            "file_path": file,
                            "fractured": 1 if category == "fractured" else 0,
                        }
                    )

    df_archive_tt = pd.DataFrame(archive_train_test_list)
    print(f"Total Archive (train+test) images to split: {len(df_archive_tt)}")

    # Stratified split by fractured status only
    arch_train_df, arch_val_df = train_test_split(
        df_archive_tt,
        test_size=0.2,
        random_state=42,
        stratify=df_archive_tt["fractured"],
    )
    print(f"Archive Split (from train+test): Train={len(arch_train_df)}, Val={len(arch_val_df)}")

    # Copy Archive Train
    print("Copying Archive train images...")
    for _, row in arch_train_df.iterrows():
        src_img = row["file_path"]
        copied_path = copy_file_safe(src_img, train_dest, "archive")
        rel_path = "data/stage2/train/" + copied_path.name
        records.append(
            {
                "source": "Archive",
                "split": "train",
                "fractured": "yes" if row["fractured"] == 1 else "no",
                "region": "unknown",
                "image_path": rel_path,
            }
        )

    # Copy Archive Val
    print("Copying Archive val images (from re-split)...")
    for _, row in arch_val_df.iterrows():
        src_img = row["file_path"]
        copied_path = copy_file_safe(src_img, val_dest, "archive")
        rel_path = "data/stage2/val/" + copied_path.name
        records.append(
            {
                "source": "Archive",
                "split": "val",
                "fractured": "yes" if row["fractured"] == 1 else "no",
                "region": "unknown",
                "image_path": rel_path,
            }
        )

    # Also copy original val images directly to stage2/val
    original_val_list = []
    for category in ["fractured", "not fractured"]:
        cat_dir = archive_root / "val" / category
        if cat_dir.exists():
            for file in cat_dir.iterdir():
                if file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    original_val_list.append(
                        {
                            "file_path": file,
                            "fractured": 1 if category == "fractured" else 0,
                        }
                    )

    print(f"Copying Archive original val images ({len(original_val_list)} files) directly to validation...")
    for row in original_val_list:
        src_img = row["file_path"]
        copied_path = copy_file_safe(src_img, val_dest, "archive")
        rel_path = "data/stage2/val/" + copied_path.name
        records.append(
            {
                "source": "Archive",
                "split": "val",
                "fractured": "yes" if row["fractured"] == 1 else "no",
                "region": "unknown",
                "image_path": rel_path,
            }
        )

    # ==========================================
    # OUTPUTS & CHECKS
    # ==========================================

    # 1. Write master CSV
    # CSV Columns: image_path, fractured (0 or 1), body_region
    csv_records = []
    for r in records:
        fractured_int = 1 if r["fractured"] == "yes" else 0
        csv_records.append(
            {
                "image_path": r["image_path"],
                "fractured": fractured_int,
                "body_region": r["region"],
            }
        )

    df_out = pd.DataFrame(csv_records)
    # Output to both designated locations for maximum integration safety
    df_out.to_csv(args.output_csv, index=False)
    # Also save inside the data/stage2 folder
    df_out.to_csv(dest_root / "stage2_labels.csv", index=False)
    print(f"\n[+] Master CSV saved successfully to: {args.output_csv}")
    print(f"[+] Backup master CSV saved to: {dest_root}/stage2_labels.csv")

    # 2. Print Summary Table
    print("\n" + "=" * 80)
    print("STAGE 2 FINE-TUNING DATA SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Source':<10} | {'Split':<5} | {'Fractured':<9} | {'Region':<8} | {'Count'}")
    print("-" * 50)

    # Combinations list to guarantee order
    combinations = []
    for split in ["train", "val"]:
        for fractured_str in ["yes", "no"]:
            for region in ["hand", "leg", "hip", "shoulder"]:
                combinations.append(("FracAtlas", split, fractured_str, region))

    for split in ["train", "val"]:
        for fractured_str in ["yes", "no"]:
            combinations.append(("Archive", split, fractured_str, "unknown"))

    for src, split, fractured_str, region in combinations:
        cnt = sum(
            1
            for r in records
            if r["source"] == src and r["split"] == split and r["fractured"] == fractured_str and r["region"] == region
        )
        print(f"{src:<10} | {split:<5} | {fractured_str:<9} | {region:<8} | {cnt}")

    total_train = sum(1 for r in records if r["split"] == "train")
    total_val = sum(1 for r in records if r["split"] == "val")
    total_combined = len(records)

    print("-" * 50)
    print(f"Total train images: {total_train}")
    print(f"Total val images:   {total_val}")
    print(f"Total combined:     {total_combined}")
    print("=" * 80)

    # 3. Class Balance Checks
    train_frac_yes = sum(1 for r in records if r["split"] == "train" and r["fractured"] == "yes")
    train_frac_no = sum(1 for r in records if r["split"] == "train" and r["fractured"] == "no")

    ratio_yes = train_frac_yes / total_train if total_train > 0 else 0
    ratio_no = train_frac_no / total_train if total_train > 0 else 0

    print("\nClass Balance Check (Train Set):")
    print(f"  Fractured (yes):     {train_frac_yes} ({ratio_yes*100.0:.2f}%)")
    print(f"  Not Fractured (no):  {train_frac_no} ({ratio_no*100.0:.2f}%)")

    if ratio_yes > 0.70 or ratio_no > 0.70:
        print("  [WARNING] Fractured vs not fractured ratio is worse than 70/30!")
    else:
        print("  [OK] Fractured vs not fractured ratio is within the balanced 70/30 threshold.")

    print("\nRegion Distribution (Train Set):")
    for reg in ["hand", "leg", "hip", "shoulder", "unknown"]:
        reg_count = sum(1 for r in records if r["split"] == "train" and r["region"] == reg)
        pct = reg_count / total_train * 100 if total_train > 0 else 0
        print(f"  {reg:<8}: {reg_count} ({pct:.2f}%)")

    print("=" * 80)


if __name__ == "__main__":
    main()
