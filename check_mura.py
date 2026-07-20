import os
import sys
from pathlib import Path
import pandas as pd
from PIL import Image


def main():
    print("=" * 80)
    print("MURA DATASET INTEGRITY & USABILITY CHECKER")
    print("=" * 80)
    print()

    mura_images_dir = Path("mura_images")
    metadata_csv_path = Path("mura_metadata.csv")

    checks = {}

    # Check 1: Verify directories exist
    print("[*] Check 1: Directory verification...")
    if mura_images_dir.exists() and mura_images_dir.is_dir():
        print("  [PASS] 'mura_images' directory exists.")
        checks["directory_exists"] = True
    else:
        print("  [FAIL] 'mura_images' directory not found.")
        checks["directory_exists"] = False

    # Check 2: Walk through folder, count extensions, check for corruption, and collect study types
    print("\n[*] Check 2: Scanning images and checking for corruption...")
    extensions_count = {".png": 0, ".jpg": 0, ".jpeg": 0, ".dcm": 0}
    other_extensions = {}
    corrupted_files = []
    study_type_counts = {
        "wrist": 0,
        "elbow": 0,
        "humerus": 0,
        "shoulder": 0,
        "forearm": 0,
        "finger": 0,
        "hand": 0,
        "unknown": 0,
    }

    total_scanned = 0

    if checks["directory_exists"]:
        for root, _, files in os.walk(mura_images_dir):
            for file in files:
                file_path = Path(root) / file
                ext = file_path.suffix.lower()

                # Filter for image extensions we care about
                if ext in [".png", ".jpg", ".jpeg", ".dcm"]:
                    extensions_count[ext] = extensions_count.get(ext, 0) + 1
                    total_scanned += 1

                    # Identify study type from folder path
                    parts = file_path.parts
                    detected_study = "unknown"
                    for part in parts:
                        if part.startswith("XR_"):
                            detected_study = part.replace("XR_", "").lower()
                            break

                    if detected_study in study_type_counts:
                        study_type_counts[detected_study] += 1
                    else:
                        study_type_counts["unknown"] += 1

                    # Check for corruption (PIL read check for standard images, skip DCM PIL opening)
                    if ext in [".png", ".jpg", ".jpeg"]:
                        try:
                            with Image.open(file_path) as img:
                                img.verify()  # verify integrity
                        except Exception:
                            corrupted_files.append(str(file_path))
                else:
                    other_extensions[ext] = other_extensions.get(ext, 0) + 1

        print(f"  Scanned {total_scanned} images in total.")
        print(f"  Extensions found: {dict(extensions_count)}")
        if len(other_extensions) > 0:
            print(f"  Other files found: {other_extensions}")

        if len(corrupted_files) == 0:
            print("  [PASS] No corrupted or unreadable images found.")
            checks["no_corruption"] = True
        else:
            print(f"  [FAIL] Found {len(corrupted_files)} corrupted images.")
            checks["no_corruption"] = False
    else:
        checks["no_corruption"] = False

    # Check 3: Verify Metadata CSV
    print("\n[*] Check 3: Metadata CSV verification...")
    if metadata_csv_path.exists():
        try:
            df = pd.read_csv(metadata_csv_path)
            print("  [PASS] 'mura_metadata.csv' exists and is readable.")
            checks["csv_exists"] = True

            # Check expected columns
            expected_cols = ["file_path", "label", "study_type"]
            missing_cols = [col for col in expected_cols if col not in df.columns]
            if len(missing_cols) == 0:
                print(
                    "  [PASS] CSV has all expected columns: file_path, label, study_type."
                )
                checks["csv_columns"] = True
            else:
                print(f"  [FAIL] CSV is missing columns: {missing_cols}")
                checks["csv_columns"] = False
        except Exception as e:
            print(f"  [FAIL] Failed to read metadata CSV: {e}")
            checks["csv_exists"] = False
            checks["csv_columns"] = False
    else:
        print("  [FAIL] 'mura_metadata.csv' not found.")
        checks["csv_exists"] = False
        checks["csv_columns"] = False

    # 4. Print Summary Table
    print("\n" + "=" * 50)
    print("MURA DATASET SUMMARY")
    print("=" * 50)
    print(f"{'Metric':<25} | {'Value':<15}")
    print("-" * 50)
    print(f"{'Total Images Found':<25} | {total_scanned:<15}")
    print(f"{'Total Corrupted Files':<25} | {len(corrupted_files):<15}")
    print("-" * 50)
    print("Study Type Breakdown:")
    for study, count in study_type_counts.items():
        print(f" - {study:<22} | {count:<15}")
    print("=" * 50)

    # 5. Overall Status
    print("\n" + "=" * 50)
    print("OVERALL CHECKLIST STATUS")
    print("=" * 50)

    all_passed = True
    for name, status in checks.items():
        status_str = "PASS" if status else "FAIL"
        print(f"Check {name:<20}: {status_str}")
        if not status:
            all_passed = False

    print("-" * 50)
    if all_passed:
        print(">>> OVERALL STATUS: PASS <<<")
        sys.exit(0)
    else:
        print(">>> OVERALL STATUS: FAIL <<<")
        sys.exit(1)


if __name__ == "__main__":
    main()
