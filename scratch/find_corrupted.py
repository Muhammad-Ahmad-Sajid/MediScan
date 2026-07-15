import os
from PIL import Image
from pathlib import Path


def main():
    mura_images_dir = Path("mura_images")
    corrupted = []

    print("Scanning for corrupted images...")
    for root, _, files in os.walk(mura_images_dir):
        for file in files:
            if file.lower().endswith(".png"):
                file_path = Path(root) / file
                try:
                    with Image.open(file_path) as img:
                        img.verify()
                except Exception as e:
                    print(f"Corrupted: {file_path} - Error: {e}")
                    corrupted.append(file_path)

    print(f"\nScan complete. Total corrupted files found: {len(corrupted)}")


if __name__ == "__main__":
    main()
