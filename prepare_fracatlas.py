import sys
import json
from pathlib import Path

# Define paths
FRACATLAS_DIR = Path("d:/X-ray ML Model/FracAtlas1/FracAtlas")
IMAGES_DIR = FRACATLAS_DIR / "images"
COCO_JSON_PATH = FRACATLAS_DIR / "Annotations/COCO JSON/COCO_fracture_masks.json"
DATASET_CSV_PATH = FRACATLAS_DIR / "dataset.csv"
DEST_DIR = Path("d:/X-ray ML Model/data/fracatlas")
OUTPUT_CSV_PATH = Path("d:/X-ray ML Model/fracatlas_labels.csv")


def print_mismatch_and_exit(data, message):
    print("=" * 80)
    print("WARNING: JSON annotations file structure is different than expected!")
    print(message)
    print("=" * 80)
    print("\n--- FIRST IMAGE ENTRY ---")
    if "images" in data and len(data["images"]) > 0:
        print(json.dumps(data["images"][0], indent=2))
    else:
        print("No images found.")

    print("\n--- FIRST ANNOTATION ENTRY ---")
    if "annotations" in data and len(data["annotations"]) > 0:
        print(json.dumps(data["annotations"][0], indent=2))
    else:
        print("No annotations found.")

    print("\n--- CATEGORIES LIST ---")
    if "categories" in data:
        print(json.dumps(data["categories"], indent=2))
    else:
        print("No categories found.")
    print("=" * 80)
    sys.exit(0)


def main():
    print("FracAtlas Dataset Organizer")
    print("=" * 80)

    # 1. Load COCO JSON
    print(f"Loading annotations from {COCO_JSON_PATH}...")
    if not COCO_JSON_PATH.exists():
        print(f"Error: Annotations file not found at {COCO_JSON_PATH}")
        sys.exit(1)

    with open(COCO_JSON_PATH, "r") as f:
        try:
            coco_data = json.load(f)
        except Exception as e:
            print(f"Error parsing JSON: {e}")
            sys.exit(1)

    # 2. Check JSON structure for expected fracture_type and body_part fields
    # Standard COCO annotations do not have these keys. We check for their presence.
    first_annotation = coco_data.get("annotations", [{}])[0] if coco_data.get("annotations") else {}
    first_image = coco_data.get("images", [{}])[0] if coco_data.get("images") else {}

    # We inspect keys in the first annotation or image to see if there is any custom metadata
    has_fracture_type = "fracture_type" in first_annotation or "fracture_type" in first_image
    has_body_part = (
        "body_part" in first_annotation
        or "body_part" in first_image
        or "body_part" in first_image.get("metadata", {})
    )

    if not has_fracture_type or not has_body_part:
        msg = (
            "The JSON file does not contain 'fracture_type' or 'body_part' keys in the image/annotation objects.\n"
            "COCO Annotations only contain: " + str(list(first_annotation.keys()))
        )
        print_mismatch_and_exit(coco_data, msg)

    # If the JSON did match the expected schema, the rest of the script would run:
    # 3. Mapping logic
    # (The following code represents the expected processing if the keys were present)

    severity_map = {  # noqa: F841
        "stress": "hairline",
        "hairline": "hairline",
        "transverse": "simple",
        "oblique": "simple",
        "spiral": "simple",
        "displaced": "displaced",
        "avulsion": "displaced",
        "comminuted": "comminuted",
        "segmental": "comminuted",
    }

    bone_map = {  # noqa: F841
        "wrist": "distal_radius",
        "radius": "distal_radius",
        "distal radius": "distal_radius",
        "clavicle": "clavicle",
        "collarbone": "clavicle",
        "ankle": "ankle",
        "fibula": "ankle",
        "tibia": "ankle",
        "femur": "femur",
        "thigh": "femur",
        "humerus": "humerus",
        "upper arm": "humerus",
        "foot": "metatarsal",
        "metatarsal": "metatarsal",
        "toe": "metatarsal",
    }

    # (Placeholder execution logic)
    print("Processing annotations...")


if __name__ == "__main__":
    main()
