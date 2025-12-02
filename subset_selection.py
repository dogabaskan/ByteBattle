import os
import random
import shutil
from pathlib import Path

# ---------------------------------------------
# CONFIGURATION
# ---------------------------------------------
SOURCE_DIR = Path("data/static/images")   # your full dataset folder
TARGET_DIR = Path("data/static/subset")   # output subset folder
IMAGES_PER_CLASS = 800                    # choose how many images you want
EXTENSIONS = (".jpg", ".jpeg", ".png")    # allowed image types
# ---------------------------------------------

def make_subset():
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    # list all gesture class folders
    class_folders = [d for d in SOURCE_DIR.iterdir() if d.is_dir()]

    print(f"Found {len(class_folders)} gesture classes.")
    print("Creating subset...")

    for class_dir in class_folders:
        class_name = class_dir.name
        print(f"\nProcessing class: {class_name}")

        # create target folder
        out_dir = TARGET_DIR / class_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # collect image paths
        all_images = [
            p for p in class_dir.iterdir()
            if p.suffix.lower() in EXTENSIONS
        ]

        print(f"  Total images: {len(all_images)}")

        # randomly choose a subset
        selected = random.sample(all_images, min(IMAGES_PER_CLASS, len(all_images)))
        print(f"  Selected: {len(selected)}")

        # copy files
        for img_path in selected:
            shutil.copy(img_path, out_dir / img_path.name)

        print(f"  → Saved to: {out_dir}")

    print("\nSubset creation complete!")
    print(f"Subset stored in: {TARGET_DIR}")


if __name__ == "__main__":
    make_subset()
