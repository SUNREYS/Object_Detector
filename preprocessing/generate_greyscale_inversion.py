"""
=============================================================================
Generate Greyscale-Inverted Images from Thermal
=============================================================================

Takes the processed thermal images (512x512 JPEG) and creates inverted
copies: pixel_value -> 255 - pixel_value.

This makes hot objects (bright in thermal) appear dark and cold backgrounds
appear bright, producing images structurally closer to visible-light imagery.

Annotations are shared across modalities, so only images need to be created.

Usage:
    python preprocessing/generate_greyscale_inversion.py
    train: docker compose up train-greyscale-inversion
=============================================================================
"""

import os
import sys
from PIL import Image, ImageOps
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths — relative to project root (exjobb/)
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATASET_ROOT = os.path.join(PROJECT_ROOT, "datasets", "kaist_processed")

SRC_MODALITY = "thermal"
DST_MODALITY = "greyscale_inversion"
SPLITS = ["train", "val"]


def generate():
    total = 0
    for split in SPLITS:
        src_dir = os.path.join(DATASET_ROOT, "images", SRC_MODALITY, split)
        dst_dir = os.path.join(DATASET_ROOT, "images", DST_MODALITY, split)

        if not os.path.isdir(src_dir):
            print(f"WARNING: Source directory not found: {src_dir}")
            continue

        os.makedirs(dst_dir, exist_ok=True)

        files = sorted(f for f in os.listdir(src_dir) if f.endswith(".jpg"))
        print(f"\n[{split}] Inverting {len(files)} images...")

        for fname in tqdm(files, desc=split):
            src_path = os.path.join(src_dir, fname)
            dst_path = os.path.join(dst_dir, fname)

            img = Image.open(src_path).convert("L")
            inverted = ImageOps.invert(img)
            inverted = ImageOps.autocontrast(inverted)
            inverted.save(dst_path, quality=95)
            total += 1

    print(f"\nDone. {total} inverted images saved to:")
    print(f"  {os.path.join(DATASET_ROOT, 'images', DST_MODALITY)}")


if __name__ == "__main__":
    generate()
