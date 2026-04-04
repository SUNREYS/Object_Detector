"""
create_val_new.py
-----------------
Randomly samples ~10% of total dataset frames (~5,921) from the existing
val split and creates a consistent val_new/ folder in every modality
(and labels/) using hard links so no extra disk space is used.

A txt file listing the chosen stems is saved to:
  datasets/kaist_processed/val_new_list.txt

Run from workspace root:
  python preprocessing/create_val_new.py
"""

import os
import random
import math
import shutil
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
PROCESSED_ROOT = Path(
    r"C:\Users\poopi\Documents\programming projectzx\exjobb\datasets\kaist_processed"
)
IMAGES_ROOT = PROCESSED_ROOT / "images"
LABELS_VAL  = PROCESSED_ROOT / "labels" / "val"
LABELS_OUT  = PROCESSED_ROOT / "labels" / "val_new"

# Total dataset size ~ 59,212 → 10% = 5,921
TOTAL_FRAMES   = 59212
TARGET_PERCENT = 0.10
TARGET_COUNT   = math.ceil(TOTAL_FRAMES * TARGET_PERCENT)   # 5922

SEED = 42
LIST_PATH = PROCESSED_ROOT / "val_new_list.txt"
# ──────────────────────────────────────────────────────────────────────────────


def hard_link_or_copy(src: Path, dst: Path):
    """Hard-link src→dst; fall back to copy if cross-device."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    random.seed(SEED)

    # 1. Collect all stems from visible/val (canonical source)
    vis_val = IMAGES_ROOT / "visible" / "val"
    all_stems = sorted(p.stem for p in vis_val.glob("*.jpg"))
    print(f"Current visible/val: {len(all_stems)} frames")

    # 2. Sample
    sampled = sorted(random.sample(all_stems, min(TARGET_COUNT, len(all_stems))))
    sampled_set = set(sampled)
    print(f"Sampled {len(sampled)} frames (~{len(sampled)/TOTAL_FRAMES*100:.1f}% of total)")

    # 3. Save list
    LIST_PATH.write_text("\n".join(sampled), encoding="utf-8")
    print(f"Saved stem list -> {LIST_PATH}")

    # 4. For each modality create val_new/
    modalities = [d for d in IMAGES_ROOT.iterdir() if d.is_dir()]
    for mod_dir in sorted(modalities):
        val_dir = mod_dir / "val"
        if not val_dir.exists():
            print(f"  [SKIP] {mod_dir.name}/val not found")
            continue
        out_dir = mod_dir / "val_new"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Pre-index: stem → list of Path objects (handles .jpg + .npy, etc.)
        index: dict[str, list[Path]] = {}
        for f in val_dir.iterdir():
            if f.is_file():
                index.setdefault(f.stem, []).append(f)

        linked = 0
        missing = 0
        for stem in sampled:
            files = index.get(stem)
            if not files:
                missing += 1
                continue
            for src in files:
                hard_link_or_copy(src, out_dir / src.name)
                linked += 1

        print(f"  {mod_dir.name}/val_new: {linked} files linked  (missing stems: {missing})")

    # 5. Labels val_new/
    if LABELS_VAL.exists():
        LABELS_OUT.mkdir(parents=True, exist_ok=True)
        linked = 0
        missing = 0
        for stem in sampled:
            src = LABELS_VAL / f"{stem}.txt"
            if not src.exists():
                # Frame with no objects → create empty label file
                (LABELS_OUT / f"{stem}.txt").touch()
                linked += 1
                continue
            hard_link_or_copy(src, LABELS_OUT / src.name)
            linked += 1
        print(f"  labels/val_new: {linked} files linked  (missing: {missing})")
    else:
        print(f"  [SKIP] labels/val not found at {LABELS_VAL}")

    print(f"\nDone. val_new contains {len(sampled)} frames per modality.")
    print("To use val_new in training, update the dataset YAML to point to val_new/.")


if __name__ == "__main__":
    main()
