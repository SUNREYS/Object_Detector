"""
=============================================================================
Step 1: CONVERT ANNOTATIONS AND PROCESS IMAGES
=============================================================================

Step 1: ANNOTATION CONVERSION AND IMAGE PROCESSING
    This script handles the core preprocessing pipeline:
    1.1  Load train/test frame lists from the imageSets directory
    1.2  Split training data into train/val sets
    1.3  For each frame:
         - Parse the XML annotation
         - Center-crop the image from 640x512 to 512x512
         - Adjust bounding boxes for the crop (clip partially visible,
           discard fully outside or below threshold)
         - Resize to TARGET_SIZE if different from 512
         - Convert bounding boxes to YOLO format (normalized x_center,
           y_center, width, height)
         - Save processed visible and thermal images
         - Save YOLO .txt label files
    1.4  Generate a YOLO dataset .yaml config file

    Input:  Raw KAIST dataset (640x512 images + XML annotations)
    Output: Cropped/resized 512x512 images + YOLO .txt labels organized
            in train/val/test splits.
=============================================================================
"""

import os
import sys
import random
from PIL import Image
from tqdm import tqdm

from config import (
    KAIST_ROOT, TRAIN_LIST_FILE, TEST_LIST_FILE, VAL_FRACTION,
    DATASET_FRACTION, RANDOM_SEED, CROP_LEFT, CROP_RIGHT, CROP_TOP, CROP_BOTTOM,
    CROPPED_WIDTH, CROPPED_HEIGHT, TARGET_SIZE,
    OUTPUT_ROOT, OUTPUT_IMAGES_VISIBLE, OUTPUT_IMAGES_THERMAL, OUTPUT_LABELS,
    CLASS_NAMES, NUM_CLASSES,
)
from utils import (
    parse_frame_id, get_image_path, get_annotation_path,
    parse_kaist_xml, adjust_bbox_for_crop, bbox_to_yolo, scale_bbox,
    load_image_set, make_output_dirs,
)


# =============================================================================
# Step 1.1: LOAD FRAME LISTS
#   Read both KAIST train and test lists and combine them into a single
#   pool. All frames are used for training the OD (the user has a separate
#   authentic thermal test set for evaluation).
# =============================================================================

def load_frame_lists():
    """
    Load and combine all KAIST frame lists into one pool,
    then sample down to DATASET_FRACTION of the total.

    Returns:
        List of frame identifier strings.
    """
    print("[Step 1.1] Loading frame lists from imageSets...")
    train_frames = load_image_set(TRAIN_LIST_FILE)
    test_frames = load_image_set(TEST_LIST_FILE)
    all_frames = train_frames + test_frames
    print(f"  KAIST train list: {len(train_frames)}")
    print(f"  KAIST test list:  {len(test_frames)}")
    print(f"  Combined total:   {len(all_frames)}")

    # Sample down if DATASET_FRACTION < 1.0
    if DATASET_FRACTION < 1.0:
        random.seed(RANDOM_SEED)
        keep_count = int(len(all_frames) * DATASET_FRACTION)
        all_frames = random.sample(all_frames, keep_count)
        print(f"  Using {DATASET_FRACTION*100:.0f}%: {len(all_frames)} frames")

    return all_frames


# =============================================================================
# Step 1.2: SPLIT TRAINING INTO TRAIN + VALIDATION
#   Hold out a fraction of the training data for validation.
#   We shuffle deterministically using RANDOM_SEED.
# =============================================================================

def split_train_val(train_frames):
    """
    Split training frames into train and validation subsets.

    Args:
        train_frames: List of frame identifiers.

    Returns:
        Tuple (train_subset, val_subset).
    """
    print("[Step 1.2] Splitting training data into train/val...")
    random.seed(RANDOM_SEED)
    shuffled = train_frames.copy()
    random.shuffle(shuffled)

    val_count = int(len(shuffled) * VAL_FRACTION)
    val_subset = shuffled[:val_count]
    train_subset = shuffled[val_count:]

    print(f"  Training subset: {len(train_subset)}")
    print(f"  Validation subset: {len(val_subset)}")
    return train_subset, val_subset


# =============================================================================
# Step 1.3: PROCESS A SINGLE FRAME
#   For one frame: parse XML, crop image, adjust bboxes, convert to YOLO,
#   save outputs. This is the core function called for every frame.
# =============================================================================

def process_frame(frame_id_str: str, split: str) -> dict:
    """
    Process a single KAIST frame: crop images and convert annotations.

    Args:
        frame_id_str: Frame identifier, e.g., "set00/V000/I00001"
        split: One of "train", "val", "test"

    Returns:
        Dict with processing metadata for this frame:
        {
            'frame_id': str,
            'split': str,
            'num_objects_original': int,
            'num_objects_after_crop': int,
            'objects': list of dicts with adjusted bbox info,
            'visible_path': str,
            'thermal_path': str,
            'label_path': str,
            'skipped': bool,
            'skip_reason': str or None,
        }
    """
    set_id, video_id, frame_name = parse_frame_id(frame_id_str)

    # Build a flat filename: set00_V000_I00001
    flat_name = f"{set_id}_{video_id}_{frame_name}"

    result = {
        "frame_id": frame_id_str,
        "flat_name": flat_name,
        "split": split,
        "set_id": set_id,
        "video_id": video_id,
        "frame_name": frame_name,
        "num_objects_original": 0,
        "num_objects_after_crop": 0,
        "objects": [],
        "visible_path": "",
        "thermal_path": "",
        "label_path": "",
        "skipped": False,
        "skip_reason": None,
    }

    # --- Check that source files exist ---
    vis_path = get_image_path(set_id, video_id, frame_name, "visible")
    therm_path = get_image_path(set_id, video_id, frame_name, "lwir")
    xml_path = get_annotation_path(set_id, video_id, frame_name)

    if not os.path.exists(vis_path):
        result["skipped"] = True
        result["skip_reason"] = f"Visible image not found: {vis_path}"
        return result

    if not os.path.exists(therm_path):
        result["skipped"] = True
        result["skip_reason"] = f"Thermal image not found: {therm_path}"
        return result

    # --- Parse XML annotation ---
    if os.path.exists(xml_path):
        img_info, objects = parse_kaist_xml(xml_path)
    else:
        img_info = {"width": 640, "height": 512, "filename": frame_id_str}
        objects = []

    result["num_objects_original"] = len(objects)

    # --- Crop and save visible image ---
    vis_img = Image.open(vis_path).convert("RGB")
    vis_cropped = vis_img.crop((CROP_LEFT, CROP_TOP, CROP_RIGHT, CROP_BOTTOM))
    if CROPPED_WIDTH != TARGET_SIZE or CROPPED_HEIGHT != TARGET_SIZE:
        vis_cropped = vis_cropped.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)

    out_vis_path = os.path.join(OUTPUT_IMAGES_VISIBLE, split, f"{flat_name}.jpg")
    vis_cropped.save(out_vis_path, quality=95)
    result["visible_path"] = out_vis_path

    # --- Crop and save thermal image ---
    therm_img = Image.open(therm_path).convert("RGB")
    therm_cropped = therm_img.crop((CROP_LEFT, CROP_TOP, CROP_RIGHT, CROP_BOTTOM))
    if CROPPED_WIDTH != TARGET_SIZE or CROPPED_HEIGHT != TARGET_SIZE:
        therm_cropped = therm_cropped.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)

    out_therm_path = os.path.join(OUTPUT_IMAGES_THERMAL, split, f"{flat_name}.jpg")
    therm_cropped.save(out_therm_path, quality=95)
    result["thermal_path"] = out_therm_path

    # --- Adjust bounding boxes and convert to YOLO ---
    yolo_lines = []
    adjusted_objects = []

    for obj in objects:
        adjusted = adjust_bbox_for_crop(obj["x"], obj["y"], obj["w"], obj["h"])
        if adjusted is None:
            continue

        ax, ay, aw, ah = adjusted

        # Scale if we need to resize (512->TARGET_SIZE)
        if CROPPED_WIDTH != TARGET_SIZE or CROPPED_HEIGHT != TARGET_SIZE:
            ax, ay, aw, ah = scale_bbox(
                ax, ay, aw, ah,
                CROPPED_WIDTH, CROPPED_HEIGHT,
                TARGET_SIZE, TARGET_SIZE
            )

        yolo_str = bbox_to_yolo(ax, ay, aw, ah, TARGET_SIZE, TARGET_SIZE, obj["name"])
        if yolo_str is not None:
            yolo_lines.append(yolo_str)

        adjusted_objects.append({
            "name": obj["name"],
            "x": ax,
            "y": ay,
            "w": aw,
            "h": ah,
            "truncated": obj["truncated"],
            "difficult": obj["difficult"],
            "occlusion": obj["occlusion"],
            "original_x": obj["x"],
            "original_y": obj["y"],
            "original_w": obj["w"],
            "original_h": obj["h"],
        })

    result["num_objects_after_crop"] = len(yolo_lines)
    result["objects"] = adjusted_objects

    # --- Save YOLO label file (always, even if empty) ---
    out_label_path = os.path.join(OUTPUT_LABELS, split, f"{flat_name}.txt")
    with open(out_label_path, "w") as f:
        f.write("\n".join(yolo_lines))
    result["label_path"] = out_label_path

    return result


# =============================================================================
# Step 1.4: PROCESS ALL FRAMES
#   Iterate over all frames in each split (train/val) and call
#   process_frame(). Collect all metadata for downstream analysis.
# =============================================================================

def process_all_frames(train_frames, val_frames):
    """
    Process all frames across train and val splits.

    Args:
        train_frames, val_frames: Lists of frame identifiers.

    Returns:
        List of result dicts from process_frame().
    """
    all_results = []

    splits = [
        ("train", train_frames),
        ("val", val_frames),
    ]

    for split_name, frames in splits:
        print(f"\n[Step 1.4] Processing {split_name} split ({len(frames)} frames)...")
        for frame_id in tqdm(frames, desc=f"  {split_name}", unit="frame"):
            result = process_frame(frame_id, split_name)
            all_results.append(result)

    return all_results


# =============================================================================
# Step 1.5: GENERATE YOLO DATASET CONFIG
#   Create the .yaml file needed by YOLOv5/v8 for training.
# =============================================================================

def generate_yolo_yaml(modality: str = "visible"):
    """
    Generate a YOLO dataset configuration .yaml file.

    Args:
        modality: "visible" or "thermal" — determines image path.
    """
    if modality == "visible":
        img_dir = OUTPUT_IMAGES_VISIBLE
    else:
        img_dir = OUTPUT_IMAGES_THERMAL

    yaml_path = os.path.join(OUTPUT_ROOT, f"kaist_{modality}.yaml")
    yaml_content = f"""# KAIST Pedestrian Detection Dataset - {modality.capitalize()} images
# Auto-generated by preprocessing pipeline

path: {OUTPUT_ROOT}
train: {os.path.join(img_dir, 'train')}
val: {os.path.join(img_dir, 'val')}

# Class configuration
nc: {NUM_CLASSES}
names: {CLASS_NAMES}
"""

    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"[Step 1.5] YOLO config written to: {yaml_path}")


# =============================================================================
# Step 1.6: MAIN ENTRY POINT
# =============================================================================

def run_step1():
    """
    Execute the full Step 1 pipeline: load all frames, split into
    train/val, crop images, convert annotations.

    Returns:
        List of per-frame result dicts.
    """
    print("=" * 70)
    print("STEP 1: ANNOTATION CONVERSION AND IMAGE PROCESSING")
    print("=" * 70)

    # Create output directories
    make_output_dirs()

    # Load all frame lists (combine KAIST train + test into one pool)
    all_frames = load_frame_lists()

    # Split into train + val
    train_subset, val_subset = split_train_val(all_frames)

    # Process all frames
    all_results = process_all_frames(train_subset, val_subset)

    # Generate YOLO configs for both modalities
    generate_yolo_yaml("visible")
    generate_yolo_yaml("thermal")

    # Summary
    total = len(all_results)
    skipped = sum(1 for r in all_results if r["skipped"])
    with_objects = sum(1 for r in all_results if r["num_objects_after_crop"] > 0)
    total_objects = sum(r["num_objects_after_crop"] for r in all_results)

    print(f"\n[Step 1] Summary:")
    print(f"  Total frames processed: {total}")
    print(f"  Skipped frames:         {skipped}")
    print(f"  Frames with objects:    {with_objects}")
    print(f"  Total objects (after crop): {total_objects}")

    return all_results


if __name__ == "__main__":
    results = run_step1()
