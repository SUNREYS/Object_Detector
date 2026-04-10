"""
=============================================================================
Step 2: FEATURE EXTRACTION FOR ANALYSIS
=============================================================================

Step 2: EXTRACT PER-IMAGE AND PER-OBJECT FEATURES
    This script extracts features from each processed frame that will be
    used later to analyze why the object detector fails on certain images.
    The goal is to correlate detection performance with image/object
    properties so we can determine if a translation model (thermal
    synthesis) hurts specific subgroups.

    Features extracted per IMAGE:
    2.1  Number of people in the frame (crowd density)
    2.2  Number of overlapping annotation pairs (IoU > threshold)
    2.3  Mean / max / min bounding box height (proxy for distance)
    2.4  Image brightness (mean pixel intensity — proxy for lighting)
    2.5  Image contrast (std of pixel intensity)
    2.6  Day vs. night label (from KAIST set mapping)
    2.7  Edge density (Sobel magnitude — scene complexity)

    Features extracted per OBJECT (bounding box):
    2.8  Bounding box height in pixels (distance proxy)
    2.9  Bounding box area in pixels
    2.10 Size category: small / medium / large
    2.11 Aspect ratio (width / height)
    2.12 Occlusion level (from annotation)
    2.13 Truncation flag (from annotation)
    2.14 Relative position in image (center vs edge)
    2.15 Has overlapping neighbor (IoU > threshold with any other bbox)

    Input:  Per-frame result dicts from Step 1.
    Output: Enriched metadata dicts ready for CSV export.
=============================================================================
"""

import os
import numpy as np
from PIL import Image
from typing import List, Dict

from config import (
    TARGET_SIZE, DAY_SETS, NIGHT_SETS,
    SIZE_SMALL_MAX, SIZE_MEDIUM_MAX,
    OVERLAP_IOU_THRESHOLD,
    OUTPUT_IMAGES_VISIBLE,
    MODALITY_IMAGE_DIRS,
)
from utils import compute_iou


# =============================================================================
# Step 2.1: COUNT PEOPLE IN FRAME
#   Simple count of how many bounding boxes remain after cropping.
#   High counts indicate crowded scenes where detection is harder.
# =============================================================================

def count_people(objects: List[Dict]) -> int:
    """
    Count the number of person annotations in a frame.

    Args:
        objects: List of adjusted object dicts from Step 1.

    Returns:
        Number of people detected/annotated in this frame.
    """
    return len(objects)


# =============================================================================
# Step 2.2: COUNT OVERLAPPING ANNOTATION PAIRS
#   Find pairs of bounding boxes that overlap significantly. High overlap
#   makes detection harder since NMS may suppress valid detections.
# =============================================================================

def count_overlapping_pairs(objects: List[Dict]) -> int:
    """
    Count the number of pairs of bounding boxes with IoU above threshold.

    Args:
        objects: List of adjusted object dicts with 'x', 'y', 'w', 'h'.

    Returns:
        Number of overlapping pairs.
    """
    n = len(objects)
    overlap_count = 0

    for i in range(n):
        for j in range(i + 1, n):
            box_i = (objects[i]["x"], objects[i]["y"],
                     objects[i]["w"], objects[i]["h"])
            box_j = (objects[j]["x"], objects[j]["y"],
                     objects[j]["w"], objects[j]["h"])
            if compute_iou(box_i, box_j) > OVERLAP_IOU_THRESHOLD:
                overlap_count += 1

    return overlap_count


# =============================================================================
# Step 2.3: BOUNDING BOX HEIGHT STATISTICS
#   Bounding box height is a strong proxy for how far away a person is.
#   Small boxes = far away = harder to detect.
# =============================================================================

def bbox_height_stats(objects: List[Dict]) -> Dict:
    """
    Compute statistics on bounding box heights in a frame.

    Args:
        objects: List of adjusted object dicts.

    Returns:
        Dict with 'mean_height', 'max_height', 'min_height', 'std_height'.
        Returns zeros if no objects.
    """
    if not objects:
        return {"mean_height": 0, "max_height": 0, "min_height": 0, "std_height": 0}

    heights = [obj["h"] for obj in objects]
    return {
        "mean_height": float(np.mean(heights)),
        "max_height": float(np.max(heights)),
        "min_height": float(np.min(heights)),
        "std_height": float(np.std(heights)),
    }


# =============================================================================
# Step 2.4: IMAGE BRIGHTNESS
#   Mean pixel intensity across the grayscale image. Low brightness
#   indicates night/dark scenes. Used alongside the day/night label
#   for a continuous measure of scene illumination.
# =============================================================================

def compute_brightness(image_path: str) -> float:
    """
    Compute mean pixel intensity of a grayscale version of the image.

    Args:
        image_path: Path to the processed image file.

    Returns:
        Mean brightness value in [0, 255].
    """
    img = Image.open(image_path).convert("L")
    pixels = np.array(img, dtype=np.float32)
    return float(np.mean(pixels))


# =============================================================================
# Step 2.5: IMAGE CONTRAST
#   Standard deviation of pixel intensities. Low contrast makes it
#   harder to distinguish pedestrians from the background.
# =============================================================================

def compute_contrast(image_path: str) -> float:
    """
    Compute the standard deviation of pixel intensities (contrast measure).

    Args:
        image_path: Path to the processed image file.

    Returns:
        Contrast value (std of pixel intensities).
    """
    img = Image.open(image_path).convert("L")
    pixels = np.array(img, dtype=np.float32)
    return float(np.std(pixels))


# =============================================================================
# Step 2.6: DAY / NIGHT CLASSIFICATION (from set ID)
#   The KAIST dataset groups day and night sequences into different sets.
#   This provides a binary label based on the known mapping.
# =============================================================================

def classify_day_night_by_set(set_id: str) -> str:
    """
    Classify a frame as day or night based on its KAIST set ID.

    Args:
        set_id: e.g., "set00"

    Returns:
        "day" or "night"
    """
    if set_id in DAY_SETS:
        return "day"
    elif set_id in NIGHT_SETS:
        return "night"
    else:
        return "unknown"


# =============================================================================
# Step 2.7: EDGE DENSITY
#   Measure scene complexity using the mean magnitude of Sobel edges.
#   Complex backgrounds (many edges) can confuse detectors.
# =============================================================================

def compute_edge_density(image_path: str) -> float:
    """
    Compute edge density using Sobel filter magnitude.
    Uses numpy-based convolution to avoid dependency on cv2 for this step.

    Args:
        image_path: Path to the processed image file.

    Returns:
        Mean edge magnitude value.
    """
    img = Image.open(image_path).convert("L")
    pixels = np.array(img, dtype=np.float64)

    # Sobel kernels
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)

    # Pad the image
    padded = np.pad(pixels, 1, mode="edge")
    h, w = pixels.shape
    gx = np.zeros_like(pixels)
    gy = np.zeros_like(pixels)

    # Simple convolution (avoiding scipy dependency)
    for i in range(h):
        for j in range(w):
            patch = padded[i:i+3, j:j+3]
            gx[i, j] = np.sum(patch * sobel_x)
            gy[i, j] = np.sum(patch * sobel_y)

    magnitude = np.sqrt(gx**2 + gy**2)
    return float(np.mean(magnitude))


def compute_edge_density_fast(image_path: str) -> float:
    """
    Compute edge density using a vectorized Sobel approximation.
    Much faster than the pixel-by-pixel version above.

    Args:
        image_path: Path to the processed image file.

    Returns:
        Mean edge magnitude value.
    """
    img = Image.open(image_path).convert("L")
    pixels = np.array(img, dtype=np.float64)

    # Approximate Sobel using shifted arrays (avoids explicit convolution)
    # Gx ≈ right column - left column (simplified horizontal gradient)
    # Gy ≈ bottom row - top row (simplified vertical gradient)
    gx = np.zeros_like(pixels)
    gy = np.zeros_like(pixels)

    # Horizontal gradient (Sobel-like)
    gx[:, 1:-1] = (
        -pixels[:, :-2] + pixels[:, 2:]
        - 2 * pixels[:, :-2] + 2 * pixels[:, 2:]
    ) / 4.0

    # Vertical gradient (Sobel-like)
    gy[1:-1, :] = (
        -pixels[:-2, :] + pixels[2:, :]
        - 2 * pixels[:-2, :] + 2 * pixels[2:, :]
    ) / 4.0

    magnitude = np.sqrt(gx**2 + gy**2)
    return float(np.mean(magnitude))


# =============================================================================
# Step 2.8 - 2.15: PER-OBJECT FEATURE EXTRACTION
#   Extract features for each individual bounding box annotation.
#
#   Occlusion (from KAIST XML):
#     0 = Not occluded — the person is fully visible
#     1 = Partially occluded — part of the person is hidden behind
#         another object (car, pole, another person, etc.)
#     2 = Heavily occluded — most of the person is hidden; only a
#         small portion is visible. Hardest to detect.
#
#   Truncated (from KAIST XML):
#     0 = Not truncated — the person is fully within the image frame
#     1 = Truncated — the person extends beyond the image boundary
#         (cut off by the edge of the frame). After our center crop
#         from 640x512 to 512x512, more people near the left/right
#         edges may become truncated.
# =============================================================================

def extract_object_features(obj: Dict, all_objects: List[Dict]) -> Dict:
    """
    Extract analysis features for a single object annotation.

    Args:
        obj: Dict with adjusted bbox info ('x', 'y', 'w', 'h', 'occlusion', etc.)
        all_objects: All objects in the same frame (for overlap computation).

    Returns:
        Dict of extracted features for this object.
    """
    h = obj["h"]
    w = obj["w"]
    area = w * h

    # Step 2.10: Size category based on height
    if h <= SIZE_SMALL_MAX:
        size_category = "small"
    elif h <= SIZE_MEDIUM_MAX:
        size_category = "medium"
    else:
        size_category = "large"

    # Step 2.11: Aspect ratio
    aspect_ratio = w / h if h > 0 else 0.0

    # Step 2.14: Relative position in image (normalized center coords)
    #The Euclidean distance from the image center. It doesn't encode direction (left vs right, top vs bottom) but gives a single value for how close to the center the object is.
    cx = (obj["x"] + w / 2.0) / TARGET_SIZE
    cy = (obj["y"] + h / 2.0) / TARGET_SIZE

    # Distance from image center (0 = center, ~0.707 = corner)
    dist_from_center = np.sqrt((cx - 0.5)**2 + (cy - 0.5)**2)

    # Step 2.15: Check if this object overlaps with any other
    box_self = (obj["x"], obj["y"], obj["w"], obj["h"])
    has_overlap = False
    max_iou_with_neighbor = 0.0

    for other in all_objects:
        if other is obj:
            continue
        box_other = (other["x"], other["y"], other["w"], other["h"])
        iou = compute_iou(box_self, box_other)
        max_iou_with_neighbor = max(max_iou_with_neighbor, iou)
        if iou > OVERLAP_IOU_THRESHOLD:
            has_overlap = True

    return {
        "bbox_height_px": h,
        "bbox_width_px": w,
        "bbox_area_px": area,
        "size_category": size_category,
        "aspect_ratio": round(aspect_ratio, 4),
        "occlusion": obj.get("occlusion", 0),
        "truncated": obj.get("truncated", 0),
        "center_x_norm": round(cx, 4),
        "center_y_norm": round(cy, 4),
        "dist_from_center": round(dist_from_center, 4),
        "has_overlap": has_overlap,
        "max_iou_neighbor": round(max_iou_with_neighbor, 4),
        "original_class": obj.get("name", "person"),
    }


# =============================================================================
# Step 2.16: EXTRACT ALL FEATURES FOR A FRAME
#   Combine image-level and object-level features into a single record.
# =============================================================================

def extract_frame_features(frame_result: Dict, img_dir: str = None) -> Dict:
    """
    Extract all analysis features for a processed frame.

    Args:
        img_dir: If provided, pixel-level features (brightness, contrast,
                 edge_density) are computed from images in this directory.
                 Falls back to ``visible_path`` when None.

    Args:
        frame_result: Dict output from step1 process_frame().

    Returns:
        Dict with all image-level and object-level features.
    """
    objects = frame_result.get("objects", [])
    set_id = frame_result.get("set_id", "")

    # Image-level features — use the caller-specified modality directory,
    # or fall back to the visible image path stored by Step 1.
    if img_dir is not None:
        split = frame_result.get("split", "")
        fname = frame_result.get("flat_name", "") + ".jpg"
        # Handle both flat layout (kaist: split/name.jpg)
        # and images-subdir layout (pid: split/images/name.jpg)
        direct_path = os.path.join(img_dir, split, fname)
        images_subdir_path = os.path.join(img_dir, split, "images", fname)
        if os.path.exists(direct_path):
            vis_path = direct_path
        elif os.path.exists(images_subdir_path):
            vis_path = images_subdir_path
        else:
            vis_path = direct_path  # fallback (will gracefully fail later)
    else:
        vis_path = frame_result.get("visible_path", "")

    features = {
        "frame_id": frame_result["frame_id"],
        "flat_name": frame_result["flat_name"],
        "split": frame_result["split"],
        "set_id": set_id,
        "video_id": frame_result.get("video_id", ""),
        "day_night": classify_day_night_by_set(set_id),
        "num_people": count_people(objects),
        "num_overlapping_pairs": count_overlapping_pairs(objects),
        "skipped": frame_result.get("skipped", False),
    }

    # Image-level metrics (only if image exists)
    if vis_path and os.path.exists(vis_path):
        features["brightness"] = round(compute_brightness(vis_path), 2)
        features["contrast"] = round(compute_contrast(vis_path), 2)
        features["edge_density"] = round(compute_edge_density_fast(vis_path), 2)
    else:
        features["brightness"] = 0.0
        features["contrast"] = 0.0
        features["edge_density"] = 0.0

    # Per-object features
    object_features = []
    for obj in objects:
        obj_feat = extract_object_features(obj, objects)
        obj_feat["frame_id"] = frame_result["frame_id"]
        obj_feat["flat_name"] = frame_result["flat_name"]
        obj_feat["split"] = frame_result["split"]
        obj_feat["day_night"] = features["day_night"]
        object_features.append(obj_feat)

    features["object_features"] = object_features

    return features


# =============================================================================
# Step 2.17: PROCESS ALL FRAMES
#   Run feature extraction on all processed frames.
# =============================================================================

def run_step2(frame_results: List[Dict], modality: str = "visible",
              split_filter: str = "val") -> List[Dict]:
    """
    Execute Step 2: extract features for all frames.

    Args:
        frame_results:  List of result dicts from Step 1.
        modality:       Which modality's images to use for pixel-level features
                        (brightness, contrast, edge_density).  Must be a key in
                        ``MODALITY_IMAGE_DIRS``.  Defaults to ``"visible"``.
        split_filter:   Only process frames belonging to this split
                        (``"val"``, ``"train"``, or ``None`` for all).
                        Defaults to ``"val"`` because evaluation only uses
                        the validation set.

    Returns:
        List of feature dicts, one per frame.
    """
    from tqdm import tqdm

    print("\n" + "=" * 70)
    print("STEP 2: FEATURE EXTRACTION FOR ANALYSIS")
    print("=" * 70)

    img_dir = MODALITY_IMAGE_DIRS.get(modality)
    if img_dir is None:
        print(f"  WARNING: Unknown modality '{modality}', falling back to visible images.")
        img_dir = MODALITY_IMAGE_DIRS["visible"]
    print(f"[Step 2] Using images from: {img_dir}")
    if split_filter:
        print(f"[Step 2] Processing split: '{split_filter}' only")

    all_features = []
    valid_results = [
        r for r in frame_results
        if not r.get("skipped", False)
        and (split_filter is None or r.get("split") == split_filter)
    ]

    print(f"[Step 2] Extracting features from {len(valid_results)} frames...")
    for result in tqdm(valid_results, desc="  Extracting features", unit="frame"):
        features = extract_frame_features(result, img_dir=img_dir)
        all_features.append(features)

    # Summary
    total_objects = sum(f["num_people"] for f in all_features)
    frames_with_people = sum(1 for f in all_features if f["num_people"] > 0)
    day_frames = sum(1 for f in all_features if f["day_night"] == "day")
    night_frames = sum(1 for f in all_features if f["day_night"] == "night")

    print(f"\n[Step 2] Summary:")
    print(f"  Frames with features: {len(all_features)}")
    print(f"  Frames with people:   {frames_with_people}")
    print(f"  Total objects:        {total_objects}")
    print(f"  Day frames:           {day_frames}")
    print(f"  Night frames:         {night_frames}")

    return all_features


if __name__ == "__main__":
    print("Step 2 must be run after Step 1. Use main.py to run the full pipeline.")
