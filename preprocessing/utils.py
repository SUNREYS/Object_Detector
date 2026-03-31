"""
=============================================================================
Utility Module for KAIST Dataset Preprocessing Pipeline
=============================================================================

Step 0.5: UTILITIES
    Shared helper functions used across multiple preprocessing scripts.
    Includes XML parsing, path construction, bounding box math, and I/O.
=============================================================================
"""

import os
import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple, Optional

from config import (
    KAIST_IMAGES, KAIST_ANNOTATIONS, KAIST_IMAGE_SETS,
    HUMAN_CLASSES, CLASS_MAP,
    CROP_LEFT, CROP_RIGHT, CROP_TOP, CROP_BOTTOM,
    CROPPED_WIDTH, CROPPED_HEIGHT, TARGET_SIZE,
    MIN_BBOX_AREA_PX, MIN_VISIBLE_FRACTION,
)


# =============================================================================
# Step 0.5.1: XML ANNOTATION PARSING
#   Parse a KAIST XML annotation file and extract all object bounding boxes.
#   Returns a list of dicts with keys: name, x, y, w, h, truncated,
#   difficult, occlusion.
# =============================================================================

def parse_kaist_xml(xml_path: str) -> Tuple[Dict, List[Dict]]:
    """
    Parse a KAIST annotation XML file.

    Args:
        xml_path: Path to the XML annotation file.

    Returns:
        Tuple of (image_info, objects) where:
            image_info: dict with keys 'width', 'height', 'filename'
            objects: list of dicts, each with keys:
                'name', 'x', 'y', 'w', 'h', 'truncated', 'difficult', 'occlusion'
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Extract image metadata
    size = root.find("size")
    image_info = {
        "filename": root.findtext("filename", ""),
        "width": int(size.findtext("width", "640")),
        "height": int(size.findtext("height", "512")),
    }

    # Extract all object annotations
    objects = []
    for obj in root.findall("object"):
        name = obj.findtext("name", "").strip().lower()

        # Only keep human-related classes
        if name not in HUMAN_CLASSES:
            continue

        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue

        obj_dict = {
            "name": name,
            "x": int(bndbox.findtext("x", "0")),
            "y": int(bndbox.findtext("y", "0")),
            "w": int(bndbox.findtext("w", "0")),
            "h": int(bndbox.findtext("h", "0")),
            "truncated": int(obj.findtext("truncated", "0")),
            "difficult": int(obj.findtext("difficult", "0")),
            "occlusion": int(obj.findtext("occlusion", "0")),
        }
        objects.append(obj_dict)

    return image_info, objects


# =============================================================================
# Step 0.5.2: BOUNDING BOX CROPPING AND ADJUSTMENT
#   When we center-crop the 640x512 image to 512x512, we need to adjust
#   bounding boxes. Boxes partially outside the crop region are clipped.
#   Boxes fully outside or with too little remaining area are discarded.
#   The KAIST format uses x,y as top-left corner with w,h.
# =============================================================================

def adjust_bbox_for_crop(
    x: int, y: int, w: int, h: int
) -> Optional[Tuple[int, int, int, int]]:
    """
    Adjust a bounding box (top-left x, y, width, height) for center crop.

    The crop removes CROP_LEFT pixels from the left side. The bounding box
    is clipped to the crop region. Returns None if the box is fully outside
    or the remaining area is too small.

    Args:
        x, y, w, h: Original bounding box in pixel coordinates.

    Returns:
        Tuple (new_x, new_y, new_w, new_h) in cropped image coordinates,
        or None if the box should be discarded.
    """
    # Original box edges in full image coordinates
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h

    original_area = w * h
    if original_area <= 0:
        return None

    # Clip to crop region
    cx1 = max(x1, CROP_LEFT)
    cy1 = max(y1, CROP_TOP)
    cx2 = min(x2, CROP_RIGHT)
    cy2 = min(y2, CROP_BOTTOM)

    # Check if anything remains
    if cx1 >= cx2 or cy1 >= cy2:
        return None

    # New dimensions after clipping
    new_w = cx2 - cx1
    new_h = cy2 - cy1
    new_area = new_w * new_h

    # Check minimum visible fraction
    if new_area / original_area < MIN_VISIBLE_FRACTION:
        return None

    # Check minimum absolute area
    if new_area < MIN_BBOX_AREA_PX:
        return None

    # Shift coordinates to cropped image space (subtract CROP_LEFT from x)
    new_x = cx1 - CROP_LEFT
    new_y = cy1 - CROP_TOP

    return (new_x, new_y, new_w, new_h)


# =============================================================================
# Step 0.5.3: CONVERT TO YOLO FORMAT
#   YOLO format uses normalized center coordinates:
#   class_id  x_center  y_center  width  height
#   All values are normalized to [0, 1] relative to image dimensions.
# =============================================================================

def bbox_to_yolo(
    x: int, y: int, w: int, h: int,
    img_w: int, img_h: int,
    class_name: str
) -> Optional[str]:
    """
    Convert a bounding box to YOLO format string.

    Args:
        x, y, w, h: Bounding box in pixel coordinates (top-left + size),
                     already adjusted for crop.
        img_w, img_h: Image dimensions after crop.
        class_name: Object class name (e.g., 'person').

    Returns:
        YOLO format string: "class_id x_center y_center width height"
        or None if class is not recognized.
    """
    if class_name not in CLASS_MAP:
        return None

    class_id = CLASS_MAP[class_name]

    x_center = (x + w / 2.0) / img_w
    y_center = (y + h / 2.0) / img_h
    norm_w = w / img_w
    norm_h = h / img_h

    # Clamp to [0, 1] for safety
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    norm_w = max(0.0, min(1.0, norm_w))
    norm_h = max(0.0, min(1.0, norm_h))

    return f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}"


# =============================================================================
# Step 0.5.4: RESIZE BOUNDING BOX
#   If the cropped image (512x512) needs to be resized to a different
#   TARGET_SIZE, scale the bounding boxes accordingly. Since YOLO format
#   is normalized, this is only needed for pixel-coordinate operations.
# =============================================================================

def scale_bbox(
    x: int, y: int, w: int, h: int,
    src_w: int, src_h: int,
    dst_w: int, dst_h: int
) -> Tuple[int, int, int, int]:
    """
    Scale a bounding box from source dimensions to destination dimensions.

    Args:
        x, y, w, h: Bounding box in source pixel coordinates.
        src_w, src_h: Source image dimensions.
        dst_w, dst_h: Destination image dimensions.

    Returns:
        Tuple (new_x, new_y, new_w, new_h) in destination coords.
    """
    scale_x = dst_w / src_w
    scale_y = dst_h / src_h

    return (
        int(round(x * scale_x)),
        int(round(y * scale_y)),
        int(round(w * scale_x)),
        int(round(h * scale_y)),
    )


# =============================================================================
# Step 0.5.5: PATH CONSTRUCTION HELPERS
#   Build file paths for images, annotations, and output files using
#   the KAIST naming convention: set##/V###/I#####
# =============================================================================

def get_image_path(set_id: str, video_id: str, frame_id: str,
                   modality: str = "visible") -> str:
    """
    Construct the full path to a KAIST image file.

    Args:
        set_id: e.g., "set00"
        video_id: e.g., "V000"
        frame_id: e.g., "I00001"
        modality: "visible" or "lwir"

    Returns:
        Full file path string.
    """
    return os.path.join(KAIST_IMAGES, set_id, video_id, modality, f"{frame_id}.jpg")


def get_annotation_path(set_id: str, video_id: str, frame_id: str) -> str:
    """
    Construct the full path to a KAIST XML annotation file.
    """
    return os.path.join(KAIST_ANNOTATIONS, set_id, video_id, f"{frame_id}.xml")


def parse_frame_id(frame_str: str) -> Tuple[str, str, str]:
    """
    Parse a KAIST frame identifier string into components.

    Args:
        frame_str: e.g., "set00/V000/I00001"

    Returns:
        Tuple (set_id, video_id, frame_id), e.g., ("set00", "V000", "I00001")
    """
    parts = frame_str.strip().split("/")
    return parts[0], parts[1], parts[2]


# =============================================================================
# Step 0.5.6: LOAD IMAGE SET LISTS
#   Read the imageSets/*.txt files to get the list of frames for
#   train/test/day/night splits.
# =============================================================================

def load_image_set(filename: str) -> List[str]:
    """
    Load a list of frame identifiers from an imageSets text file.

    Args:
        filename: Name of the file in the imageSets directory.

    Returns:
        List of frame identifier strings, e.g., ["set00/V000/I00001", ...]
    """
    filepath = os.path.join(KAIST_IMAGE_SETS, filename)
    with open(filepath, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines


# =============================================================================
# Step 0.5.7: IoU COMPUTATION
#   Intersection over Union for detecting overlapping bounding boxes.
#   Used in the feature extraction step to find crowded scenes.
# =============================================================================

def compute_iou(box1: Tuple[int, int, int, int],
                box2: Tuple[int, int, int, int]) -> float:
    """
    Compute IoU between two boxes in (x, y, w, h) format (top-left + size).

    Args:
        box1, box2: Bounding boxes as (x, y, w, h).

    Returns:
        IoU value in [0, 1].
    """
    x1_a, y1_a = box1[0], box1[1]
    x2_a, y2_a = box1[0] + box1[2], box1[1] + box1[3]

    x1_b, y1_b = box2[0], box2[1]
    x2_b, y2_b = box2[0] + box2[2], box2[1] + box2[3]

    inter_x1 = max(x1_a, x1_b)
    inter_y1 = max(y1_a, y1_b)
    inter_x2 = min(x2_a, x2_b)
    inter_y2 = min(y2_a, y2_b)

    if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = box1[2] * box1[3]
    area_b = box2[2] * box2[3]
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def make_output_dirs():
    """
    Create all required output directories.
    """
    for split in ["train", "val"]:
        os.makedirs(os.path.join(OUTPUT_IMAGES_VISIBLE, split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_IMAGES_THERMAL, split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_LABELS, split), exist_ok=True)
    os.makedirs(OUTPUT_METADATA, exist_ok=True)
    os.makedirs(OUTPUT_PLOTS, exist_ok=True)


# Import these at module level so they're available
from config import OUTPUT_IMAGES_VISIBLE, OUTPUT_IMAGES_THERMAL, OUTPUT_LABELS, OUTPUT_METADATA, OUTPUT_PLOTS
