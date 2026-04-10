"""
=============================================================================
eval_data.py — Data Loading, Detection Matching, and Metadata Merging
=============================================================================

This module handles the core detection evaluation pipeline:
  1. Loading YOLO-format ground truth labels from .txt files
  2. Coordinate conversion (normalised YOLO centre → pixel xyxy)
  3. Greedy IoU-based prediction-to-GT matching
  4. Day/night classification from KAIST set IDs
  5. Matching GT objects to preprocessing metadata CSVs

The detection matching uses a confidence-ranked greedy strategy:
  - Sort predictions by confidence (highest first)
  - For each prediction, find the unmatched GT with highest IoU
  - If IoU ≥ threshold → True Positive; otherwise → False Positive
  - Unmatched GT boxes → False Negatives

This is the standard PASCAL VOC / COCO matching protocol at a fixed
IoU threshold (default 0.5).

Functions:
    infer_day_night(flat_name)          → "day" or "night"
    load_yolo_labels(label_path)        → list of (cls, cx, cy, w, h)
    yolo_to_xyxy(cx, cy, w, h, size)   → (x1, y1, x2, y2) in pixels
    compute_iou_xyxy(box1, box2)        → IoU float
    evaluate_image(pred_boxes, confs, gt_labels)
                                        → {tp, fp, fn, precision, recall,
                                           gt_matched, fp_boxes, tp_confs}
    match_gt_to_metadata(gt_labels, meta_objects)
                                        → list of metadata dicts
=============================================================================
"""

import os
import numpy as np
from typing import List, Dict, Tuple, Optional

# =============================================================================
# DAY / NIGHT SET MAPPING
#   KAIST pedestrian dataset: 12 video sets (set00 – set11).
#   Day/night assignment matches preprocessing/config.py exactly.
#
#   Day:   set00, set01, set02, set06, set07, set08
#   Night: set03, set04, set05, set09, set10, set11
# =============================================================================

DAY_SETS = {"set00", "set01", "set02", "set06", "set07", "set08"}
NIGHT_SETS = {"set03", "set04", "set05", "set09", "set10", "set11"}


def infer_day_night(flat_name: str) -> str:
    """
    Derive day/night from the KAIST set number embedded in the filename.

    Filename format: "set06_V003_I00123" → extract "set06"
    Lookup in DAY_SETS / NIGHT_SETS.

    Returns "day", "night", or "unknown" if parsing fails.
    """
    try:
        set_id = flat_name.split("_")[0]  # e.g. "set06"
        if set_id in DAY_SETS:
            return "day"
        if set_id in NIGHT_SETS:
            return "night"
        return "unknown"
    except Exception:
        return "unknown"


# =============================================================================
# YOLO LABEL LOADING
# =============================================================================

def load_yolo_labels(label_path: str) -> List[Tuple[int, float, float, float, float]]:
    """
    Load YOLO-format labels from a .txt file.

    YOLO format (one object per line):
        <class_id> <x_centre> <y_centre> <width> <height>
    All coordinates are normalised to [0, 1] relative to image dimensions.

    Returns list of (class_id, x_centre, y_centre, width, height).
    """
    if not os.path.exists(label_path):
        return []

    labels = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls = int(parts[0])
                x, y, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                labels.append((cls, x, y, w, h))
    return labels


# =============================================================================
# COORDINATE CONVERSION
# =============================================================================

def yolo_to_xyxy(cx: float, cy: float, w: float, h: float,
                 img_size: int = 512) -> Tuple[float, float, float, float]:
    """
    Convert normalised YOLO centre format → pixel (x1, y1, x2, y2).

    YOLO stores: centre_x, centre_y, width, height  (all in [0, 1])
    We need:     top-left (x1, y1), bottom-right (x2, y2)  (in pixels)

    Math:
        x1 = (cx - w/2) * img_size
        y1 = (cy - h/2) * img_size
        x2 = (cx + w/2) * img_size
        y2 = (cy + h/2) * img_size
    """
    x1 = (cx - w / 2) * img_size
    y1 = (cy - h / 2) * img_size
    x2 = (cx + w / 2) * img_size
    y2 = (cy + h / 2) * img_size
    return x1, y1, x2, y2


def compute_iou_xyxy(box1, box2) -> float:
    """
    Intersection over Union for two boxes in (x1, y1, x2, y2) pixel format.

    IoU = Area_intersection / Area_union
    where Area_union = Area_box1 + Area_box2 - Area_intersection

    Used as the matching criterion: IoU ≥ 0.5 → match (True Positive).
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x1 >= x2 or y1 >= y2:
        return 0.0

    inter = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


# =============================================================================
# PER-IMAGE DETECTION EVALUATION
#   Greedy IoU matching: highest-confidence prediction matched first.
#
#   Algorithm:
#     1. Sort all predictions by confidence (descending)
#     2. For each prediction in order:
#        a. Find the unmatched GT box with highest IoU
#        b. If IoU ≥ IOU_THRESHOLD → True Positive (mark GT as matched)
#        c. Otherwise → False Positive
#     3. Unmatched GT boxes → False Negatives
#
#   Returns TP/FP/FN counts, per-GT match flags, AND per-FP box details
#   (coordinates + confidence) for false positive analysis.
# =============================================================================

IOU_THRESHOLD = 0.5


def evaluate_image(
    pred_boxes: np.ndarray,
    pred_confs: np.ndarray,
    gt_labels: List[Tuple[int, float, float, float, float]],
    img_size: int = 512,
) -> Dict:
    """
    Match predictions to GT for one image.

    Returns dict with:
        tp, fp, fn          : int counts
        precision, recall   : float
        gt_matched          : list[bool] — which GT boxes were detected
        fp_boxes            : list of {box: [x1,y1,x2,y2], conf: float}
                              for each false positive prediction
        tp_confs            : list of float — confidence of each TP prediction
    """
    gt_boxes = [yolo_to_xyxy(cx, cy, w, h, img_size)
                for _, cx, cy, w, h in gt_labels]

    n_gt = len(gt_boxes)
    n_pred = len(pred_boxes)

    empty_result = lambda tp, fp, fn, prec, rec: {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": prec, "recall": rec,
        "gt_matched": [False] * n_gt,
        "fp_boxes": [],
        "tp_confs": [],
    }

    if n_gt == 0 and n_pred == 0:
        return empty_result(0, 0, 0, 1.0, 1.0)
    if n_gt == 0:
        # All predictions are false positives
        fp_boxes = [{"box": pred_boxes[i].tolist(), "conf": float(pred_confs[i])}
                    for i in range(n_pred)]
        return {"tp": 0, "fp": n_pred, "fn": 0,
                "precision": 0.0, "recall": 1.0,
                "gt_matched": [], "fp_boxes": fp_boxes, "tp_confs": []}
    if n_pred == 0:
        return empty_result(0, 0, n_gt, 1.0, 0.0)

    # Sort predictions by confidence (highest first)
    order = np.argsort(-pred_confs)
    pred_boxes_sorted = pred_boxes[order]
    pred_confs_sorted = pred_confs[order]

    gt_matched = [False] * n_gt
    tp = 0
    fp = 0
    fp_boxes = []
    tp_confs = []

    for i, pred_box in enumerate(pred_boxes_sorted):
        best_iou = 0.0
        best_idx = -1

        for gi, gt_box in enumerate(gt_boxes):
            if gt_matched[gi]:
                continue
            iou = compute_iou_xyxy(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_idx = gi

        if best_iou >= IOU_THRESHOLD and best_idx >= 0:
            tp += 1
            gt_matched[best_idx] = True
            tp_confs.append(float(pred_confs_sorted[i]))
        else:
            fp += 1
            fp_boxes.append({
                "box": pred_box.tolist(),
                "conf": float(pred_confs_sorted[i]),
            })

    fn = sum(1 for m in gt_matched if not m)

    # ── Precision & Recall ──────────────────────────────
    # precision = TP / (TP + FP)  — of all predictions, how many are correct?
    # recall    = TP / (TP + FN)  — of all GT objects, how many are found?
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "gt_matched": gt_matched,
        "fp_boxes": fp_boxes,
        "tp_confs": tp_confs,
    }


# =============================================================================
# MATCH GT OBJECTS TO PREPROCESSING METADATA
#   The preprocessing pipeline writes YOLO labels and object_metadata.csv
#   in the same order.  As a safety measure we match by approximate centre
#   coordinates rather than relying on index alone.
# =============================================================================

def match_gt_to_metadata(
    gt_labels: List[Tuple[int, float, float, float, float]],
    meta_objects: List[Dict],
    img_size: int = 512,
    tol: float = 5.0,
) -> List[Optional[Dict]]:
    """
    For each GT label, find the best-matching metadata object by centre
    distance.  Returns a list parallel to gt_labels (None if no match).
    """
    if not meta_objects:
        return [None] * len(gt_labels)

    matched: List[Optional[Dict]] = []
    used = set()

    for _, cx, cy, w, h in gt_labels:
        gt_cx_px = cx * img_size
        gt_cy_px = cy * img_size
        best_dist = float("inf")
        best_idx = None

        for mi, mo in enumerate(meta_objects):
            if mi in used:
                continue
            mo_cx = float(mo.get("center_x_norm", 0)) * img_size
            mo_cy = float(mo.get("center_y_norm", 0)) * img_size
            dist = ((gt_cx_px - mo_cx) ** 2 + (gt_cy_px - mo_cy) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_idx = mi

        if best_idx is not None and best_dist < tol:
            matched.append(meta_objects[best_idx])
            used.add(best_idx)
        else:
            matched.append(None)

    return matched
