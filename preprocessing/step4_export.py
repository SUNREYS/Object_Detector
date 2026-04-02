"""
=============================================================================
Step 4: METADATA AGGREGATION AND CSV EXPORT
=============================================================================

Step 4: EXPORT ALL METADATA TO CSV FILES
    This script collects all extracted features and metadata from previous
    steps and exports them into structured CSV files for analysis.

    Output CSVs:
    4.1  frame_metadata.csv  — One row per frame with image-level features
    4.2  object_metadata.csv — One row per bounding box with object features
    4.3  dataset_summary.csv — Aggregated statistics by split, day/night, etc.
    4.4  size_distribution.csv — Distribution of object sizes across subgroups

    These CSVs are designed to be loaded into pandas/Excel for creating
    plots and performing statistical analysis of detection performance
    across subgroups (day vs night, small vs large, occluded vs visible, etc.)
=============================================================================
"""

import os
import csv
import pandas as pd
from collections import defaultdict
from typing import List, Dict, Optional

from config import OUTPUT_METADATA


# =============================================================================
# Step 4.1: EXPORT FRAME-LEVEL METADATA
#   One row per frame containing all image-level features extracted in Step 2.
#   This CSV is the primary data source for image-level analysis.
# =============================================================================

FRAME_CSV_COLUMNS = [
    "frame_id",
    "flat_name",
    "modality",
    "split",
    "set_id",
    "video_id",
    "day_night",
    "num_people",
    "num_overlapping_pairs",
    "brightness",
    "contrast",
    "edge_density",
    "skipped",
]


def export_frame_metadata(all_features: List[Dict], modalities: Optional[List[str]] = None) -> str:
    """
    Export frame-level metadata to CSV.

    Writes one row per (frame × modality).  If the CSV already exists and
    ``modalities`` contains only a subset (e.g. ``["greyscale_inversion"]``),
    the existing rows for OTHER modalities are preserved (upsert behaviour).

    Args:
        all_features: List of feature dicts from Step 2.
        modalities:   Which modalities to write rows for.  Defaults to
                      ``["visible", "thermal"]`` (original behaviour).

    Returns:
        Path to the output CSV file.
    """
    if modalities is None:
        modalities = ["visible", "thermal"]

    print(f"[Step 4.1] Exporting frame-level metadata for {modalities}...")

    new_rows = []
    for feat in all_features:
        base = {col: feat.get(col, "") for col in FRAME_CSV_COLUMNS}
        for m in modalities:
            row = dict(base)
            row["modality"] = m
            new_rows.append(row)

    path = os.path.join(OUTPUT_METADATA, "frame_metadata.csv")

    if os.path.exists(path):
        df_existing = pd.read_csv(path, dtype=str)
        df_existing = df_existing[~df_existing["modality"].isin(modalities)]
        df_new = pd.DataFrame(new_rows, columns=FRAME_CSV_COLUMNS).astype(str)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(path, index=False)
        print(f"  Upserted {len(df_new)} rows → {len(df_combined)} total rows in: {path}")
    else:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FRAME_CSV_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            for row in new_rows:
                writer.writerow(row)
        print(f"  Wrote {len(new_rows)} rows to: {path}")

    return path


# =============================================================================
# Step 4.2: EXPORT OBJECT-LEVEL METADATA
#   One row per bounding box annotation. Each row includes both the
#   object features and the parent frame's metadata for easy joining.
# =============================================================================

OBJECT_CSV_COLUMNS = [
    "frame_id",
    "flat_name",
    "modality",
    "split",
    "day_night",
    "original_class",
    "bbox_height_px",
    "bbox_width_px",
    "bbox_area_px",
    "size_category",
    "aspect_ratio",
    "occlusion",
    "truncated",
    "center_x_norm",
    "center_y_norm",
    "dist_from_center",
    "has_overlap",
    "max_iou_neighbor",
]


def export_object_metadata(all_features: List[Dict], modalities: Optional[List[str]] = None) -> str:
    """
    Export per-object metadata to CSV.

    Writes one row per (object × modality).  If the CSV already exists and
    ``modalities`` is a subset, existing rows for other modalities are kept.

    Args:
        all_features: List of feature dicts from Step 2.
        modalities:   Which modalities to write rows for.  Defaults to
                      ``["visible", "thermal"]``.

    Returns:
        Path to the output CSV file.
    """
    if modalities is None:
        modalities = ["visible", "thermal"]

    print(f"[Step 4.2] Exporting object-level metadata for {modalities}...")

    new_rows = []
    for feat in all_features:
        for obj_feat in feat.get("object_features", []):
            base = {col: obj_feat.get(col, "") for col in OBJECT_CSV_COLUMNS}
            for m in modalities:
                row = dict(base)
                row["modality"] = m
                new_rows.append(row)

    path = os.path.join(OUTPUT_METADATA, "object_metadata.csv")

    if os.path.exists(path):
        df_existing = pd.read_csv(path, dtype=str)
        df_existing = df_existing[~df_existing["modality"].isin(modalities)]
        df_new = pd.DataFrame(new_rows, columns=OBJECT_CSV_COLUMNS).astype(str)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(path, index=False)
        print(f"  Upserted {len(df_new)} rows → {len(df_combined)} total rows in: {path}")
    else:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=OBJECT_CSV_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            for row in new_rows:
                writer.writerow(row)
        print(f"  Wrote {len(new_rows)} rows to: {path}")

    return path


# =============================================================================
# Step 4.3: EXPORT DATASET SUMMARY
#   Aggregated statistics broken down by split, day/night, and size category.
#   Useful for high-level overview of the dataset composition.
# =============================================================================

def export_dataset_summary(all_features: List[Dict]) -> str:
    """
    Compute and export aggregated dataset statistics.

    Args:
        all_features: List of feature dicts from Step 2.

    Returns:
        Path to the output CSV file.
    """
    print("[Step 4.3] Computing and exporting dataset summary...")

    # Group by (split, day_night)
    groups = defaultdict(lambda: {
        "num_frames": 0,
        "num_frames_with_people": 0,
        "total_people": 0,
        "total_overlapping_pairs": 0,
        "brightness_sum": 0.0,
        "contrast_sum": 0.0,
        "small_count": 0,
        "medium_count": 0,
        "large_count": 0,
        "occluded_count": 0,
        "truncated_count": 0,
    })

    for feat in all_features:
        key = (feat["split"], feat.get("day_night", "unknown"))
        g = groups[key]
        g["num_frames"] += 1
        g["total_people"] += feat["num_people"]
        g["total_overlapping_pairs"] += feat["num_overlapping_pairs"]
        g["brightness_sum"] += feat.get("brightness", 0)
        g["contrast_sum"] += feat.get("contrast", 0)

        if feat["num_people"] > 0:
            g["num_frames_with_people"] += 1

        for obj_feat in feat.get("object_features", []):
            cat = obj_feat.get("size_category", "")
            if cat == "small":
                g["small_count"] += 1
            elif cat == "medium":
                g["medium_count"] += 1
            elif cat == "large":
                g["large_count"] += 1
            if obj_feat.get("occlusion", 0) > 0:
                g["occluded_count"] += 1
            if obj_feat.get("truncated", 0) > 0:
                g["truncated_count"] += 1

    path = os.path.join(OUTPUT_METADATA, "dataset_summary.csv")
    columns = [
        "split", "day_night", "num_frames", "num_frames_with_people",
        "total_people", "avg_people_per_frame", "total_overlapping_pairs",
        "avg_brightness", "avg_contrast",
        "small_objects", "medium_objects", "large_objects",
        "occluded_objects", "truncated_objects",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()

        for (split, dn), g in sorted(groups.items()):
            n = g["num_frames"]
            row = {
                "split": split,
                "day_night": dn,
                "num_frames": n,
                "num_frames_with_people": g["num_frames_with_people"],
                "total_people": g["total_people"],
                "avg_people_per_frame": round(g["total_people"] / n, 2) if n else 0,
                "total_overlapping_pairs": g["total_overlapping_pairs"],
                "avg_brightness": round(g["brightness_sum"] / n, 2) if n else 0,
                "avg_contrast": round(g["contrast_sum"] / n, 2) if n else 0,
                "small_objects": g["small_count"],
                "medium_objects": g["medium_count"],
                "large_objects": g["large_count"],
                "occluded_objects": g["occluded_count"],
                "truncated_objects": g["truncated_count"],
            }
            writer.writerow(row)

    print(f"  Wrote summary to: {path}")
    return path


# =============================================================================
# Step 4.4: EXPORT SIZE DISTRIBUTION
#   Detailed breakdown of object size distribution across subgroups.
#   Helps identify if the translation model degrades on small/far objects.
# =============================================================================

def export_size_distribution(all_features: List[Dict]) -> str:
    """
    Export detailed size distribution across subgroups.

    Args:
        all_features: List of feature dicts from Step 2.

    Returns:
        Path to the output CSV file.
    """
    print("[Step 4.4] Exporting size distribution...")

    # Group by (split, day_night, size_category)
    counts = defaultdict(int)
    height_sums = defaultdict(float)

    for feat in all_features:
        for obj_feat in feat.get("object_features", []):
            key = (
                feat["split"],
                feat.get("day_night", "unknown"),
                obj_feat.get("size_category", "unknown"),
            )
            counts[key] += 1
            height_sums[key] += obj_feat.get("bbox_height_px", 0)

    path = os.path.join(OUTPUT_METADATA, "size_distribution.csv")
    columns = ["split", "day_night", "size_category", "count", "avg_height_px"]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()

        for (split, dn, cat), count in sorted(counts.items()):
            avg_h = height_sums[(split, dn, cat)] / count if count > 0 else 0
            writer.writerow({
                "split": split,
                "day_night": dn,
                "size_category": cat,
                "count": count,
                "avg_height_px": round(avg_h, 1),
            })

    print(f"  Wrote size distribution to: {path}")
    return path


# =============================================================================
# Step 4.5: EXPORT CROWD DENSITY ANALYSIS
#   Analyze how crowd density (number of people per frame) relates to
#   other features. Dense crowds with many overlapping boxes are harder
#   to detect accurately.
# =============================================================================

def export_crowd_analysis(all_features: List[Dict]) -> str:
    """
    Export crowd density analysis: frames grouped by people count.

    Args:
        all_features: List of feature dicts.

    Returns:
        Path to the output CSV file.
    """
    print("[Step 4.5] Exporting crowd density analysis...")

    # Bin frames by number of people
    bins = defaultdict(lambda: {
        "count": 0,
        "overlap_pairs": 0,
        "brightness_sum": 0.0,
        "contrast_sum": 0.0,
    })

    for feat in all_features:
        n_people = feat["num_people"]
        # Bin: 0, 1, 2, 3, 4, 5+
        bin_label = str(n_people) if n_people <= 4 else "5+"
        b = bins[bin_label]
        b["count"] += 1
        b["overlap_pairs"] += feat["num_overlapping_pairs"]
        b["brightness_sum"] += feat.get("brightness", 0)
        b["contrast_sum"] += feat.get("contrast", 0)

    path = os.path.join(OUTPUT_METADATA, "crowd_analysis.csv")
    columns = ["people_count", "num_frames", "total_overlapping_pairs",
               "avg_brightness", "avg_contrast"]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()

        for label in ["0", "1", "2", "3", "4", "5+"]:
            if label in bins:
                b = bins[label]
                n = b["count"]
                writer.writerow({
                    "people_count": label,
                    "num_frames": n,
                    "total_overlapping_pairs": b["overlap_pairs"],
                    "avg_brightness": round(b["brightness_sum"] / n, 2) if n else 0,
                    "avg_contrast": round(b["contrast_sum"] / n, 2) if n else 0,
                })

    print(f"  Wrote crowd analysis to: {path}")
    return path


# =============================================================================
# Step 4.6: MAIN ENTRY POINT
# =============================================================================

def run_step4(all_features: List[Dict], modalities: Optional[List[str]] = None) -> Dict[str, str]:
    """
    Execute Step 4: export all metadata to CSV files.

    Args:
        all_features: List of feature dicts from Steps 2-3.
        modalities:   Which modalities to write rows for.  Defaults to
                      ``["visible", "thermal"]``.

    Returns:
        Dict mapping CSV name to file path.
    """
    print("\n" + "=" * 70)
    print("STEP 4: METADATA AGGREGATION AND CSV EXPORT")
    print("=" * 70)

    os.makedirs(OUTPUT_METADATA, exist_ok=True)

    outputs = {}
    outputs["frame_metadata"] = export_frame_metadata(all_features, modalities=modalities)
    outputs["object_metadata"] = export_object_metadata(all_features, modalities=modalities)
    outputs["dataset_summary"] = export_dataset_summary(all_features)
    outputs["size_distribution"] = export_size_distribution(all_features)
    outputs["crowd_analysis"] = export_crowd_analysis(all_features)

    print(f"\n[Step 4] All CSV files exported to: {OUTPUT_METADATA}")
    return outputs


if __name__ == "__main__":
    print("Step 4 must be run after Steps 1-3. Use main.py to run the full pipeline.")
