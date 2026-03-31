"""
=============================================================================
Step 5: PLOTTING AND VISUALIZATION
=============================================================================

Step 5: GENERATE ALL ANALYSIS PLOTS
    This script reads the CSV metadata exported in Step 4 and generates
    comprehensive visualizations for the thesis analysis. All plots are
    saved as high-resolution PNGs.

    Plot categories:
    5.1  Dataset composition (split, day/night, frames with/without people)
    5.2  Object size distribution (small/medium/large by day/night)
    5.3  Brightness and contrast distributions (day vs night)
    5.4  Crowd density analysis
    5.5  Occlusion and truncation analysis
    5.6  Bounding box height histogram (distance proxy)
    5.7  Object position heatmap
    5.8  Overlap (IoU) distribution
    5.9  Aspect ratio distribution
    5.10 Per-frame people count distribution
    5.11 Edge density vs brightness scatter
    5.12 Combined subgroup analysis grid

    Note: Training/validation loss curves (cls, dfl, bbox) and PR curves,
    precision, mAP50, Recall plots are generated AFTER training, not here.
    This script creates a template function for those post-training plots
    that reads the YOLO training results CSV.
=============================================================================
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import List, Dict

from config import OUTPUT_METADATA, OUTPUT_PLOTS


# =============================================================================
# Step 5.0: HELPER — LOAD CSV
#   Load a CSV file into a list of dicts.
# =============================================================================

def load_csv(filename: str) -> List[Dict]:
    """
    Load a CSV file from the metadata directory.

    Args:
        filename: CSV filename (relative to OUTPUT_METADATA).

    Returns:
        List of row dicts.
    """
    path = os.path.join(OUTPUT_METADATA, filename)
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def save_plot(fig, name: str):
    """Save a matplotlib figure to the plots directory."""
    path = os.path.join(OUTPUT_PLOTS, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# =============================================================================
# Step 5.1: DATASET COMPOSITION
#   Bar charts showing the number of frames per split and day/night.
# =============================================================================

def plot_dataset_composition(frame_data: List[Dict]):
    """
    Plot dataset composition: frames per split and day/night breakdown.
    """
    print("[Step 5.1] Plotting dataset composition...")

    split_counts = defaultdict(int)
    split_dn_counts = defaultdict(lambda: defaultdict(int))

    for row in frame_data:
        split = row["split"]
        dn = row.get("day_night", "unknown")
        split_counts[split] += 1
        split_dn_counts[split][dn] += 1

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Total frames per split
    splits = ["train", "val", "test"]
    counts = [split_counts.get(s, 0) for s in splits]
    colors = ["#2196F3", "#4CAF50", "#FF9800"]
    axes[0].bar(splits, counts, color=colors)
    axes[0].set_title("Frames per Split", fontsize=14)
    axes[0].set_ylabel("Number of Frames")
    for i, c in enumerate(counts):
        axes[0].text(i, c + max(counts)*0.01, str(c), ha="center", fontsize=10)

    # Day/night per split (stacked bar)
    day_counts = [split_dn_counts[s].get("day", 0) for s in splits]
    night_counts = [split_dn_counts[s].get("night", 0) for s in splits]
    x = np.arange(len(splits))
    width = 0.4
    axes[1].bar(x - width/2, day_counts, width, label="Day", color="#FFC107")
    axes[1].bar(x + width/2, night_counts, width, label="Night", color="#3F51B5")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(splits)
    axes[1].set_title("Day vs Night per Split", fontsize=14)
    axes[1].set_ylabel("Number of Frames")
    axes[1].legend()

    fig.suptitle("Dataset Composition", fontsize=16, y=1.02)
    fig.tight_layout()
    save_plot(fig, "01_dataset_composition")


# =============================================================================
# Step 5.2: OBJECT SIZE DISTRIBUTION
#   Bar chart of small/medium/large objects by day and night.
# =============================================================================

def plot_size_distribution(object_data: List[Dict]):
    """
    Plot the distribution of object sizes (small/medium/large) by day/night.
    """
    print("[Step 5.2] Plotting object size distribution...")

    groups = defaultdict(lambda: defaultdict(int))
    for row in object_data:
        dn = row.get("day_night", "unknown")
        cat = row.get("size_category", "unknown")
        groups[dn][cat] += 1

    fig, ax = plt.subplots(figsize=(10, 6))

    categories = ["small", "medium", "large"]
    x = np.arange(len(categories))
    width = 0.35

    day_vals = [groups.get("day", {}).get(c, 0) for c in categories]
    night_vals = [groups.get("night", {}).get(c, 0) for c in categories]

    bars1 = ax.bar(x - width/2, day_vals, width, label="Day", color="#FFC107")
    bars2 = ax.bar(x + width/2, night_vals, width, label="Night", color="#3F51B5")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n(≤{t}px)" for c, t in
                        zip(categories, ["32", "96", ">96"])])
    ax.set_ylabel("Number of Objects")
    ax.set_title("Object Size Distribution by Day/Night", fontsize=14)
    ax.legend()

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h + max(day_vals + night_vals)*0.01,
                        str(int(h)), ha="center", fontsize=9)

    fig.tight_layout()
    save_plot(fig, "02_size_distribution")


# =============================================================================
# Step 5.3: BRIGHTNESS AND CONTRAST DISTRIBUTIONS
#   Histograms comparing day vs night images.
# =============================================================================

def plot_brightness_contrast(frame_data: List[Dict]):
    """
    Plot brightness and contrast histograms for day vs night.
    """
    print("[Step 5.3] Plotting brightness and contrast distributions...")

    day_brightness = []
    night_brightness = []
    day_contrast = []
    night_contrast = []

    for row in frame_data:
        b = float(row.get("brightness", 0))
        c = float(row.get("contrast", 0))
        if row.get("day_night") == "day":
            day_brightness.append(b)
            day_contrast.append(c)
        elif row.get("day_night") == "night":
            night_brightness.append(b)
            night_contrast.append(c)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Brightness histogram
    axes[0].hist(day_brightness, bins=50, alpha=0.7, label="Day", color="#FFC107")
    axes[0].hist(night_brightness, bins=50, alpha=0.7, label="Night", color="#3F51B5")
    axes[0].set_xlabel("Mean Brightness")
    axes[0].set_ylabel("Number of Frames")
    axes[0].set_title("Brightness Distribution", fontsize=14)
    axes[0].legend()

    # Contrast histogram
    axes[1].hist(day_contrast, bins=50, alpha=0.7, label="Day", color="#FFC107")
    axes[1].hist(night_contrast, bins=50, alpha=0.7, label="Night", color="#3F51B5")
    axes[1].set_xlabel("Contrast (Std Dev)")
    axes[1].set_ylabel("Number of Frames")
    axes[1].set_title("Contrast Distribution", fontsize=14)
    axes[1].legend()

    fig.suptitle("Image Quality Metrics: Day vs Night", fontsize=16, y=1.02)
    fig.tight_layout()
    save_plot(fig, "03_brightness_contrast")


# =============================================================================
# Step 5.4: CROWD DENSITY ANALYSIS
#   Distribution of frames by number of people.
# =============================================================================

def plot_crowd_density(frame_data: List[Dict]):
    """
    Plot the distribution of people count per frame.
    """
    print("[Step 5.4] Plotting crowd density analysis...")

    people_counts = [int(row.get("num_people", 0)) for row in frame_data]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    max_count = min(max(people_counts) if people_counts else 10, 15)
    bins = range(0, max_count + 2)
    axes[0].hist(people_counts, bins=bins, color="#2196F3", edgecolor="black", alpha=0.8)
    axes[0].set_xlabel("Number of People per Frame")
    axes[0].set_ylabel("Number of Frames")
    axes[0].set_title("People Count Distribution", fontsize=14)

    # Overlap pairs vs people count
    overlaps = defaultdict(list)
    for row in frame_data:
        n = min(int(row.get("num_people", 0)), 10)
        o = int(row.get("num_overlapping_pairs", 0))
        overlaps[n].append(o)

    x_vals = sorted(overlaps.keys())
    y_means = [np.mean(overlaps[x]) for x in x_vals]
    axes[1].bar(x_vals, y_means, color="#E91E63", edgecolor="black", alpha=0.8)
    axes[1].set_xlabel("Number of People per Frame")
    axes[1].set_ylabel("Avg Overlapping Pairs")
    axes[1].set_title("Annotation Overlap vs Crowd Density", fontsize=14)

    fig.suptitle("Crowd Density Analysis", fontsize=16, y=1.02)
    fig.tight_layout()
    save_plot(fig, "04_crowd_density")


# =============================================================================
# Step 5.5: OCCLUSION AND TRUNCATION ANALYSIS
#   Bar charts showing occlusion levels and truncation by size category.
# =============================================================================

def plot_occlusion_truncation(object_data: List[Dict]):
    """
    Plot occlusion level and truncation flag distributions.
    """
    print("[Step 5.5] Plotting occlusion and truncation analysis...")

    occ_counts = defaultdict(int)
    trunc_by_size = defaultdict(lambda: {"truncated": 0, "not_truncated": 0})
    occ_by_size = defaultdict(lambda: defaultdict(int))

    for row in object_data:
        occ = int(row.get("occlusion", 0))
        trunc = int(row.get("truncated", 0))
        cat = row.get("size_category", "unknown")

        occ_counts[occ] += 1

        if trunc:
            trunc_by_size[cat]["truncated"] += 1
        else:
            trunc_by_size[cat]["not_truncated"] += 1

        occ_by_size[cat][occ] += 1

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Occlusion level distribution
    occ_labels = ["None (0)", "Partial (1)", "Heavy (2)"]
    occ_vals = [occ_counts.get(i, 0) for i in range(3)]
    colors_occ = ["#4CAF50", "#FFC107", "#F44336"]
    axes[0].bar(occ_labels, occ_vals, color=colors_occ)
    axes[0].set_ylabel("Number of Objects")
    axes[0].set_title("Occlusion Level Distribution", fontsize=14)
    for i, v in enumerate(occ_vals):
        axes[0].text(i, v + max(occ_vals)*0.01, str(v), ha="center", fontsize=10)

    # Truncation by size
    categories = ["small", "medium", "large"]
    trunc_vals = [trunc_by_size[c]["truncated"] for c in categories]
    not_trunc = [trunc_by_size[c]["not_truncated"] for c in categories]
    x = np.arange(len(categories))
    width = 0.35
    axes[1].bar(x - width/2, not_trunc, width, label="Not Truncated", color="#4CAF50")
    axes[1].bar(x + width/2, trunc_vals, width, label="Truncated", color="#F44336")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(categories)
    axes[1].set_ylabel("Number of Objects")
    axes[1].set_title("Truncation by Size Category", fontsize=14)
    axes[1].legend()

    fig.suptitle("Occlusion & Truncation Analysis", fontsize=16, y=1.02)
    fig.tight_layout()
    save_plot(fig, "05_occlusion_truncation")


# =============================================================================
# Step 5.6: BOUNDING BOX HEIGHT HISTOGRAM
#   Height is a proxy for distance — small heights mean the person
#   is far away and harder to detect.
# =============================================================================

def plot_bbox_height_histogram(object_data: List[Dict]):
    """
    Plot histogram of bounding box heights (distance proxy).
    """
    print("[Step 5.6] Plotting bounding box height histogram...")

    day_heights = []
    night_heights = []
    for row in object_data:
        h = float(row.get("bbox_height_px", 0))
        if row.get("day_night") == "day":
            day_heights.append(h)
        else:
            night_heights.append(h)

    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.linspace(0, 300, 60)
    ax.hist(day_heights, bins=bins, alpha=0.7, label=f"Day (n={len(day_heights)})",
            color="#FFC107")
    ax.hist(night_heights, bins=bins, alpha=0.7, label=f"Night (n={len(night_heights)})",
            color="#3F51B5")
    ax.axvline(x=32, color="red", linestyle="--", alpha=0.5, label="Small threshold (32px)")
    ax.axvline(x=96, color="green", linestyle="--", alpha=0.5, label="Medium threshold (96px)")
    ax.set_xlabel("Bounding Box Height (pixels)")
    ax.set_ylabel("Number of Objects")
    ax.set_title("Object Height Distribution (Distance Proxy)", fontsize=14)
    ax.legend()

    fig.tight_layout()
    save_plot(fig, "06_bbox_height_histogram")


# =============================================================================
# Step 5.7: OBJECT POSITION HEATMAP
#   2D histogram showing where objects appear in the image frame.
#   Helps identify if certain image regions have more detection failures.
# =============================================================================

def plot_position_heatmap(object_data: List[Dict]):
    """
    Plot a 2D heatmap of object center positions in the image.
    """
    print("[Step 5.7] Plotting object position heatmap...")

    cx_vals = [float(row.get("center_x_norm", 0.5)) for row in object_data]
    cy_vals = [float(row.get("center_y_norm", 0.5)) for row in object_data]

    fig, ax = plt.subplots(figsize=(8, 8))
    h, xedges, yedges, im = ax.hist2d(
        cx_vals, cy_vals, bins=30, cmap="YlOrRd",
        range=[[0, 1], [0, 1]]
    )
    ax.set_xlabel("Horizontal Position (normalized)")
    ax.set_ylabel("Vertical Position (normalized)")
    ax.set_title("Object Center Position Heatmap", fontsize=14)
    ax.invert_yaxis()  # Image coordinates: y increases downward
    plt.colorbar(im, ax=ax, label="Object Count")

    fig.tight_layout()
    save_plot(fig, "07_position_heatmap")


# =============================================================================
# Step 5.8: IoU OVERLAP DISTRIBUTION
#   Histogram of max IoU with neighbors for each object.
# =============================================================================

def plot_iou_distribution(object_data: List[Dict]):
    """
    Plot distribution of max IoU overlap with neighbors.
    """
    print("[Step 5.8] Plotting IoU overlap distribution...")

    ious = [float(row.get("max_iou_neighbor", 0)) for row in object_data]
    nonzero_ious = [v for v in ious if v > 0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # All objects
    axes[0].hist(ious, bins=50, color="#9C27B0", edgecolor="black", alpha=0.8)
    axes[0].set_xlabel("Max IoU with Neighbor")
    axes[0].set_ylabel("Number of Objects")
    axes[0].set_title("IoU Overlap Distribution (All Objects)", fontsize=14)

    # Only objects with nonzero overlap
    if nonzero_ious:
        axes[1].hist(nonzero_ious, bins=30, color="#E91E63", edgecolor="black", alpha=0.8)
    axes[1].set_xlabel("Max IoU with Neighbor")
    axes[1].set_ylabel("Number of Objects")
    axes[1].set_title("IoU Distribution (Overlapping Only)", fontsize=14)

    fig.suptitle("Annotation Overlap Analysis", fontsize=16, y=1.02)
    fig.tight_layout()
    save_plot(fig, "08_iou_distribution")


# =============================================================================
# Step 5.9: ASPECT RATIO DISTRIBUTION
#   Histogram of bounding box aspect ratios.
# =============================================================================

def plot_aspect_ratio(object_data: List[Dict]):
    """
    Plot distribution of bounding box aspect ratios (width/height).
    """
    print("[Step 5.9] Plotting aspect ratio distribution...")

    ratios = [float(row.get("aspect_ratio", 0)) for row in object_data
              if float(row.get("aspect_ratio", 0)) > 0]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(ratios, bins=50, color="#00BCD4", edgecolor="black", alpha=0.8)
    ax.set_xlabel("Aspect Ratio (Width / Height)")
    ax.set_ylabel("Number of Objects")
    ax.set_title("Bounding Box Aspect Ratio Distribution", fontsize=14)
    ax.axvline(x=np.median(ratios) if ratios else 0.4, color="red",
               linestyle="--", label=f"Median: {np.median(ratios):.2f}" if ratios else "")
    ax.legend()

    fig.tight_layout()
    save_plot(fig, "09_aspect_ratio")


# =============================================================================
# Step 5.10: PER-FRAME PEOPLE COUNT BY DAY/NIGHT
# =============================================================================

def plot_people_count_by_daynight(frame_data: List[Dict]):
    """
    Compare people-per-frame distributions for day vs night.
    """
    print("[Step 5.10] Plotting people count by day/night...")

    day_counts = [int(row.get("num_people", 0))
                  for row in frame_data if row.get("day_night") == "day"]
    night_counts = [int(row.get("num_people", 0))
                    for row in frame_data if row.get("day_night") == "night"]

    fig, ax = plt.subplots(figsize=(10, 5))
    max_val = min(max(day_counts + night_counts) if (day_counts or night_counts) else 10, 15)
    bins = range(0, max_val + 2)
    ax.hist(day_counts, bins=bins, alpha=0.7, label=f"Day (n={len(day_counts)})",
            color="#FFC107")
    ax.hist(night_counts, bins=bins, alpha=0.7, label=f"Night (n={len(night_counts)})",
            color="#3F51B5")
    ax.set_xlabel("Number of People per Frame")
    ax.set_ylabel("Number of Frames")
    ax.set_title("People Count Distribution: Day vs Night", fontsize=14)
    ax.legend()

    fig.tight_layout()
    save_plot(fig, "10_people_count_daynight")


# =============================================================================
# Step 5.11: EDGE DENSITY VS BRIGHTNESS SCATTER
#   Scatter plot to see if night images have different scene complexity.
# =============================================================================

def plot_edge_vs_brightness(frame_data: List[Dict]):
    """
    Scatter plot of edge density vs brightness, colored by day/night.
    """
    print("[Step 5.11] Plotting edge density vs brightness...")

    day_b, day_e = [], []
    night_b, night_e = [], []

    for row in frame_data:
        b = float(row.get("brightness", 0))
        e = float(row.get("edge_density", 0))
        if row.get("day_night") == "day":
            day_b.append(b)
            day_e.append(e)
        else:
            night_b.append(b)
            night_e.append(e)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(day_b, day_e, alpha=0.15, s=5, label="Day", color="#FFC107")
    ax.scatter(night_b, night_e, alpha=0.15, s=5, label="Night", color="#3F51B5")
    ax.set_xlabel("Mean Brightness")
    ax.set_ylabel("Edge Density")
    ax.set_title("Scene Complexity vs Illumination", fontsize=14)
    ax.legend(markerscale=5)

    fig.tight_layout()
    save_plot(fig, "11_edge_vs_brightness")


# =============================================================================
# Step 5.12: COMBINED SUBGROUP GRID
#   Multi-panel figure summarizing key metrics across subgroups for
#   quick comparison in the thesis.
# =============================================================================

def plot_subgroup_grid(frame_data: List[Dict], object_data: List[Dict]):
    """
    Create a 2x3 grid of key subgroup comparisons.
    """
    print("[Step 5.12] Plotting combined subgroup analysis grid...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Panel (0,0): Object count by size and day/night
    size_dn = defaultdict(lambda: defaultdict(int))
    for row in object_data:
        size_dn[row.get("size_category", "?")][row.get("day_night", "?")] += 1

    cats = ["small", "medium", "large"]
    x = np.arange(len(cats))
    w = 0.35
    d = [size_dn[c].get("day", 0) for c in cats]
    n = [size_dn[c].get("night", 0) for c in cats]
    axes[0, 0].bar(x - w/2, d, w, label="Day", color="#FFC107")
    axes[0, 0].bar(x + w/2, n, w, label="Night", color="#3F51B5")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(cats)
    axes[0, 0].set_title("Objects by Size & Time")
    axes[0, 0].legend(fontsize=8)

    # Panel (0,1): Occlusion by day/night
    occ_dn = defaultdict(lambda: defaultdict(int))
    for row in object_data:
        occ_dn[int(row.get("occlusion", 0))][row.get("day_night", "?")] += 1

    occ_labels = ["None", "Partial", "Heavy"]
    d = [occ_dn[i].get("day", 0) for i in range(3)]
    n = [occ_dn[i].get("night", 0) for i in range(3)]
    x = np.arange(3)
    axes[0, 1].bar(x - w/2, d, w, label="Day", color="#FFC107")
    axes[0, 1].bar(x + w/2, n, w, label="Night", color="#3F51B5")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(occ_labels)
    axes[0, 1].set_title("Occlusion by Time")
    axes[0, 1].legend(fontsize=8)

    # Panel (0,2): Average brightness by split
    split_brightness = defaultdict(list)
    for row in frame_data:
        b = float(row.get("brightness", 0))
        split_brightness[row["split"]].append(b)

    splits = ["train", "val", "test"]
    means = [np.mean(split_brightness.get(s, [0])) for s in splits]
    stds = [np.std(split_brightness.get(s, [0])) for s in splits]
    axes[0, 2].bar(splits, means, yerr=stds, color=["#2196F3", "#4CAF50", "#FF9800"],
                   capsize=5)
    axes[0, 2].set_title("Avg Brightness by Split")
    axes[0, 2].set_ylabel("Mean Brightness")

    # Panel (1,0): Object overlap rates
    overlap_rates = defaultdict(lambda: {"has": 0, "no": 0})
    for row in object_data:
        cat = row.get("size_category", "?")
        if row.get("has_overlap") in ("True", True, "1"):
            overlap_rates[cat]["has"] += 1
        else:
            overlap_rates[cat]["no"] += 1

    has_vals = [overlap_rates[c]["has"] for c in cats]
    no_vals = [overlap_rates[c]["no"] for c in cats]
    total_vals = [h + n for h, n in zip(has_vals, no_vals)]
    pct_vals = [h/t*100 if t > 0 else 0 for h, t in zip(has_vals, total_vals)]
    axes[1, 0].bar(cats, pct_vals, color="#E91E63")
    axes[1, 0].set_title("% Objects with Overlap by Size")
    axes[1, 0].set_ylabel("% with Overlap")

    # Panel (1,1): Distance from center by size
    dist_by_size = defaultdict(list)
    for row in object_data:
        dist_by_size[row.get("size_category", "?")].append(
            float(row.get("dist_from_center", 0)))

    means = [np.mean(dist_by_size.get(c, [0])) for c in cats]
    axes[1, 1].bar(cats, means, color="#009688")
    axes[1, 1].set_title("Avg Distance from Center by Size")
    axes[1, 1].set_ylabel("Normalized Distance")

    # Panel (1,2): Original class breakdown
    class_counts = defaultdict(int)
    for row in object_data:
        class_counts[row.get("original_class", "person")] += 1

    labels = list(class_counts.keys())
    values = list(class_counts.values())
    axes[1, 2].pie(values, labels=labels, autopct="%1.1f%%",
                   colors=["#2196F3", "#4CAF50", "#FF9800"][:len(labels)])
    axes[1, 2].set_title("Original Class Distribution")

    fig.suptitle("Combined Subgroup Analysis", fontsize=16, y=1.02)
    fig.tight_layout()
    save_plot(fig, "12_subgroup_grid")


# =============================================================================
# Step 5.13: POST-TRAINING PLOTS (TEMPLATE)
#   These functions generate plots from YOLO training results.
#   Call them after training with the path to the results.csv file.
# =============================================================================

def plot_training_curves(results_csv_path: str):
    """
    Plot training and validation loss curves from YOLO results.csv.

    The YOLO results.csv typically has columns like:
    epoch, train/box_loss, train/cls_loss, train/dfl_loss,
    metrics/precision(B), metrics/recall(B), metrics/mAP50(B),
    metrics/mAP50-95(B), val/box_loss, val/cls_loss, val/dfl_loss

    Args:
        results_csv_path: Path to the YOLO results.csv file.
    """
    print("[Step 5.13] Plotting training curves from YOLO results...")

    rows = []
    with open(results_csv_path, "r") as f:
        reader = csv.DictReader(f)
        # YOLO CSVs have spaces in headers, strip them
        reader.fieldnames = [h.strip() for h in reader.fieldnames]
        rows = list(reader)

    if not rows:
        print("  WARNING: No data found in results.csv")
        return

    epochs = list(range(1, len(rows) + 1))

    # Helper to safely extract float values
    def get_col(name):
        return [float(r.get(name, 0)) for r in rows]

    # --- Loss curves ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    loss_types = [
        ("train/box_loss", "val/box_loss", "Box Loss"),
        ("train/cls_loss", "val/cls_loss", "Classification Loss"),
        ("train/dfl_loss", "val/dfl_loss", "DFL Loss"),
    ]

    for i, (train_col, val_col, title) in enumerate(loss_types):
        train_vals = get_col(train_col)
        val_vals = get_col(val_col)

        axes[i].plot(epochs, train_vals, label="Train", color="#2196F3", linewidth=2)
        axes[i].plot(epochs, val_vals, label="Val", color="#F44336", linewidth=2)
        axes[i].set_xlabel("Epoch")
        axes[i].set_ylabel("Loss")
        axes[i].set_title(title, fontsize=14)
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    fig.suptitle("Training & Validation Loss Curves", fontsize=16, y=1.02)
    fig.tight_layout()
    save_plot(fig, "13_training_loss_curves")

    # --- Metrics curves ---
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    metric_types = [
        ("metrics/precision(B)", "Precision"),
        ("metrics/recall(B)", "Recall"),
        ("metrics/mAP50(B)", "mAP@50"),
        ("metrics/mAP50-95(B)", "mAP@50-95"),
    ]

    colors = ["#4CAF50", "#FF9800", "#9C27B0", "#E91E63"]
    for i, (col, title) in enumerate(metric_types):
        vals = get_col(col)
        axes[i].plot(epochs, vals, color=colors[i], linewidth=2)
        axes[i].set_xlabel("Epoch")
        axes[i].set_ylabel(title)
        axes[i].set_title(title, fontsize=14)
        axes[i].grid(True, alpha=0.3)
        axes[i].set_ylim(0, 1)

    fig.suptitle("Detection Metrics Over Training", fontsize=16, y=1.02)
    fig.tight_layout()
    save_plot(fig, "14_training_metrics")

    print("  Training curves plotted.")


def plot_pr_curve(predictions_csv_path: str = None):
    """
    Plot Precision-Recall curve. This is a template — actual implementation
    depends on the YOLO evaluation output format.

    For YOLOv8, the PR curve is typically saved automatically during
    validation. This function provides a manual alternative using
    per-image predictions.

    Args:
        predictions_csv_path: Path to per-image prediction results.
    """
    print("[Step 5.13b] PR curve plotting template ready.")
    print("  Note: YOLOv8 generates PR curves automatically during val.")
    print("  Use: model.val() which saves PR_curve.png to runs/detect/val/")


# =============================================================================
# Step 5.14: MAIN ENTRY POINT
# =============================================================================

def run_step5(results_csv_path: str = None):
    """
    Execute Step 5: generate all preprocessing analysis plots.

    Args:
        results_csv_path: Optional path to YOLO results.csv for post-training plots.
    """
    print("\n" + "=" * 70)
    print("STEP 5: PLOTTING AND VISUALIZATION")
    print("=" * 70)

    os.makedirs(OUTPUT_PLOTS, exist_ok=True)

    # Load CSVs
    frame_data = load_csv("frame_metadata.csv")
    object_data = load_csv("object_metadata.csv")

    print(f"[Step 5] Loaded {len(frame_data)} frames, {len(object_data)} objects")

    # Generate all dataset analysis plots
    plot_dataset_composition(frame_data)
    plot_size_distribution(object_data)
    plot_brightness_contrast(frame_data)
    plot_crowd_density(frame_data)
    plot_occlusion_truncation(object_data)
    plot_bbox_height_histogram(object_data)
    plot_position_heatmap(object_data)
    plot_iou_distribution(object_data)
    plot_aspect_ratio(object_data)
    plot_people_count_by_daynight(frame_data)
    plot_edge_vs_brightness(frame_data)
    plot_subgroup_grid(frame_data, object_data)

    # Post-training plots (if results available)
    if results_csv_path and os.path.exists(results_csv_path):
        plot_training_curves(results_csv_path)

    print(f"\n[Step 5] All plots saved to: {OUTPUT_PLOTS}")


if __name__ == "__main__":
    run_step5()
