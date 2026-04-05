"""
=============================================================================
Paired Image Quality Evaluation: Synthetic vs Real Thermal
=============================================================================

Compares synthetic thermal translations (PI-GAN, greyscale_inversion, etc.)
against real KAIST thermal images using paired per-image and per-object
metrics.  Produces thesis-quality plots split by ALL / DAY / NIGHT.

Metrics computed per IMAGE:
    - SSIM              (structural similarity, range [0, 1])
    - PSNR              (peak signal-to-noise ratio, dB)
    - Mean absolute error (MAE)
    - Edge coherence    (correlation of Canny edge maps)
    - Brightness shift  (mean intensity difference)
    - Contrast ratio    (std_synthetic / std_real)
    - Ghost score       (FFT spectral anomaly ratio)
    - Hallucination score (pixel-level artifact intensity in non-person regions)

Metrics computed per OBJECT (inside each GT bounding box):
    - SSIM (local)
    - Object blur difference (Laplacian variance delta)
    - Edge strength difference (Sobel magnitude delta)
    - Thermal signature error (mean intensity difference inside bbox)
    - Foreground contrast preservation (fg-bg contrast delta)

Grouping dimensions:
    - ALL images
    - DAY vs NIGHT  (based on KAIST set ID)

Output:
    - CSV with per-image metrics
    - CSV with per-object metrics
    - 10+ publication-ready plots saved to preprocessing/image_quality_plots/

Usage:
    python preprocessing/evaluate_image_quality.py
    python preprocessing/evaluate_image_quality.py --modality greyscale_inversion
    python preprocessing/evaluate_image_quality.py --modality PI-GAN_gen --split val
    python preprocessing/evaluate_image_quality.py --modality PI-GAN_gen greyscale_inversion

=============================================================================
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import cv2
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

from skimage.metrics import structural_similarity as ssim

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ── Paths ────────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DATASET_ROOT = os.path.join(
    PROJECT_ROOT, "datasets", "kaist_processed"
)
IMAGES_ROOT = os.path.join(DATASET_ROOT, "images")
LABELS_ROOT = os.path.join(DATASET_ROOT, "labels")

THERMAL_DIR = os.path.join(IMAGES_ROOT, "thermal")       # reference
PLOTS_DIR   = os.path.join(SCRIPT_DIR, "image_quality_plots")
CSV_DIR     = os.path.join(SCRIPT_DIR, "image_quality_csv")

# ── KAIST day / night mapping ───────────────────────────────────────────────

DAY_SETS   = {"set00", "set01", "set02", "set06", "set07", "set08"}
NIGHT_SETS = {"set03", "set04", "set05", "set09", "set10", "set11"}

# ── Plot style ───────────────────────────────────────────────────────────────

sns.set_theme(style="whitegrid", font_scale=1.15)
PALETTE = sns.color_palette("colorblind")

MODALITY_DISPLAY = {
    "greyscale_inversion": "Greyscale Inv.",
    "PI-GAN_gen":          "PI-GAN",
    "ThermalGAN":          "ThermalGAN",
    "sRGB-TIR":            "sRGB-TIR",
    "ThermalGen":          "ThermalGen",
}


def display_name(modality: str) -> str:
    return MODALITY_DISPLAY.get(modality, modality)


def save_fig(fig, name: str):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, f"{name}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  PER-IMAGE METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_image_metrics(
    real_gray: np.ndarray,
    synth_gray: np.ndarray,
    gt_masks: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute all image-level quality metrics between a real thermal image
    and its synthetic counterpart.

    Args:
        real_gray:  Real thermal image, uint8 grayscale H x W.
        synth_gray: Synthetic thermal image, uint8 grayscale H x W.
        gt_masks:   Optional binary mask where 1 = person regions.

    Returns:
        Dict of metric_name -> value.
    """
    # Cast to float for numeric operations
    r = real_gray.astype(np.float64)
    s = synth_gray.astype(np.float64)

    # 1. SSIM (full image)
    ssim_val, _ = ssim(real_gray, synth_gray, full=True)

    # 2. PSNR
    mse = np.mean((r - s) ** 2)
    psnr_val = 10.0 * np.log10(255.0 ** 2 / mse) if mse > 0 else 100.0

    # 3. MAE
    mae_val = float(np.mean(np.abs(r - s)))

    # 4. Edge coherence (Canny edge correlation)
    edges_r = cv2.Canny(real_gray, 50, 150).astype(np.float64)
    edges_s = cv2.Canny(synth_gray, 50, 150).astype(np.float64)
    if edges_r.std() > 0 and edges_s.std() > 0:
        edge_corr = float(np.corrcoef(edges_r.ravel(), edges_s.ravel())[0, 1])
    else:
        edge_corr = 0.0

    # 5. Brightness shift
    brightness_shift = float(np.mean(s) - np.mean(r))

    # 6. Contrast ratio
    std_r = float(np.std(r))
    std_s = float(np.std(s))
    contrast_ratio = std_s / std_r if std_r > 0 else 1.0

    # 7. Ghost score (FFT spectral anomaly)
    ghost_score = _ghost_score_fft(r, s)

    # 8. Hallucination score (pixel-level in non-person regions)
    if gt_masks is not None and gt_masks.any():
        non_person = ~gt_masks.astype(bool)
        if non_person.any():
            residual = np.abs(r - s)
            halluc_score = float(np.mean(residual[non_person]))
        else:
            halluc_score = float("nan")
    else:
        # No GT => use full image residual (less precise but still useful)
        halluc_score = float(np.mean(np.abs(r - s)))

    return {
        "ssim":              round(ssim_val, 5),
        "psnr":              round(psnr_val, 3),
        "mae":               round(mae_val, 3),
        "edge_coherence":    round(edge_corr, 5),
        "brightness_shift":  round(brightness_shift, 3),
        "contrast_ratio":    round(contrast_ratio, 5),
        "ghost_score":       round(ghost_score, 5),
        "hallucination_score": round(halluc_score, 3),
    }


def _ghost_score_fft(r: np.ndarray, s: np.ndarray) -> float:
    """
    Ghost pattern detection via frequency-domain analysis.

    Compares the 2-D power spectra of real vs synthetic images.  Periodic
    artefacts (ghost patterns) produce sharp peaks in the synthetic
    spectrum that the real image lacks.  The score is the ratio of the 99th
    percentile of the absolute spectral difference to its mean — higher
    values indicate suspicious repeating structures.
    """
    f_r = np.fft.fftshift(np.fft.fft2(r))
    f_s = np.fft.fftshift(np.fft.fft2(s))

    mag_diff = np.abs(np.abs(f_s) - np.abs(f_r))

    mean_diff = np.mean(mag_diff)
    if mean_diff < 1e-8:
        return 0.0
    return float(np.percentile(mag_diff, 99) / mean_diff)


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  PER-OBJECT METRICS (inside each GT bounding box)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_object_metrics(
    real_gray: np.ndarray,
    synth_gray: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    border_px: int = 15,
) -> Dict[str, float]:
    """
    Compute quality metrics for a single GT bounding box region.

    Compares the crop from the synthetic image to the same crop from
    the real thermal reference.

    Args:
        real_gray, synth_gray: Full images, uint8 H x W.
        x1, y1, x2, y2: Pixel bbox in xyxy format.
        border_px: Extra border for background computation.

    Returns:
        Dict of metric_name -> value.
    """
    nan_result = {
        "obj_ssim":           float("nan"),
        "obj_psnr":           float("nan"),
        "obj_thermal_err":    float("nan"),
        "obj_edge_diff":      float("nan"),
        "obj_blur_diff":      float("nan"),
        "obj_fg_bg_contrast_real": float("nan"),
        "obj_fg_bg_contrast_synth": float("nan"),
        "obj_contrast_preservation": float("nan"),
    }

    h, w = real_gray.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if (x2 - x1) < 4 or (y2 - y1) < 4:
        return nan_result

    crop_r = real_gray[y1:y2, x1:x2]
    crop_s = synth_gray[y1:y2, x1:x2]

    if crop_r.size == 0 or crop_s.size == 0:
        return nan_result

    # 1. Local SSIM
    win = min(7, crop_r.shape[0], crop_r.shape[1])
    if win < 3:
        obj_ssim = float("nan")
    else:
        if win % 2 == 0:
            win -= 1
        obj_ssim = float(ssim(crop_r, crop_s, win_size=win))

    # 2. Local PSNR
    mse = np.mean((crop_r.astype(np.float64) - crop_s.astype(np.float64)) ** 2)
    obj_psnr = 10.0 * np.log10(255.0 ** 2 / mse) if mse > 0 else 100.0

    # 3. Thermal signature error (mean intensity difference inside bbox)
    obj_thermal_err = float(np.abs(
        np.mean(crop_s.astype(np.float64)) - np.mean(crop_r.astype(np.float64))
    ))

    # 4. Edge strength difference (Sobel)
    def _edge_strength(crop):
        sx = cv2.Sobel(crop, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(crop, cv2.CV_64F, 0, 1, ksize=3)
        return float(np.mean(np.sqrt(sx ** 2 + sy ** 2)))

    es_r = _edge_strength(crop_r)
    es_s = _edge_strength(crop_s)
    obj_edge_diff = es_s - es_r   # +ve = synthetic sharper, -ve = blurrier

    # 5. Blur difference (Laplacian variance)
    blur_r = float(np.var(cv2.Laplacian(crop_r, cv2.CV_64F)))
    blur_s = float(np.var(cv2.Laplacian(crop_s, cv2.CV_64F)))
    obj_blur_diff = blur_s - blur_r  # +ve = synthetic sharper

    # 6. Foreground-background contrast preservation
    bx1 = max(0, x1 - border_px)
    by1 = max(0, y1 - border_px)
    bx2 = min(w, x2 + border_px)
    by2 = min(h, y2 + border_px)

    def _fg_bg_contrast(img, bx1, by1, bx2, by2, x1, y1, x2, y2):
        bg_region = img[by1:by2, bx1:bx2].copy()
        mask = np.ones(bg_region.shape, dtype=bool)
        # zero out the foreground area in the mask
        fy1, fy2 = y1 - by1, y2 - by1
        fx1, fx2 = x1 - bx1, x2 - bx1
        mask[fy1:fy2, fx1:fx2] = False
        bg_px = bg_region[mask]
        if bg_px.size == 0:
            return float("nan")
        fg_mean = float(np.mean(img[y1:y2, x1:x2].astype(np.float64)))
        bg_mean = float(np.mean(bg_px.astype(np.float64)))
        return abs(fg_mean - bg_mean)

    fg_bg_r = _fg_bg_contrast(real_gray, bx1, by1, bx2, by2, x1, y1, x2, y2)
    fg_bg_s = _fg_bg_contrast(synth_gray, bx1, by1, bx2, by2, x1, y1, x2, y2)

    if not np.isnan(fg_bg_r) and not np.isnan(fg_bg_s):
        contrast_pres = fg_bg_s / fg_bg_r if fg_bg_r > 0 else 1.0
    else:
        contrast_pres = float("nan")

    return {
        "obj_ssim":              round(obj_ssim, 5) if not np.isnan(obj_ssim) else float("nan"),
        "obj_psnr":              round(obj_psnr, 3),
        "obj_thermal_err":       round(obj_thermal_err, 3),
        "obj_edge_diff":         round(obj_edge_diff, 4),
        "obj_blur_diff":         round(obj_blur_diff, 4),
        "obj_fg_bg_contrast_real":  round(fg_bg_r, 3) if not np.isnan(fg_bg_r) else float("nan"),
        "obj_fg_bg_contrast_synth": round(fg_bg_s, 3) if not np.isnan(fg_bg_s) else float("nan"),
        "obj_contrast_preservation": round(contrast_pres, 5) if not np.isnan(contrast_pres) else float("nan"),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  GT LABEL LOADING & HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def load_yolo_labels(label_path: str) -> List[Tuple[int, float, float, float, float]]:
    """Load YOLO-format labels. Returns list of (cls, cx, cy, w, h) normalised."""
    if not os.path.exists(label_path):
        return []
    labels = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                labels.append((
                    int(parts[0]),
                    float(parts[1]), float(parts[2]),
                    float(parts[3]), float(parts[4]),
                ))
    return labels


def yolo_to_xyxy(cx, cy, w, h, img_size=512):
    x1 = (cx - w / 2) * img_size
    y1 = (cy - h / 2) * img_size
    x2 = (cx + w / 2) * img_size
    y2 = (cy + h / 2) * img_size
    return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))


def labels_to_mask(labels, img_size=512) -> np.ndarray:
    """Create a binary mask from YOLO labels (1 = person region)."""
    mask = np.zeros((img_size, img_size), dtype=np.uint8)
    for _, cx, cy, w, h in labels:
        x1, y1, x2, y2 = yolo_to_xyxy(cx, cy, w, h, img_size)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_size, x2), min(img_size, y2)
        mask[y1:y2, x1:x2] = 1
    return mask


def classify_day_night(flat_name: str) -> str:
    """Extract set_id from flat_name and classify day/night."""
    set_id = flat_name.split("_")[0]   # e.g.  "set00"
    if set_id in DAY_SETS:
        return "day"
    elif set_id in NIGHT_SETS:
        return "night"
    return "unknown"


# Size thresholds (COCO-style, by pixel height)
SIZE_SMALL_MAX  = 32
SIZE_MEDIUM_MAX = 96

def size_category(h_px: int) -> str:
    if h_px <= SIZE_SMALL_MAX:
        return "small"
    elif h_px <= SIZE_MEDIUM_MAX:
        return "medium"
    return "large"


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  MAIN EVALUATION LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_modality(
    modality: str,
    split: str = "val",
    max_images: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the full paired evaluation for one synthetic modality.

    Args:
        modality:   Name of the synthetic modality folder under images/.
        split:      "val" or "train".
        max_images: If > 0, limit to this many images (for quick testing).

    Returns:
        (df_image, df_object)  DataFrames of per-image and per-object metrics.
    """
    synth_dir  = os.path.join(IMAGES_ROOT, modality, split)
    real_dir   = os.path.join(THERMAL_DIR, split)

    # Labels live at labels/{split}/ (shared across modalities).
    # Fall back to labels/{modality}/{split}/ if the root one doesn't exist.
    labels_dir = os.path.join(LABELS_ROOT, split)
    if not os.path.exists(labels_dir):
        labels_dir = os.path.join(LABELS_ROOT, modality, split)

    if not os.path.exists(synth_dir):
        print(f"ERROR: Synthetic directory not found: {synth_dir}")
        sys.exit(1)
    if not os.path.exists(real_dir):
        print(f"ERROR: Real thermal directory not found: {real_dir}")
        sys.exit(1)

    # Collect paired filenames (only .jpg)
    synth_files = {f for f in os.listdir(synth_dir) if f.lower().endswith(".jpg")}
    real_files  = {f for f in os.listdir(real_dir) if f.lower().endswith(".jpg")}
    paired = sorted(synth_files & real_files)

    if max_images > 0:
        paired = paired[:max_images]

    print(f"\n{'='*70}")
    print(f"  Evaluating: {display_name(modality)}  ({split})")
    print(f"  Paired images: {len(paired)}")
    print(f"  Synthetic dir: {synth_dir}")
    print(f"  Reference dir: {real_dir}")
    print(f"  Labels dir:    {labels_dir}")
    print(f"{'='*70}\n")

    image_rows: List[Dict] = []
    object_rows: List[Dict] = []

    from tqdm import tqdm

    for fname in tqdm(paired, desc=f"  {display_name(modality)}", unit="img"):
        flat_name = os.path.splitext(fname)[0]
        day_night = classify_day_night(flat_name)

        # Load images
        real_path  = os.path.join(real_dir, fname)
        synth_path = os.path.join(synth_dir, fname)

        real_img  = cv2.imread(real_path, cv2.IMREAD_GRAYSCALE)
        synth_img = cv2.imread(synth_path, cv2.IMREAD_GRAYSCALE)

        if real_img is None or synth_img is None:
            continue

        # Ensure same size
        if real_img.shape != synth_img.shape:
            synth_img = cv2.resize(synth_img, (real_img.shape[1], real_img.shape[0]))

        # Load GT labels for person mask & per-object analysis
        label_path = os.path.join(labels_dir, f"{flat_name}.txt")
        gt_labels  = load_yolo_labels(label_path)
        person_mask = labels_to_mask(gt_labels, img_size=real_img.shape[0])

        # ── Image-level metrics ──
        img_metrics = compute_image_metrics(
            real_img, synth_img,
            gt_masks=person_mask if gt_labels else None,
        )
        img_metrics["flat_name"]  = flat_name
        img_metrics["modality"]   = modality
        img_metrics["split"]      = split
        img_metrics["day_night"]  = day_night
        img_metrics["num_people"] = len(gt_labels)

        # Extra image-level stats for cross-analysis
        img_metrics["real_brightness"]  = round(float(np.mean(real_img)), 2)
        img_metrics["synth_brightness"] = round(float(np.mean(synth_img)), 2)
        img_metrics["real_contrast"]    = round(float(np.std(real_img)), 2)
        img_metrics["synth_contrast"]   = round(float(np.std(synth_img)), 2)

        image_rows.append(img_metrics)

        # ── Per-object metrics ──
        img_h, img_w = real_img.shape[:2]
        for gi, (cls, cx, cy, w, h) in enumerate(gt_labels):
            x1, y1, x2, y2 = yolo_to_xyxy(cx, cy, w, h, img_w)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w, x2), min(img_h, y2)
            bbox_h = y2 - y1

            obj_metrics = compute_object_metrics(
                real_img, synth_img, x1, y1, x2, y2,
            )
            obj_metrics["flat_name"]     = flat_name
            obj_metrics["modality"]      = modality
            obj_metrics["day_night"]     = day_night
            obj_metrics["gt_idx"]        = gi
            obj_metrics["bbox_height_px"] = bbox_h
            obj_metrics["bbox_area_px"]  = (x2 - x1) * (y2 - y1)
            obj_metrics["size_category"] = size_category(bbox_h)

            object_rows.append(obj_metrics)

    df_img = pd.DataFrame(image_rows)
    df_obj = pd.DataFrame(object_rows)

    # Save CSVs
    os.makedirs(CSV_DIR, exist_ok=True)
    img_csv = os.path.join(CSV_DIR, f"image_quality_{modality}_{split}.csv")
    obj_csv = os.path.join(CSV_DIR, f"object_quality_{modality}_{split}.csv")
    df_img.to_csv(img_csv, index=False)
    df_obj.to_csv(obj_csv, index=False)
    print(f"\n  Saved: {img_csv}  ({len(df_img)} rows)")
    print(f"  Saved: {obj_csv}  ({len(df_obj)} rows)")

    return df_img, df_obj


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  THESIS PLOTS
# ═══════════════════════════════════════════════════════════════════════════════

def _subset_label(subset: str) -> str:
    return {"all": "All Images", "day": "Day Images", "night": "Night Images"}[subset]


def _filter(df: pd.DataFrame, subset: str) -> pd.DataFrame:
    if subset == "all":
        return df
    return df[df["day_night"] == subset]


# ── 5.1  Distribution histograms per metric (ALL / DAY / NIGHT) ─────────────

def plot_metric_distributions(df_img: pd.DataFrame, modality: str):
    """
    For each key image metric, plot distribution histograms split by
    day/night and for all images.
    """
    metrics = [
        ("ssim",              "SSIM",                 (0, 1)),
        ("psnr",              "PSNR (dB)",            None),
        ("mae",               "MAE",                  None),
        ("edge_coherence",    "Edge Coherence",       (-0.2, 1)),
        ("ghost_score",       "Ghost Score",          None),
        ("hallucination_score", "Hallucination Score", None),
        ("brightness_shift",  "Brightness Shift",     None),
        ("contrast_ratio",    "Contrast Ratio",       None),
    ]

    for metric_col, metric_label, xlim in metrics:
        if metric_col not in df_img.columns:
            continue

        fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=True)
        for ax, subset in zip(axes, ["all", "day", "night"]):
            data = _filter(df_img, subset)[metric_col].dropna()
            if data.empty:
                ax.set_title(f"{_subset_label(subset)}\n(no data)")
                continue

            ax.hist(data, bins=50, color=PALETTE[0], edgecolor="white",
                    alpha=0.85, density=True)
            ax.axvline(data.median(), color=PALETTE[3], ls="--", lw=1.5,
                       label=f"median={data.median():.3f}")
            ax.set_title(_subset_label(subset), fontweight="bold")
            ax.set_xlabel(metric_label)
            ax.legend(fontsize=9)
            if xlim:
                ax.set_xlim(xlim)

        axes[0].set_ylabel("Density")
        fig.suptitle(
            f"{metric_label} Distribution  —  {display_name(modality)} vs Real Thermal",
            fontsize=14, fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        save_fig(fig, f"{modality}_dist_{metric_col}")


# ── 5.2  Box-plots comparing day vs night per metric ────────────────────────

def plot_day_night_boxplots(df_img: pd.DataFrame, modality: str):
    """
    Side-by-side box plots: day vs night for each key metric.
    """
    metrics = ["ssim", "psnr", "mae", "edge_coherence",
               "ghost_score", "hallucination_score"]
    available = [m for m in metrics if m in df_img.columns]

    n = len(available)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
    axes = np.array(axes).flatten()

    for i, metric in enumerate(available):
        ax = axes[i]
        data = df_img[["day_night", metric]].dropna()
        order = ["day", "night"]
        existing = [o for o in order if o in data["day_night"].values]
        sns.boxplot(data=data, x="day_night", y=metric, hue="day_night",
                    order=existing, ax=ax, palette=PALETTE[:2],
                    width=0.5, legend=False)
        ax.set_title(metric.replace("_", " ").title(), fontweight="bold")
        ax.set_xlabel("")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"Day vs Night Quality Comparison  —  {display_name(modality)}",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_fig(fig, f"{modality}_daynight_boxplots")


# ── 5.3  Per-object SSIM / blur by size category ────────────────────────────

def plot_object_quality_by_size(df_obj: pd.DataFrame, modality: str):
    """
    Box-plots of per-object SSIM and blur difference, grouped by size
    category, for all / day / night.
    """
    obj_metrics = [
        ("obj_ssim",       "Object SSIM"),
        ("obj_thermal_err", "Thermal Signature Error"),
        ("obj_blur_diff", "Blur Difference (synth-real)"),
        ("obj_edge_diff", "Edge Strength Diff (synth-real)"),
    ]
    available = [(c, l) for c, l in obj_metrics if c in df_obj.columns]
    if not available:
        return

    for subset in ["all", "day", "night"]:
        data = _filter(df_obj, subset)
        if data.empty:
            continue

        fig, axes = plt.subplots(1, len(available),
                                 figsize=(5 * len(available), 4.5))
        if len(available) == 1:
            axes = [axes]

        size_order = ["small", "medium", "large"]
        for ax, (col, label) in zip(axes, available):
            plot_data = data[["size_category", col]].dropna()
            existing = [s for s in size_order if s in plot_data["size_category"].values]
            if plot_data.empty:
                ax.set_title(f"{label}\n(no data)")
                continue
            sns.boxplot(data=plot_data, x="size_category", y=col,
                        hue="size_category", order=existing, ax=ax,
                        palette="viridis", width=0.5, legend=False)
            ax.set_title(label, fontweight="bold")
            ax.set_xlabel("Object Size")

        fig.suptitle(
            f"Object Quality by Size  —  {display_name(modality)}  ({_subset_label(subset)})",
            fontsize=13, fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        save_fig(fig, f"{modality}_obj_by_size_{subset}")


# ── 5.4  Scatter: SSIM vs brightness / contrast ─────────────────────────────

def plot_ssim_vs_image_properties(df_img: pd.DataFrame, modality: str):
    """
    Scatter plots showing how SSIM relates to scene brightness and contrast.
    This reveals whether the model struggles in dark or low-contrast scenes.
    """
    props = [
        ("real_brightness",  "Real Thermal Brightness (mean)"),
        ("real_contrast",    "Real Thermal Contrast (std)"),
    ]
    available = [(c, l) for c, l in props if c in df_img.columns]
    if "ssim" not in df_img.columns or not available:
        return

    for subset in ["all", "day", "night"]:
        data = _filter(df_img, subset)
        if data.empty:
            continue

        fig, axes = plt.subplots(1, len(available),
                                 figsize=(6 * len(available), 5))
        if len(available) == 1:
            axes = [axes]

        for ax, (col, label) in zip(axes, available):
            plot_data = data[[col, "ssim"]].dropna()
            ax.scatter(plot_data[col], plot_data["ssim"],
                       alpha=0.15, s=8, color=PALETTE[0], rasterized=True)
            # Trend line
            if len(plot_data) > 10:
                z = np.polyfit(plot_data[col], plot_data["ssim"], 1)
                p = np.poly1d(z)
                xs = np.linspace(plot_data[col].min(), plot_data[col].max(), 100)
                ax.plot(xs, p(xs), color=PALETTE[3], lw=2,
                        label=f"trend (slope={z[0]:.4f})")
                ax.legend(fontsize=9)
            ax.set_xlabel(label)
            ax.set_ylabel("SSIM")
            ax.set_title(f"SSIM vs {label.split('(')[0].strip()}", fontweight="bold")

        fig.suptitle(
            f"SSIM vs Scene Properties  —  {display_name(modality)}  ({_subset_label(subset)})",
            fontsize=13, fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        save_fig(fig, f"{modality}_ssim_vs_props_{subset}")


# ── 5.5  Hallucination & Ghost heatmap: crowd density × contrast ────────────

def plot_hallucination_heatmap(df_img: pd.DataFrame, modality: str):
    """
    2-D binned heatmap: crowd density × real contrast → mean hallucination
    score.  Directly answers: "Do crowded, low-contrast scenes produce
    more hallucinated artefacts?"
    """
    if "hallucination_score" not in df_img.columns:
        return

    for metric_col, metric_label in [
        ("hallucination_score", "Mean Hallucination Score"),
        ("ghost_score", "Mean Ghost Score"),
    ]:
        if metric_col not in df_img.columns:
            continue

        for subset in ["all", "day", "night"]:
            data = _filter(df_img, subset).copy()
            if data.empty or len(data) < 20:
                continue

            # Bin crowd density and contrast
            data["crowd_bin"] = pd.cut(
                data["num_people"],
                bins=[-0.5, 0.5, 2.5, 5.5, 100],
                labels=["0", "1-2", "3-5", "6+"],
            )
            data["contrast_bin"] = pd.qcut(
                data["real_contrast"], q=4,
                labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"],
                duplicates="drop",
            )

            pivot = data.groupby(["crowd_bin", "contrast_bin"])[metric_col].mean()
            pivot = pivot.unstack(fill_value=float("nan"))

            if pivot.empty:
                continue

            fig, ax = plt.subplots(figsize=(8, 5))
            sns.heatmap(
                pivot, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax,
                linewidths=0.5, cbar_kws={"label": metric_label},
            )
            ax.set_xlabel("Real Thermal Contrast Quartile")
            ax.set_ylabel("People per Frame")
            ax.set_title(
                f"{metric_label}  —  {display_name(modality)}  ({_subset_label(subset)})",
                fontweight="bold",
            )
            fig.tight_layout()
            save_fig(fig, f"{modality}_heatmap_{metric_col}_{subset}")


# ── 5.6  Edge coherence by day/night AND size ───────────────────────────────

def plot_edge_coherence_combined(df_img: pd.DataFrame, df_obj: pd.DataFrame,
                                 modality: str):
    """
    Combined plot: left = image-level edge coherence by day/night,
    right = per-object edge diff by size × day/night.
    """
    if "edge_coherence" not in df_img.columns:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: image-level
    data1 = df_img[["day_night", "edge_coherence"]].dropna()
    order = ["day", "night"]
    existing = [o for o in order if o in data1["day_night"].values]
    if not data1.empty:
        sns.violinplot(data=data1, x="day_night", y="edge_coherence",
                       hue="day_night", order=existing, ax=ax1,
                       palette=PALETTE[:2], inner="quartile", cut=0,
                       legend=False)
        ax1.set_title("Image-Level Edge Coherence", fontweight="bold")
        ax1.set_ylabel("Edge Coherence (Canny correlation)")
        ax1.set_xlabel("")

    # Right: per-object edge difference by size
    if "obj_edge_diff" in df_obj.columns:
        data2 = df_obj[["size_category", "day_night", "obj_edge_diff"]].dropna()
        if not data2.empty:
            sns.boxplot(data=data2, x="size_category", y="obj_edge_diff",
                        hue="day_night", order=["small", "medium", "large"],
                        ax=ax2, palette=PALETTE[:2], width=0.6)
            ax2.axhline(0, color="gray", ls=":", lw=1)
            ax2.set_title("Per-Object Edge Strength Difference", fontweight="bold")
            ax2.set_ylabel("Edge Diff (synth - real)")
            ax2.set_xlabel("Object Size")
            ax2.legend(title="", loc="upper right")

    fig.suptitle(
        f"Edge Preservation Analysis  —  {display_name(modality)}",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_fig(fig, f"{modality}_edge_analysis")


# ── 5.7  Thermal signature error by size + day/night ────────────────────────

def plot_thermal_signature(df_obj: pd.DataFrame, modality: str):
    """
    How much does the synthetic image deviate in mean thermal intensity
    within each person bbox, broken down by size and day/night?
    """
    if "obj_thermal_err" not in df_obj.columns:
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    for ax, subset in zip(axes, ["all", "day", "night"]):
        data = _filter(df_obj, subset)
        if data.empty:
            ax.set_title(f"{_subset_label(subset)}\n(no data)")
            continue

        plot_data = data[["size_category", "obj_thermal_err"]].dropna()
        order = ["small", "medium", "large"]
        existing = [s for s in order if s in plot_data["size_category"].values]
        sns.boxplot(data=plot_data, x="size_category", y="obj_thermal_err",
                    hue="size_category", order=existing, ax=ax,
                    palette="magma", width=0.5, legend=False)
        ax.set_title(_subset_label(subset), fontweight="bold")
        ax.set_xlabel("Object Size")

    axes[0].set_ylabel("Thermal Signature Error (mean |I_synth - I_real|)")
    fig.suptitle(
        f"Thermal Signature Accuracy per Object  —  {display_name(modality)}",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_fig(fig, f"{modality}_thermal_signature")


# ── 5.8  Contrast preservation scatter ──────────────────────────────────────

def plot_contrast_preservation(df_obj: pd.DataFrame, modality: str):
    """
    Scatter of real fg-bg contrast vs synthetic fg-bg contrast per object.
    Perfect preservation = y=x line.
    """
    if "obj_fg_bg_contrast_real" not in df_obj.columns:
        return

    for subset in ["all", "day", "night"]:
        data = _filter(df_obj, subset)[
            ["obj_fg_bg_contrast_real", "obj_fg_bg_contrast_synth", "size_category"]
        ].dropna()

        if data.empty or len(data) < 10:
            continue

        fig, ax = plt.subplots(figsize=(7, 6))
        size_colors = {"small": PALETTE[0], "medium": PALETTE[1], "large": PALETTE[2]}

        for cat in ["small", "medium", "large"]:
            sub = data[data["size_category"] == cat]
            if sub.empty:
                continue
            ax.scatter(
                sub["obj_fg_bg_contrast_real"],
                sub["obj_fg_bg_contrast_synth"],
                alpha=0.2, s=10, label=cat,
                color=size_colors.get(cat, "gray"),
                rasterized=True,
            )

        lim = max(data["obj_fg_bg_contrast_real"].max(),
                  data["obj_fg_bg_contrast_synth"].max()) * 1.05
        ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.5, label="perfect")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_xlabel("Real Thermal FG-BG Contrast")
        ax.set_ylabel("Synthetic FG-BG Contrast")
        ax.set_title(
            f"FG-BG Contrast Preservation  —  {display_name(modality)}  "
            f"({_subset_label(subset)})",
            fontweight="bold",
        )
        ax.legend(title="Size")
        ax.set_aspect("equal")
        fig.tight_layout()
        save_fig(fig, f"{modality}_contrast_preservation_{subset}")


# ── 5.9  Summary bar chart (mean metric by day / night) ─────────────────────

def plot_summary_bars(df_img: pd.DataFrame, modality: str):
    """
    Grouped bar chart: mean of each key metric for day vs night vs all.
    Provides a quick bird's-eye overview for the thesis.
    """
    metrics = ["ssim", "psnr", "edge_coherence"]
    available = [m for m in metrics if m in df_img.columns]
    if not available:
        return

    rows = []
    for subset in ["all", "day", "night"]:
        data = _filter(df_img, subset)
        if data.empty:
            continue
        row = {"subset": _subset_label(subset)}
        for m in available:
            row[m] = data[m].mean()
        rows.append(row)

    if not rows:
        return

    df_bar = pd.DataFrame(rows).set_index("subset")

    fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 4.5))
    if len(available) == 1:
        axes = [axes]

    nice_names = {
        "ssim": "SSIM", "psnr": "PSNR (dB)",
        "edge_coherence": "Edge Coherence",
    }
    colors = [PALETTE[4], PALETTE[0], PALETTE[1]]

    for ax, m in zip(axes, available):
        vals = df_bar[m]
        bars = ax.bar(vals.index, vals.values,
                      color=colors[:len(vals)], edgecolor="white", width=0.55)
        for bar, v in zip(bars, vals.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.3f}", ha="center", va="bottom", fontsize=10)
        ax.set_ylabel(nice_names.get(m, m))
        ax.set_title(nice_names.get(m, m), fontweight="bold")
        ax.set_ylim(0, vals.max() * 1.15)

    fig.suptitle(
        f"Quality Summary  —  {display_name(modality)} vs Real Thermal",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_fig(fig, f"{modality}_summary_bars")


# ── 5.10 Multi-modality comparison (when multiple modalities evaluated) ─────

def plot_multi_modality_comparison(all_img_dfs: Dict[str, pd.DataFrame]):
    """
    If multiple modalities were evaluated, produce a single comparison figure.
    """
    if len(all_img_dfs) < 2:
        return

    metrics = ["ssim", "psnr", "mae", "edge_coherence",
               "ghost_score", "hallucination_score"]

    rows = []
    for mod, df in all_img_dfs.items():
        for subset in ["all", "day", "night"]:
            data = _filter(df, subset)
            if data.empty:
                continue
            row = {"modality": display_name(mod), "subset": _subset_label(subset)}
            for m in metrics:
                if m in data.columns:
                    row[m] = data[m].mean()
            rows.append(row)

    df_cmp = pd.DataFrame(rows)
    available = [m for m in metrics if m in df_cmp.columns]
    n = len(available)
    cols = 3
    nrows = (n + cols - 1) // cols

    fig, axes = plt.subplots(nrows, cols, figsize=(6 * cols, 5 * nrows))
    axes_flat = np.array(axes).flatten()

    for i, m in enumerate(available):
        ax = axes_flat[i]
        pivot = df_cmp.pivot(index="modality", columns="subset", values=m)
        pivot = pivot.reindex(columns=["All Images", "Day Images", "Night Images"])
        pivot.plot(kind="bar", ax=ax, rot=0, edgecolor="white", width=0.7)
        ax.set_title(m.replace("_", " ").title(), fontweight="bold")
        ax.set_xlabel("")
        ax.legend(fontsize=8, title="")

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        "Multi-Modality Quality Comparison vs Real Thermal",
        fontsize=15, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_fig(fig, "multi_modality_comparison")


# ── 5.11 Print textual summary ──────────────────────────────────────────────

def print_summary_table(df_img: pd.DataFrame, modality: str):
    metrics = ["ssim", "psnr", "mae", "edge_coherence",
               "ghost_score", "hallucination_score",
               "brightness_shift", "contrast_ratio"]

    print(f"\n{'='*70}")
    print(f"  SUMMARY: {display_name(modality)} vs Real Thermal")
    print(f"{'='*70}")
    print(f"  {'Metric':<25s} {'All':>10s} {'Day':>10s} {'Night':>10s}")
    print(f"  {'-'*55}")

    for m in metrics:
        if m not in df_img.columns:
            continue
        vals = {}
        for subset in ["all", "day", "night"]:
            data = _filter(df_img, subset)[m].dropna()
            vals[subset] = f"{data.mean():.4f}" if len(data) > 0 else "n/a"
        print(f"  {m:<25s} {vals['all']:>10s} {vals['day']:>10s} {vals['night']:>10s}")

    print(f"{'='*70}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate synthetic thermal image quality against real thermal references."
    )
    parser.add_argument(
        "--modality", nargs="+", type=str,
        default=["greyscale_inversion", "PI-GAN_gen"],
        help="One or more synthetic modality names (folder names under images/).",
    )
    parser.add_argument(
        "--split", type=str, default="val",
        choices=["val", "train"],
        help="Which split to evaluate (default: val).",
    )
    parser.add_argument(
        "--max-images", type=int, default=0,
        help="Limit evaluation to N images (0 = all). Useful for quick testing.",
    )
    parser.add_argument(
        "--skip-plots", action="store_true",
        help="Only compute CSVs, skip plot generation.",
    )
    args = parser.parse_args()

    all_img_dfs: Dict[str, pd.DataFrame] = {}
    all_obj_dfs: Dict[str, pd.DataFrame] = {}

    for mod in args.modality:
        df_img, df_obj = evaluate_modality(mod, args.split, args.max_images)
        all_img_dfs[mod] = df_img
        all_obj_dfs[mod] = df_obj

        print_summary_table(df_img, mod)

        if not args.skip_plots and not df_img.empty:
            print(f"\n  Generating plots for {display_name(mod)}...")
            plot_metric_distributions(df_img, mod)
            plot_day_night_boxplots(df_img, mod)
            plot_ssim_vs_image_properties(df_img, mod)
            plot_hallucination_heatmap(df_img, mod)
            plot_summary_bars(df_img, mod)

            if not df_obj.empty:
                plot_object_quality_by_size(df_obj, mod)
                plot_edge_coherence_combined(df_img, df_obj, mod)
                plot_thermal_signature(df_obj, mod)
                plot_contrast_preservation(df_obj, mod)

    # Multi-modality comparison (if > 1 modality evaluated)
    if not args.skip_plots and len(all_img_dfs) > 1:
        print("\n  Generating multi-modality comparison...")
        plot_multi_modality_comparison(all_img_dfs)

    print(f"\nDone. Plots saved to: {PLOTS_DIR}")
    print(f"CSVs saved to:  {CSV_DIR}")


if __name__ == "__main__":
    main()
