"""
=============================================================================
eval_plots.py — Thesis-Quality Evaluation Plots
=============================================================================

Generates all plots for the evaluation pipeline. Each function produces
one logical figure that can be directly embedded in the thesis.

All plots use matplotlib with the "Agg" backend (no display window),
saved as PNG at 150 DPI.

Functions:
    plot_quality_vs_performance(dfs, output_dir)
        Binned scatter: quality metric → Recall / F1, one per modality

    plot_sharpness_size_heatmap(df_obj, output_dir)
        Recall heatmap: sharpness level × object size × day/night

    plot_fp_analysis(df_fp, df_img, output_dir)
        FP confidence histogram + FP rate by day/night

    plot_fid_vs_map_bar(fid_results, map_results, output_dir)
        Grouped bar chart: FID and mAP50 per modality

    plot_lpips_vs_confidence(df_obj, output_dir)
        Scatter: per-object LPIPS vs prediction confidence
=============================================================================
"""

import os
import numpy as np
import pandas as pd

# Lazy-import matplotlib to avoid backend issues when imported as module
_plt = None
def _get_plt():
    global _plt
    if _plt is None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        _plt = plt
    return _plt


# Colour palette — one colour per modality
PALETTE = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0"]

# Human-readable descriptions for each metric (shown as subtitle in plots)
METRIC_DESCRIPTIONS = {
    "brightness":
        "Mean pixel intensity across all channels  (raw scale: 0–255; 0=black, 255=white)",
    "contrast":
        "Std deviation of pixel intensities — spread between dark and bright values  (0=flat/uniform, higher=more varied)",
    "sharpness":
        "Laplacian variance of the image luminance channel — higher = more sharp edges present",
    "edge_density":
        "Mean Sobel gradient magnitude — how much edge content the image contains overall",
    "blur":
        "Inverse of sharpness  (= 1 / Laplacian variance) — higher = blurrier image",
    "ghost_score":
        "Mean absolute pixel residual between generated and original thermal crop  (0=identical, 1=completely different)",
    "hallucination_score":
        "Fraction of edges in generated image that do NOT exist in the thermal GT  (0=no invented edges, 1=fully hallucinated)",
    "object_ssim":
        "Structural Similarity Index (SSIM) between generated and thermal GT crop  (0=very different, 1=identical)",
    "lpips_mean":
        "LPIPS: deep perceptual distance between generated and thermal GT  (0=perceptually identical, higher=more different)",
}


def _scale(series, col):
    """Min-max scale a metric column to [0, 1] for plotting."""
    s = series.dropna()
    mn, mx = s.min(), s.max()
    return (s - mn) / (mx - mn + 1e-9)


def _raw_range(series):
    """Return (min, max) of the raw data for axis labelling."""
    s = series.dropna()
    return float(s.min()), float(s.max())


# =============================================================================
# QUALITY VS PERFORMANCE PLOTS
#   For each quality metric, produce a binned scatter plot:
#     X axis: quality metric value (scaled 0–1, 10 equal bins)
#     Y axis: Recall and F1 (two subplots side by side)
#     Colours: one per modality for multi-modality overlay
#
#   Bins with fewer than 3 images are excluded (too noisy).
#   Error bands show ±1 standard deviation within each bin.
# =============================================================================

# Metrics saved to /appendix rather than the main output folder
APPENDIX_QUALITY_COLS = {"brightness", "contrast", "edge_density", "blur"}


def plot_quality_vs_performance(
    dfs: dict,
    output_dir: str,
    appendix_dir: str = None,
):
    """
    Args:
        dfs:        {modality_label: df_img} dict
        output_dir: Where to save the PNG files.
    """
    plt = _get_plt()
    os.makedirs(output_dir, exist_ok=True)

    quality_cols = ["brightness", "contrast", "sharpness", "edge_density", "blur",
                    "ghost_score", "hallucination_score", "object_ssim", "lpips_mean"]
    N_BINS = 10

    for col in quality_cols:
        # Skip if no modality has this column
        if not any(col in df.columns and df[col].notna().any()
                   for df in dfs.values()):
            continue

        # Human-friendly title per metric
        TITLE_MAP = {
            "object_ssim": "SSIM (Structural Similarity) vs Detection Performance",
            "lpips_mean":  "LPIPS (Perceptual Distance) vs Detection Performance",
            "ghost_score": "Ghost Score vs Detection Performance",
            "hallucination_score": "Hallucination Score vs Detection Performance",
            "sharpness":   "Image Sharpness vs Detection Performance",
        }
        main_title = TITLE_MAP.get(
            col,
            f"{col.capitalize().replace('_', ' ')} vs Detection Performance",
        )
        desc = METRIC_DESCRIPTIONS.get(col, "")

        fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), sharey=False)
        fig.suptitle(main_title, fontsize=14, fontweight="bold", y=0.99)
        if desc:
            fig.text(0.5, 0.93, desc, ha="center", fontsize=8.5,
                     color="#555555", style="italic")

        for ax, y_col, y_label in zip(
            axes, ["recall", "f1"], ["Recall", "F1 Score"],
        ):
            raw_min, raw_max = 0.0, 1.0  # fallback if no modality has data
            for (label, df), color in zip(dfs.items(), PALETTE):
                if col not in df.columns or df[col].isna().all():
                    continue
                if y_col not in df.columns:
                    continue

                raw_min, raw_max = _raw_range(df[col])
                scaled = _scale(df[col], col)
                df_plot = df.copy()
                df_plot["_x"] = scaled.reindex(df_plot.index)
                df_plot = df_plot.dropna(subset=["_x", y_col])

                bins = np.linspace(0, 1, N_BINS + 1)
                df_plot["_bin"] = pd.cut(df_plot["_x"], bins=bins,
                                         include_lowest=True)
                grouped = df_plot.groupby("_bin", observed=True)[y_col]
                bin_means = grouped.mean()
                bin_stds = grouped.std().fillna(0)
                bin_centers = [(b.left + b.right) / 2 for b in bin_means.index]
                n_per_bin = grouped.count()

                mask = n_per_bin >= 3
                xs = np.array(bin_centers)[mask]
                ys = bin_means.values[mask]
                errs = bin_stds.values[mask]

                ax.plot(xs, ys, "o-", color=color, label=label,
                        linewidth=2, markersize=5)

            col_label = col.capitalize().replace("_", " ")
            ax.set_xlabel(
                f"{col_label}  (min-max normalised;  raw range [{raw_min:.3g} – {raw_max:.3g}])",
                fontsize=9,
            )
            YLABEL_MAP = {
                "recall": "Recall  (detected GT objects / total GT objects)",
                "f1":     "F1 Score  (harmonic mean of Precision & Recall)",
            }
            ax.set_ylabel(YLABEL_MAP.get(y_col, y_label), fontsize=9)
            ax.set_xlim(0, 1)
            ax.set_ylim(-0.02, 1.08)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)

        save_dir = appendix_dir if (appendix_dir and col in APPENDIX_QUALITY_COLS) else output_dir
        os.makedirs(save_dir, exist_ok=True)
        plt.tight_layout(rect=[0, 0, 1, 0.91])
        out_path = os.path.join(save_dir, f"quality_vs_perf_{col}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out_path}")


# =============================================================================
# SHARPNESS × SIZE HEATMAP
#   Visualises how sharpness interacts with object size to affect recall.
#   One heatmap per condition (Overall, Day, Night).
#
#   Rows: sharpness level (low / high, median split)
#   Columns: size category (small / medium / large)
#   Cell colour: recall value (0–1)
# =============================================================================

def plot_sharpness_size_heatmap(df_obj: pd.DataFrame, output_dir: str):
    """Save one recall heatmap PNG per condition (Overall, Day, Night)."""
    plt = _get_plt()

    if "object_blur" not in df_obj.columns or "size_category" not in df_obj.columns:
        return
    if df_obj["object_blur"].isna().all():
        return

    os.makedirs(output_dir, exist_ok=True)

    median_sharp = df_obj["object_blur"].median()
    df = df_obj.copy()
    df["sharp_level"] = df["object_blur"].apply(
        lambda x: "high" if x >= median_sharp else "low"
    )

    subsets = [("Overall", df)]
    if "day_night" in df.columns:
        subsets += [
            ("Day", df[df["day_night"] == "day"]),
            ("Night", df[df["day_night"] == "night"]),
        ]

    for label, sub in subsets:
        data = np.full((2, 3), np.nan)
        counts = np.zeros((2, 3), dtype=int)
        for ri, slevel in enumerate(["low", "high"]):
            for ci, scat in enumerate(["small", "medium", "large"]):
                s = sub[(sub["sharp_level"] == slevel) & (sub["size_category"] == scat)]
                counts[ri, ci] = len(s)
                if len(s) > 0:
                    data[ri, ci] = s["detected"].mean()

        fig, ax = plt.subplots(figsize=(6, 3.5))
        im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(
            ["Small\n(height ≤ 32 px)", "Medium\n(33 – 96 px)", "Large\n(> 96 px)"],
            fontsize=10,
        )
        ax.set_yticks([0, 1])
        ax.set_yticklabels(
            ["Low sharpness\n(below median)", "High sharpness\n(at/above median)"],
            fontsize=10,
        )
        ax.set_title(f"Recall: Sharpness × Object Size — {label}",
                     fontweight="bold", fontsize=12)
        ax.set_xlabel(
            "Object size  (bounding-box height at 512 × 512 px resolution)",
            fontsize=9, color="#555555",
        )
        ax.set_ylabel(
            "Sharpness level  (Laplacian variance, split at median)",
            fontsize=9, color="#555555",
        )

        for ri in range(2):
            for ci in range(3):
                if not np.isnan(data[ri, ci]):
                    ax.text(ci, ri,
                            f"{data[ri, ci]:.3f}\n(n={counts[ri, ci]})",
                            ha="center", va="center", fontsize=9,
                            color="black")

        cbar = fig.colorbar(im, ax=ax, shrink=0.9)
        cbar.set_label("Recall", fontsize=10)

        plt.tight_layout()
        fname = f"sharpness_size_heatmap_{label.lower()}.png"
        out_path = os.path.join(output_dir, fname)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out_path}")


# =============================================================================
# FALSE POSITIVE ANALYSIS PLOTS
#   1. Confidence histogram of FPs (coloured by low/high confidence)
#   2. FP rate by day/night (bar chart)
# =============================================================================

def plot_fp_analysis(df_fp: pd.DataFrame, df_img: pd.DataFrame, output_dir: str):
    """Generate FP rate by day/night bar chart."""
    plt = _get_plt()

    if df_fp is None or len(df_fp) == 0:
        return
    if "day_night" not in df_img.columns:
        return

    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 4.5))

    rates, labels, fp_counts, pred_totals = [], [], [], []
    for dn in ["day", "night"]:
        sub = df_img[df_img["day_night"] == dn]
        if len(sub) == 0:
            continue
        tp = int(sub["tp"].sum())
        fp = int(sub["fp"].sum())
        rate = fp / (tp + fp) if (tp + fp) > 0 else 0
        rates.append(rate)
        labels.append(dn.capitalize())
        fp_counts.append(fp)
        pred_totals.append(tp + fp)

    if rates:
        colors = ["#4CAF50", "#3F51B5"][:len(rates)]
        ax.bar(labels, rates, color=colors, width=0.5)
        ax.set_ylabel("FP Rate = FP / (TP + FP)")
        ax.set_title("False Positive Rate by Day/Night")
        ax.set_ylim(0, max(rates) * 1.4 if rates else 1)
        ax.grid(True, alpha=0.3, axis="y")
        for i, (r, fp, tot) in enumerate(zip(rates, fp_counts, pred_totals)):
            ax.text(i, r + 0.005, f"{r:.3f}\n({fp}/{tot})", ha="center", fontsize=10)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "fp_analysis.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# =============================================================================
# FID VS mAP BAR CHART
#   Grouped bar chart comparing FID and mAP50 across modalities.
#   Dual Y-axis: left = FID (lower is better), right = mAP50 (higher is better).
# =============================================================================

def plot_fid_vs_map_bar(
    fid_results: dict,
    map_results: dict,
    output_dir: str,
):
    """
    Args:
        fid_results: {modality: fid_value}
        map_results: {modality: mAP50_value}
        output_dir:  Save path.
    """
    plt = _get_plt()
    os.makedirs(output_dir, exist_ok=True)

    modalities = sorted(set(fid_results.keys()) | set(map_results.keys()))
    if not modalities:
        return

    x = np.arange(len(modalities))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    fid_vals = [fid_results.get(m, 0) for m in modalities]
    map_vals = [map_results.get(m, 0) for m in modalities]

    ax1.bar(x - width/2, fid_vals, width, label="FID (lower=better)",
            color="#F44336", alpha=0.8)
    ax2.bar(x + width/2, map_vals, width, label="mAP50 (higher=better)",
            color="#2196F3", alpha=0.8)

    ax1.set_xlabel("Modality")
    ax1.set_ylabel("FID", color="#F44336")
    ax2.set_ylabel("mAP50", color="#2196F3")
    ax1.set_xticks(x)
    ax1.set_xticklabels(modalities)
    ax1.tick_params(axis="y", labelcolor="#F44336")
    ax2.tick_params(axis="y", labelcolor="#2196F3")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.title("FID vs mAP50 by Modality", fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(output_dir, "fid_vs_map.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# =============================================================================
# LPIPS VS CONFIDENCE SCATTER
#   Shows whether perceptual quality (LPIPS) relates to detector confidence.
#   Each point = one detected object.
#   X = object LPIPS, Y = prediction confidence.
# =============================================================================

def plot_lpips_vs_confidence(df_obj: pd.DataFrame, output_dir: str):
    """Scatter plot of per-object LPIPS vs prediction confidence, saved to output_dir."""
    plt = _get_plt()
    from scipy import stats as _stats

    if "object_lpips" not in df_obj.columns or "tp_conf" not in df_obj.columns:
        return

    df = df_obj.dropna(subset=["object_lpips", "tp_conf"])
    df = df[df["detected"] == True]
    if len(df) < 10:
        return

    os.makedirs(output_dir, exist_ok=True)

    r, p = _stats.pearsonr(df["object_lpips"], df["tp_conf"])
    sig_str = "p < 0.05" if p < 0.05 else f"p = {p:.3f}"

    fig, ax = plt.subplots(figsize=(7, 5.4))
    fig.suptitle("LPIPS vs Detector Confidence", fontsize=13, fontweight="bold", y=0.99)
    fig.text(
        0.5, 0.93,
        "LPIPS = deep perceptual distance between generated image and thermal GT  "
        "(0 = identical, higher = more different)\n"
        "Confidence = YOLO prediction score for each correctly detected object  (0–1)",
        ha="center", fontsize=8.5, color="#555555", style="italic",
    )

    ax.scatter(df["object_lpips"], df["tp_conf"], alpha=0.25, s=8,
               color="#2196F3", edgecolors="none", label=f"Detected objects  (n={len(df):,})")

    # Trend line
    z = np.polyfit(df["object_lpips"], df["tp_conf"], 1)
    p_fn = np.poly1d(z)
    xs = np.linspace(df["object_lpips"].min(), df["object_lpips"].max(), 200)
    ax.plot(xs, p_fn(xs), "r--", linewidth=2,
            label=f"Trend  (slope = {z[0]:.3f})")

    # Annotate r
    ax.text(0.97, 0.97, f"Pearson r = {r:+.3f}\n{sig_str}",
            transform=ax.transAxes, fontsize=10, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#aaaaaa"))

    ax.set_xlabel(
        "LPIPS  (per-object; 0 = identical to thermal GT, higher = more perceptually different)",
        fontsize=9,
    )
    ax.set_ylabel("YOLO Prediction Confidence  (0 = uncertain, 1 = fully confident)", fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.91])
    out_path = os.path.join(output_dir, "lpips_vs_confidence.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# =============================================================================
# EXAMPLE CROP GALLERY
#   For each quality metric, save a strip of 3 annotated bounding-box crops
#   chosen from a "decent" extreme of the metric (not the absolute worst
#   outlier — the 5th–20th or 80th–95th percentile range).
#
#   For metrics that compare against a thermal GT (ghost_score, SSIM, LPIPS,
#   hallucination_score), the thermal GT crop is shown alongside if
#   thermal_dir is supplied.
#
#   Output: one PNG per case in  output_dir/
#     ghost_score_high.png, hallucination_high.png,
#     sharpness_low.png,
#     ssim_low.png / ssim_high.png,
#     lpips_high.png / lpips_low.png,
#     fg_bg_diff_low.png / fg_bg_diff_high.png,
#     contrast_low.png / contrast_high.png,
#     edge_strength_low.png / edge_strength_high.png
# =============================================================================

def plot_example_crops(
    df_obj: pd.DataFrame,
    image_dir: str,
    output_dir: str,
    thermal_dir: str = None,
    is_synthetic: bool = True,
    n_examples: int = 3,
    pad_px: int = 50,
    img_size: int = 512,
    report_dir: str = None,
):
    """
    Save annotated bounding-box crop strips for thesis quality-metric examples.

    Args:
        df_obj:      Per-object DataFrame (must contain flat_name, gt_cx/cy/w/h,
                     and the quality metric columns).
        image_dir:   Directory holding the modality's validation images ({flat_name}.jpg).
        output_dir:  Where to save the example PNGs.
        thermal_dir: Optional directory with thermal GT images ({flat_name}.jpg).
                     Used for side-by-side comparison on ghost/SSIM/LPIPS crops.
        n_examples:  Number of crop examples per strip (default 3).
        pad_px:      Extra padding around the bounding box in the crop (pixels).
        img_size:    Image width/height in pixels (default 512).
    """
    import cv2 as _cv2
    from matplotlib.patches import Rectangle as _Rect
    plt = _get_plt()

    os.makedirs(output_dir, exist_ok=True)

    # ── Cases: (column, extreme, percentile_range, file_stem, title) ──────
    # percentile_range selects "decent extreme" — avoids absolute outliers.
    # sharpness uses 5–20 so it's blurry but still shows a recognisable person.
    CASES = [
        ("ghost_score",          "high", (85, 97),
         "ghost_score_high",
         "Ghost Score — HIGH\n"
         "These pedestrian crops have the most pixel-level artefact between\n"
         "the generated image and the original thermal GT (0=identical, 1=completely different)."),

        ("hallucination_score",  "high", (85, 97),
         "hallucination_high",
         "Hallucination Score — HIGH\n"
         "The generated crops contain the most invented edges not present in the thermal GT.\n"
         "0% = no invented edges,  100% = all edges are hallucinated by the model."),

        ("object_blur",          "low",  (20, 35),
         "sharpness_low",
         "Sharpness — LOW  (20th–35th percentile; noticeably blurry but still visible)\n"
         "Sharpness = Laplacian variance of the object luminance crop.\n"
         "Low value → smooth, blurry object; fewer edges for the detector to use."),

        ("object_ssim",          "low",  (15, 30),
         "ssim_low",
         "SSIM — LOW  (15th–30th percentile)\n"
         "These crops are structurally least similar to the thermal GT.\n"
         "SSIM = Structural Similarity Index, 0=different, 1=identical."),

        ("object_ssim",          "high", (80, 95),
         "ssim_high",
         "SSIM — HIGH\n"
         "These crops are structurally most similar to the thermal GT.\n"
         "SSIM = Structural Similarity Index, 0=different, 1=identical."),

        ("object_lpips",         "high", (65, 80),
         "lpips_high",
         "LPIPS — HIGH  (65th–80th percentile; perceptually different from thermal GT)\n"
         "LPIPS uses deep neural-network features to measure perceptual distance.\n"
         "0 = looks identical, higher = looks more different."),

        ("object_lpips",         "low",  (5, 15),
         "lpips_low",
         "LPIPS — LOW  (perceptually closest to thermal GT)\n"
         "These crops look most realistic / most like the original thermal GT."),

        ("fg_bg_diff",           "low",  (20, 35),
         "fg_bg_diff_low",
         "Foreground-Background Difference — LOW  (20th–35th percentile)\n"
         "Object pixel intensity is close to background → pedestrian blends in,\n"
         "making it hard for the detector to distinguish foreground from background."),

        ("fg_bg_diff",           "high", (80, 95),
         "fg_bg_diff_high",
         "Foreground-Background Difference — HIGH\n"
         "Object pixel intensity is clearly different from background → easy to separate."),

        ("object_contrast",      "low",  (20, 35),
         "contrast_low",
         "Object Contrast — LOW  (20th–35th percentile)\n"
         "Std deviation of pixel values inside the bbox is low → low-texture, flat appearance.\n"
         "Less variation for the detector to latch onto."),

        ("object_contrast",      "high", (80, 95),
         "contrast_high",
         "Object Contrast — HIGH\n"
         "Std deviation of pixel values is high → rich texture and variation inside the crop."),

        ("object_edge_strength", "low",  (20, 35),
         "edge_strength_low",
         "Edge Strength — LOW  (20th–35th percentile)\n"
         "Mean Sobel gradient magnitude is low → object has weak or faint visible edges.\n"
         "Unclear object boundary makes detection harder."),

        ("object_edge_strength", "high", (80, 95),
         "edge_strength_high",
         "Edge Strength — HIGH\n"
         "Mean Sobel gradient magnitude is high → strong, well-defined edges around the object."),
    ]

    # For visible (non-synthetic) modalities, skip metrics that require thermal GT
    SYNTHETIC_ONLY = {"ghost_score", "hallucination_score", "object_ssim", "object_lpips"}
    if not is_synthetic:
        CASES = [(m, ex, pr, st, ti) for m, ex, pr, st, ti in CASES
                 if m not in SYNTHETIC_ONLY]

    # For these metrics compare generated vs thermal GT side by side
    THERMAL_COMPARE = {"ghost_score", "hallucination_score", "object_ssim", "object_lpips"}

    def _load_rgb(path):
        img = _cv2.imread(path)
        if img is None:
            return None
        return _cv2.cvtColor(img, _cv2.COLOR_BGR2RGB)

    def _get_crop(img, cx, cy, w, h, extra=0):
        """Return square (1:1) crop + bbox coords relative to the padded crop."""
        x1 = int((cx - w / 2) * img_size)
        y1 = int((cy - h / 2) * img_size)
        x2 = int((cx + w / 2) * img_size)
        y2 = int((cy + h / 2) * img_size)
        ih, iw = img.shape[:2]
        c1x = max(0, x1 - pad_px - extra)
        c1y = max(0, y1 - pad_px - extra)
        c2x = min(iw, x2 + pad_px + extra)
        c2y = min(ih, y2 + pad_px + extra)
        crop = img[c1y:c2y, c1x:c2x]
        # Pad shorter dimension to make crop square (1:1) — avoids tiny slivers
        ch, cw_c = crop.shape[:2]
        dl = dr = dt = db = 0
        if ch != cw_c:
            side = max(ch, cw_c)
            dt = (side - ch) // 2;  db = side - ch - dt
            dl = (side - cw_c) // 2; dr = side - cw_c - dl
            crop = _cv2.copyMakeBorder(crop, dt, db, dl, dr, _cv2.BORDER_REPLICATE)
        return crop, (x1 - c1x + dl, y1 - c1y + dt, x2 - c1x + dl, y2 - c1y + dt)

    for (metric, extreme, pct_range, stem, title) in CASES:
        if metric not in df_obj.columns:
            continue
        valid = df_obj.dropna(subset=[metric, "flat_name", "gt_cx", "gt_cy", "gt_w", "gt_h"])
        if len(valid) < 10:
            continue

        lo_pct, hi_pct = pct_range
        lo_val = valid[metric].quantile(lo_pct / 100)
        hi_val = valid[metric].quantile(hi_pct / 100)
        band = valid[(valid[metric] >= lo_val) & (valid[metric] <= hi_val)]

        if extreme == "high":
            band = band.sort_values(metric, ascending=False)
        else:
            band = band.sort_values(metric, ascending=True)

        # Pick examples spread across the band (not all clustered at one image)
        # de-duplicate by flat_name first so examples come from different images
        seen_imgs = set()
        picks = []
        for _, r in band.iterrows():
            if r["flat_name"] not in seen_imgs:
                picks.append(r)
                seen_imgs.add(r["flat_name"])
            if len(picks) >= n_examples:
                break

        if not picks:
            continue

        show_thermal = (metric in THERMAL_COMPARE) and (thermal_dir is not None)
        n_cols = len(picks) * (2 if show_thermal else 1)
        fig, axes = plt.subplots(1, n_cols, figsize=(3.8 * n_cols, 5.0))
        if n_cols == 1:
            axes = [axes]

        fig.suptitle(title, fontsize=9, fontweight="bold",
                     y=1.02, ha="center")

        case_dir = os.path.join(report_dir, stem) if report_dir is not None else None
        if case_dir is not None:
            os.makedirs(case_dir, exist_ok=True)

        ax_idx = 0
        shown = 0
        for row in picks:
            img_path = os.path.join(image_dir, f"{row['flat_name']}.jpg")
            img = _load_rgb(img_path)
            if img is None:
                ax_idx += (2 if show_thermal else 1)
                continue

            crop, (rx1, ry1, rx2, ry2) = _get_crop(
                img, row["gt_cx"], row["gt_cy"], row["gt_w"], row["gt_h"]
            )
            det_lbl = "detected ✓" if row.get("detected", False) else "missed ✗"
            val_str = f"{metric} = {row[metric]:.3f}"

            # ── Upscale small crops for report quality ────────────────────
            crop_h, crop_w = crop.shape[:2]
            min_size = 256  # pixels — upscale if smaller
            if crop_h < min_size or crop_w < min_size:
                scale = max(min_size / crop_h, min_size / crop_w)
                new_h, new_w = int(crop_h * scale), int(crop_w * scale)
                crop = _cv2.resize(crop, (new_w, new_h),
                                   interpolation=_cv2.INTER_LANCZOS4)
                # Adjust bbox coords to match upscaled crop
                rx1, ry1, rx2, ry2 = (int(x * scale) for x in (rx1, ry1, rx2, ry2))

            # ── Save square crop + full image to report_dir ──────────────
            if case_dir is not None:
                fname = row["flat_name"]
                # Square crop with bbox drawn (blue rectangle)
                _tmp = _cv2.cvtColor(crop.copy(), _cv2.COLOR_RGB2BGR)
                _cv2.rectangle(_tmp, (rx1, ry1), (rx2, ry2), (51, 51, 255), 2)
                plt.imsave(
                    os.path.join(case_dir, f"{fname}_pid_crop.png"),
                    _cv2.cvtColor(_tmp, _cv2.COLOR_BGR2RGB),
                )
                # Full image with bbox highlighted (blue rectangle, thicker)
                _full = _cv2.cvtColor(img.copy(), _cv2.COLOR_RGB2BGR)
                _iw_f, _ih_f = _full.shape[1], _full.shape[0]
                _fx1 = max(0, int((row["gt_cx"] - row["gt_w"] / 2) * _iw_f))
                _fy1 = max(0, int((row["gt_cy"] - row["gt_h"] / 2) * _ih_f))
                _fx2 = min(_iw_f, int((row["gt_cx"] + row["gt_w"] / 2) * _iw_f))
                _fy2 = min(_ih_f, int((row["gt_cy"] + row["gt_h"] / 2) * _ih_f))
                _cv2.rectangle(_full, (_fx1, _fy1), (_fx2, _fy2), (51, 51, 255), 3)
                plt.imsave(
                    os.path.join(case_dir, f"{fname}_pid_full.png"),
                    _cv2.cvtColor(_full, _cv2.COLOR_BGR2RGB),
                )

            # ── Generated image crop ──────────────────────────────────────
            ax = axes[ax_idx]
            ax.imshow(crop)
            rect = _Rect((rx1, ry1), rx2 - rx1, ry2 - ry1,
                         linewidth=2, edgecolor="#FF3333", facecolor="none")
            ax.add_patch(rect)
            ax.set_title(f"{val_str}\n{det_lbl}", fontsize=8, pad=4)
            ax.set_xlabel(row["flat_name"], fontsize=6, color="#666666")
            ax.axis("off")
            ax_idx += 1

            # ── Thermal GT crop (side by side) ────────────────────────────
            if show_thermal:
                ax2 = axes[ax_idx]
                thm_path = os.path.join(thermal_dir, f"{row['flat_name']}.jpg")
                if os.path.exists(thm_path):
                    thm = _load_rgb(thm_path)
                    if thm is not None:
                        thm_crop, (tx1, ty1, tx2, ty2) = _get_crop(
                            thm, row["gt_cx"], row["gt_cy"], row["gt_w"], row["gt_h"]
                        )
                        # Apply same upscaling as main crop
                        if crop_h < min_size or crop_w < min_size:
                            thm_crop = _cv2.resize(thm_crop, (new_w, new_h),
                                                   interpolation=_cv2.INTER_LANCZOS4)
                            tx1, ty1, tx2, ty2 = (int(x * scale) for x in (tx1, ty1, tx2, ty2))
                        ax2.imshow(thm_crop)
                        if case_dir is not None:
                            plt.imsave(
                                os.path.join(case_dir, f"{row['flat_name']}_pid_crop_thermal.png"),
                                thm_crop,
                            )
                        rect2 = _Rect((tx1, ty1), tx2 - tx1, ty2 - ty1,
                                      linewidth=2, edgecolor="#33BB33",
                                      facecolor="none")
                        ax2.add_patch(rect2)
                        ax2.set_title("thermal GT", fontsize=8, pad=4,
                                      color="#226622")
                    else:
                        ax2.text(0.5, 0.5, "thermal\nnot found",
                                 ha="center", va="center", fontsize=8,
                                 color="gray", transform=ax2.transAxes)
                        ax2.set_facecolor("#eeeeee")
                else:
                    ax2.text(0.5, 0.5, "thermal\nnot found",
                             ha="center", va="center", fontsize=8,
                             color="gray", transform=ax2.transAxes)
                    ax2.set_facecolor("#eeeeee")
                ax2.axis("off")
                ax_idx += 1

            shown += 1

        # Hide unused axes
        while ax_idx < len(axes):
            axes[ax_idx].axis("off")
            ax_idx += 1

        if shown == 0:
            plt.close()
            continue

        plt.tight_layout()
        out_path = os.path.join(output_dir, f"{stem}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved example crops: {out_path}")
