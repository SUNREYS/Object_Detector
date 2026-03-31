"""
=============================================================================
Plotting Script: Training Curves & Subgroup Evaluation Plots
=============================================================================

Generates all plots for the thesis from two data sources:

  A) YOLO training results.csv   -- loss curves, mAP, precision, recall
  B) Evaluation CSVs from evaluate.py -- subgroup detection performance

Plots produced:
    01  Training & validation loss curves (box, cls, dfl)
    02  Detection metrics over epochs (precision, recall, mAP50, mAP50-95)
    03  Combined training overview (2x2 grid)
    04  Learning rate schedule
    05  Subgroup recall bar chart (day/night, size, occlusion)
    06  Detection rate by object size category
    07  Detection rate by day vs night
    08  Detection rate by occlusion level
    09  Detection rate by crowd density
    10  Confusion matrix (from subgroup data)
    11  Missed-object analysis (size distribution of FN objects)
    12  Combined evaluation dashboard (2x3 grid)

Usage:
    python plot_results.py                                    # all plots
    python plot_results.py --training-only                    # training plots only
    python plot_results.py --eval-only                        # evaluation plots only
    python plot_results.py --modality thermal
=============================================================================
"""

import argparse
import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# PATHS
# =============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(SCRIPT_DIR, "runs")
EVAL_DIR = os.path.join(SCRIPT_DIR, "eval_results")
PLOTS_DIR = os.path.join(SCRIPT_DIR, "plots")


def save_plot(fig, name: str):
    """Save figure to the plots directory."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# =============================================================================
# A)  TRAINING PLOTS — from YOLO results.csv
# =============================================================================

def load_training_results(modality: str) -> pd.DataFrame:
    """Load and clean the ultralytics results.csv."""
    path = os.path.join(RUNS_DIR, f"kaist_{modality}", "results.csv")
    if not os.path.exists(path):
        print(f"WARNING: Training results not found: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df["epoch"] = range(1, len(df) + 1)
    return df


def plot_loss_curves(df: pd.DataFrame, modality: str):
    """Plot 01: training and validation loss curves (box, cls, dfl)."""
    print("[Plot 01] Training & validation loss curves...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    epochs = df["epoch"]

    loss_pairs = [
        ("train/box_loss", "val/box_loss", "Box Loss"),
        ("train/cls_loss", "val/cls_loss", "Classification Loss"),
        ("train/dfl_loss", "val/dfl_loss", "DFL Loss"),
    ]

    for i, (train_col, val_col, title) in enumerate(loss_pairs):
        if train_col in df.columns:
            axes[i].plot(epochs, df[train_col], label="Train",
                         color="#2196F3", linewidth=1.5)
        if val_col in df.columns:
            axes[i].plot(epochs, df[val_col], label="Val",
                         color="#F44336", linewidth=1.5)
        axes[i].set_xlabel("Epoch")
        axes[i].set_ylabel("Loss")
        axes[i].set_title(title, fontsize=13)
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    fig.suptitle(f"Training & Validation Loss — {modality.capitalize()}",
                 fontsize=15, y=1.02)
    fig.tight_layout()
    save_plot(fig, f"01_loss_curves_{modality}")


def plot_metrics_over_epochs(df: pd.DataFrame, modality: str):
    """Plot 02: precision, recall, mAP50, mAP50-95 over epochs."""
    print("[Plot 02] Detection metrics over epochs...")

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    epochs = df["epoch"]

    metrics = [
        ("metrics/precision(B)", "Precision", "#4CAF50"),
        ("metrics/recall(B)", "Recall", "#FF9800"),
        ("metrics/mAP50(B)", "mAP@50", "#9C27B0"),
        ("metrics/mAP50-95(B)", "mAP@50-95", "#E91E63"),
    ]

    for i, (col, title, color) in enumerate(metrics):
        if col in df.columns:
            axes[i].plot(epochs, df[col], color=color, linewidth=1.5)
            best_val = df[col].max()
            best_ep = df.loc[df[col].idxmax(), "epoch"]
            axes[i].axhline(y=best_val, color=color, linestyle="--",
                            alpha=0.4)
            axes[i].set_title(f"{title}\nbest={best_val:.4f} @ ep {best_ep}",
                              fontsize=12)
        else:
            axes[i].set_title(f"{title}\n(column not found)", fontsize=12)

        axes[i].set_xlabel("Epoch")
        axes[i].set_ylabel(title)
        axes[i].set_ylim(0, 1)
        axes[i].grid(True, alpha=0.3)

    fig.suptitle(f"Detection Metrics — {modality.capitalize()}",
                 fontsize=15, y=1.02)
    fig.tight_layout()
    save_plot(fig, f"02_metrics_{modality}")


def plot_training_overview(df: pd.DataFrame, modality: str):
    """Plot 03: 2x2 combined overview (box loss, mAP50, precision, recall)."""
    print("[Plot 03] Combined training overview...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    epochs = df["epoch"]

    panels = [
        (axes[0, 0], "train/box_loss", "val/box_loss", "Box Loss", False),
        (axes[0, 1], "metrics/mAP50(B)", None, "mAP@50", True),
        (axes[1, 0], "metrics/precision(B)", None, "Precision", True),
        (axes[1, 1], "metrics/recall(B)", None, "Recall", True),
    ]

    for ax, col1, col2, title, is_metric in panels:
        if col1 in df.columns:
            ax.plot(epochs, df[col1],
                    label="Train" if col2 else title,
                    color="#2196F3", linewidth=1.5)
        if col2 and col2 in df.columns:
            ax.plot(epochs, df[col2], label="Val",
                    color="#F44336", linewidth=1.5)

        ax.set_xlabel("Epoch")
        ax.set_title(title, fontsize=13)
        ax.grid(True, alpha=0.3)
        if is_metric:
            ax.set_ylim(0, 1)
        if col2:
            ax.legend()

    fig.suptitle(f"Training Overview — {modality.capitalize()}",
                 fontsize=15, y=1.02)
    fig.tight_layout()
    save_plot(fig, f"03_training_overview_{modality}")


def plot_lr_schedule(df: pd.DataFrame, modality: str):
    """Plot 04: learning rate schedule over epochs."""
    print("[Plot 04] Learning rate schedule...")

    lr_cols = [c for c in df.columns if c.startswith("lr/")]
    if not lr_cols:
        print("  No LR columns found, skipping.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    epochs = df["epoch"]
    colors = ["#2196F3", "#4CAF50", "#FF9800"]

    for i, col in enumerate(lr_cols):
        ax.plot(epochs, df[col], label=col,
                color=colors[i % len(colors)], linewidth=1.5)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title(f"Learning Rate Schedule — {modality.capitalize()}",
                 fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save_plot(fig, f"04_lr_schedule_{modality}")


def generate_training_plots(modality: str):
    """Generate all training-related plots."""
    df = load_training_results(modality)
    if df.empty:
        print("No training results available. Skipping training plots.")
        return

    print(f"\nGenerating training plots for {modality} "
          f"({len(df)} epochs)...\n")

    plot_loss_curves(df, modality)
    plot_metrics_over_epochs(df, modality)
    plot_training_overview(df, modality)
    plot_lr_schedule(df, modality)


# =============================================================================
# B)  EVALUATION PLOTS — from evaluate.py output CSVs
# =============================================================================

def load_eval_results(modality: str):
    """Load evaluation CSVs produced by evaluate.py."""
    img_path = os.path.join(EVAL_DIR, f"per_image_results_{modality}.csv")
    obj_path = os.path.join(EVAL_DIR, f"per_object_results_{modality}.csv")
    sub_path = os.path.join(EVAL_DIR, f"subgroup_summary_{modality}.csv")

    df_img = pd.read_csv(img_path) if os.path.exists(img_path) else pd.DataFrame()
    df_obj = pd.read_csv(obj_path) if os.path.exists(obj_path) else pd.DataFrame()
    df_sub = pd.read_csv(sub_path) if os.path.exists(sub_path) else pd.DataFrame()

    return df_img, df_obj, df_sub


def plot_subgroup_recall(df_sub: pd.DataFrame, modality: str):
    """Plot 05: recall bar chart for each subgroup."""
    print("[Plot 05] Subgroup recall bar chart...")

    if df_sub.empty:
        print("  No subgroup summary data. Skipping.")
        return

    # Separate subgroup types
    types = df_sub["subgroup_type"].unique()
    n_types = len(types)
    fig, axes = plt.subplots(1, n_types, figsize=(5 * n_types, 5))
    if n_types == 1:
        axes = [axes]

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0", "#00BCD4"]

    for i, stype in enumerate(types):
        sub = df_sub[df_sub["subgroup_type"] == stype].sort_values("subgroup_value")
        labels = sub["subgroup_value"].astype(str).tolist()
        recalls = sub["recall"].tolist()
        counts = sub["count"].tolist()

        bars = axes[i].bar(labels, recalls, color=colors[i % len(colors)],
                           edgecolor="black", alpha=0.85)
        axes[i].set_title(stype.replace("_", " ").title(), fontsize=12)
        axes[i].set_ylabel("Recall")
        axes[i].set_ylim(0, 1.05)
        axes[i].grid(axis="y", alpha=0.3)

        # Add count labels above bars
        for bar, cnt in zip(bars, counts):
            axes[i].text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 0.02,
                         f"n={cnt}", ha="center", fontsize=8)

    fig.suptitle(f"Detection Recall by Subgroup — {modality.capitalize()}",
                 fontsize=15, y=1.02)
    fig.tight_layout()
    save_plot(fig, f"05_subgroup_recall_{modality}")


def plot_detection_by_size(df_obj: pd.DataFrame, modality: str):
    """Plot 06: detection rate by object size category."""
    print("[Plot 06] Detection rate by size category...")

    if df_obj.empty or "size_category" not in df_obj.columns:
        print("  No size category data. Skipping.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    cats = ["small", "medium", "large"]
    detected_counts = []
    missed_counts = []

    for cat in cats:
        sub = df_obj[df_obj["size_category"] == cat]
        det = int(sub["detected"].sum())
        mis = len(sub) - det
        detected_counts.append(det)
        missed_counts.append(mis)

    # Stacked bar
    x = np.arange(len(cats))
    axes[0].bar(x, detected_counts, label="Detected", color="#4CAF50")
    axes[0].bar(x, missed_counts, bottom=detected_counts,
                label="Missed", color="#F44336")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(cats)
    axes[0].set_ylabel("Number of Objects")
    axes[0].set_title("Detection Counts by Size", fontsize=13)
    axes[0].legend()

    # Recall bars
    totals = [d + m for d, m in zip(detected_counts, missed_counts)]
    recalls = [d / t if t > 0 else 0 for d, t in zip(detected_counts, totals)]
    bars = axes[1].bar(cats, recalls, color=["#FFC107", "#FF9800", "#F44336"])
    axes[1].set_ylabel("Recall")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Recall by Size Category", fontsize=13)
    axes[1].grid(axis="y", alpha=0.3)
    for bar, r, t in zip(bars, recalls, totals):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     r + 0.02, f"{r:.2f}\n(n={t})",
                     ha="center", fontsize=9)

    fig.suptitle(f"Detection by Object Size — {modality.capitalize()}",
                 fontsize=15, y=1.02)
    fig.tight_layout()
    save_plot(fig, f"06_detection_by_size_{modality}")


def plot_detection_by_daynight(df_obj: pd.DataFrame, modality: str):
    """Plot 07: detection rate by day vs night."""
    print("[Plot 07] Detection rate by day/night...")

    if df_obj.empty or "day_night" not in df_obj.columns:
        print("  No day/night data. Skipping.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for dn_val in ["day", "night"]:
        sub = df_obj[df_obj["day_night"] == dn_val]
        if sub.empty:
            continue

    groups = df_obj.groupby("day_night")["detected"]
    labels = []
    recalls = []
    counts = []
    for name, grp in groups:
        labels.append(str(name))
        recalls.append(grp.mean())
        counts.append(len(grp))

    colors = ["#FFC107" if l == "day" else "#3F51B5" for l in labels]
    bars = axes[0].bar(labels, recalls, color=colors, edgecolor="black")
    axes[0].set_ylabel("Recall")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Recall: Day vs Night", fontsize=13)
    for bar, r, c in zip(bars, recalls, counts):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     r + 0.02, f"{r:.3f}\n(n={c})",
                     ha="center", fontsize=10)

    # Breakdown by size within day/night
    cats = ["small", "medium", "large"]
    if "size_category" in df_obj.columns:
        x = np.arange(len(cats))
        width = 0.35
        for di, (dn_val, color) in enumerate(
                [("day", "#FFC107"), ("night", "#3F51B5")]):
            vals = []
            for cat in cats:
                sub = df_obj[
                    (df_obj["day_night"] == dn_val) &
                    (df_obj["size_category"] == cat)
                ]
                vals.append(sub["detected"].mean() if len(sub) > 0 else 0)
            offset = -width / 2 + di * width
            axes[1].bar(x + offset, vals, width, label=dn_val, color=color)

        axes[1].set_xticks(x)
        axes[1].set_xticklabels(cats)
        axes[1].set_ylabel("Recall")
        axes[1].set_ylim(0, 1.05)
        axes[1].set_title("Recall by Size & Day/Night", fontsize=13)
        axes[1].legend()
        axes[1].grid(axis="y", alpha=0.3)

    fig.suptitle(f"Day vs Night Detection — {modality.capitalize()}",
                 fontsize=15, y=1.02)
    fig.tight_layout()
    save_plot(fig, f"07_detection_daynight_{modality}")


def plot_detection_by_occlusion(df_obj: pd.DataFrame, modality: str):
    """Plot 08: detection rate by occlusion level."""
    print("[Plot 08] Detection rate by occlusion level...")

    if df_obj.empty or "occlusion" not in df_obj.columns:
        print("  No occlusion data. Skipping.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    labels_map = {0: "None (0)", 1: "Partial (1)", 2: "Heavy (2)"}
    occ_levels = [0, 1, 2]
    recalls = []
    counts = []

    for occ in occ_levels:
        sub = df_obj[df_obj["occlusion"] == occ]
        n = len(sub)
        counts.append(n)
        recalls.append(sub["detected"].mean() if n > 0 else 0)

    colors = ["#4CAF50", "#FFC107", "#F44336"]
    bars = ax.bar([labels_map[o] for o in occ_levels], recalls,
                  color=colors, edgecolor="black")
    ax.set_ylabel("Recall")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Recall by Occlusion Level — {modality.capitalize()}",
                 fontsize=14)
    ax.grid(axis="y", alpha=0.3)

    for bar, r, c in zip(bars, recalls, counts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                r + 0.02, f"{r:.3f}\n(n={c})",
                ha="center", fontsize=10)

    fig.tight_layout()
    save_plot(fig, f"08_detection_occlusion_{modality}")


def plot_detection_by_crowd(df_img: pd.DataFrame, modality: str):
    """Plot 09: detection metrics by crowd density."""
    print("[Plot 09] Detection by crowd density...")

    if df_img.empty or "num_people" not in df_img.columns:
        print("  No crowd data. Skipping.")
        return

    df = df_img.copy()
    df["crowd_bin"] = df["num_people"].apply(
        lambda x: str(int(x)) if x <= 4 else "5+"
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    bin_order = ["0", "1", "2", "3", "4", "5+"]
    recall_vals = []
    prec_vals = []
    count_vals = []

    for b in bin_order:
        sub = df[df["crowd_bin"] == b]
        if len(sub) == 0:
            recall_vals.append(0)
            prec_vals.append(0)
            count_vals.append(0)
            continue
        tp = sub["tp"].sum()
        fp = sub["fp"].sum()
        fn = sub["fn"].sum()
        count_vals.append(int(tp + fn))
        recall_vals.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        prec_vals.append(tp / (tp + fp) if (tp + fp) > 0 else 0)

    # Recall
    bars = axes[0].bar(bin_order, recall_vals, color="#2196F3",
                       edgecolor="black", alpha=0.85)
    axes[0].set_xlabel("People per Frame")
    axes[0].set_ylabel("Recall")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Recall by Crowd Density", fontsize=13)
    axes[0].grid(axis="y", alpha=0.3)
    for bar, c in zip(bars, count_vals):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.02, f"n={c}",
                     ha="center", fontsize=8)

    # Precision
    axes[1].bar(bin_order, prec_vals, color="#4CAF50",
                edgecolor="black", alpha=0.85)
    axes[1].set_xlabel("People per Frame")
    axes[1].set_ylabel("Precision")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Precision by Crowd Density", fontsize=13)
    axes[1].grid(axis="y", alpha=0.3)

    fig.suptitle(f"Crowd Density Analysis — {modality.capitalize()}",
                 fontsize=15, y=1.02)
    fig.tight_layout()
    save_plot(fig, f"09_detection_crowd_{modality}")


def plot_confusion_matrix(df_img: pd.DataFrame, modality: str):
    """Plot 10: simple confusion-style summary (TP/FP/FN totals by subgroup)."""
    print("[Plot 10] Confusion summary...")

    if df_img.empty or "day_night" not in df_img.columns:
        print("  No data. Skipping.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    groups = ["day", "night"]
    tp_vals = []
    fp_vals = []
    fn_vals = []

    for g in groups:
        sub = df_img[df_img["day_night"] == g]
        tp_vals.append(int(sub["tp"].sum()))
        fp_vals.append(int(sub["fp"].sum()))
        fn_vals.append(int(sub["fn"].sum()))

    x = np.arange(len(groups))
    width = 0.25
    ax.bar(x - width, tp_vals, width, label="TP", color="#4CAF50")
    ax.bar(x, fp_vals, width, label="FP", color="#FF9800")
    ax.bar(x + width, fn_vals, width, label="FN", color="#F44336")
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("Count")
    ax.set_title(f"TP / FP / FN by Day/Night — {modality.capitalize()}",
                 fontsize=14)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    save_plot(fig, f"10_confusion_summary_{modality}")


def plot_missed_objects(df_obj: pd.DataFrame, modality: str):
    """Plot 11: analysis of missed GT objects (false negatives)."""
    print("[Plot 11] Missed object analysis...")

    if df_obj.empty:
        print("  No object data. Skipping.")
        return

    missed = df_obj[df_obj["detected"] == False]
    if missed.empty:
        print("  No missed objects (perfect recall). Skipping.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Height distribution of missed vs detected
    detected = df_obj[df_obj["detected"] == True]
    if "bbox_height_px" in df_obj.columns:
        bins = np.linspace(0, 300, 40)
        axes[0].hist(detected["bbox_height_px"].dropna(), bins=bins,
                     alpha=0.7, label="Detected", color="#4CAF50")
        axes[0].hist(missed["bbox_height_px"].dropna(), bins=bins,
                     alpha=0.7, label="Missed", color="#F44336")
        axes[0].set_xlabel("Bbox Height (px)")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Height Distribution: Detected vs Missed")
        axes[0].legend()

    # Miss rate by occlusion level
    if "occlusion" in df_obj.columns:
        occ_labels = ["None", "Partial", "Heavy"]
        miss_rates = []
        counts = []
        for i in range(3):
            sub = df_obj[df_obj["occlusion"] == i]
            n = len(sub)
            counts.append(n)
            miss_rates.append(
                1.0 - sub["detected"].mean() if n > 0 else 0
            )
        colors = ["#4CAF50", "#FFC107", "#F44336"]
        bars = axes[1].bar(occ_labels, miss_rates, color=colors)
        axes[1].set_ylabel("Miss Rate")
        axes[1].set_ylim(0, 1.05)
        axes[1].set_title("Miss Rate by Occlusion Level")
        axes[1].grid(axis="y", alpha=0.3)
        for bar, r, n in zip(bars, miss_rates, counts):
            axes[1].text(bar.get_x() + bar.get_width() / 2,
                         r + 0.02, f"{r:.1%}\n(n={n})",
                         ha="center", fontsize=9)

    # Distance from center of missed vs detected
    if "dist_from_center" in df_obj.columns:
        bins = np.linspace(0, 0.8, 30)
        axes[2].hist(detected["dist_from_center"].dropna(), bins=bins,
                     alpha=0.7, label="Detected", color="#4CAF50")
        axes[2].hist(missed["dist_from_center"].dropna(), bins=bins,
                     alpha=0.7, label="Missed", color="#F44336")
        axes[2].set_xlabel("Distance from Image Center")
        axes[2].set_ylabel("Count")
        axes[2].set_title("Position: Detected vs Missed")
        axes[2].legend()

    fig.suptitle(f"Missed Object Analysis — {modality.capitalize()}",
                 fontsize=15, y=1.02)
    fig.tight_layout()
    save_plot(fig, f"11_missed_analysis_{modality}")


def plot_eval_dashboard(df_img: pd.DataFrame, df_obj: pd.DataFrame,
                        modality: str):
    """Plot 12: combined 2x3 evaluation dashboard."""
    print("[Plot 12] Combined evaluation dashboard...")

    if df_obj.empty:
        print("  No evaluation data. Skipping.")
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # (0,0) Recall by size
    if "size_category" in df_obj.columns:
        cats = ["small", "medium", "large"]
        recalls = []
        for cat in cats:
            sub = df_obj[df_obj["size_category"] == cat]
            recalls.append(sub["detected"].mean() if len(sub) > 0 else 0)
        axes[0, 0].bar(cats, recalls, color=["#FFC107", "#FF9800", "#F44336"])
        axes[0, 0].set_ylim(0, 1.05)
        axes[0, 0].set_title("Recall by Size")
        axes[0, 0].set_ylabel("Recall")

    # (0,1) Recall by day/night
    if "day_night" in df_obj.columns:
        for dn, color in [("day", "#FFC107"), ("night", "#3F51B5")]:
            sub = df_obj[df_obj["day_night"] == dn]
            if len(sub) > 0:
                r = sub["detected"].mean()
                axes[0, 1].bar(dn, r, color=color)
        axes[0, 1].set_ylim(0, 1.05)
        axes[0, 1].set_title("Recall by Day/Night")
        axes[0, 1].set_ylabel("Recall")

    # (0,2) Recall by occlusion
    if "occlusion" in df_obj.columns:
        occ_labels = ["None", "Partial", "Heavy"]
        occ_recalls = []
        for occ in [0, 1, 2]:
            sub = df_obj[df_obj["occlusion"] == occ]
            occ_recalls.append(sub["detected"].mean() if len(sub) > 0 else 0)
        axes[0, 2].bar(occ_labels, occ_recalls,
                       color=["#4CAF50", "#FFC107", "#F44336"])
        axes[0, 2].set_ylim(0, 1.05)
        axes[0, 2].set_title("Recall by Occlusion")
        axes[0, 2].set_ylabel("Recall")

    # (1,0) Recall by truncation
    if "truncated" in df_obj.columns:
        for t_val, label, color in [(0, "Not Trunc", "#4CAF50"),
                                     (1, "Truncated", "#F44336")]:
            sub = df_obj[df_obj["truncated"] == t_val]
            if len(sub) > 0:
                r = sub["detected"].mean()
                bar = axes[1, 0].bar(label, r, color=color)
                axes[1, 0].text(bar[0].get_x() + bar[0].get_width() / 2,
                                r + 0.02, f"{r:.3f}\n(n={len(sub)})",
                                ha="center", fontsize=8)
        axes[1, 0].set_ylim(0, 1.05)
        axes[1, 0].set_title("Recall by Truncation")
        axes[1, 0].set_ylabel("Recall")

    # (1,1) Recall by overlap
    if "has_overlap" in df_obj.columns:
        for h_val, label, color in [(False, "No Overlap", "#4CAF50"),
                                     (True, "Has Overlap", "#F44336")]:
            sub = df_obj[df_obj["has_overlap"] == h_val]
            if len(sub) > 0:
                r = sub["detected"].mean()
                bar = axes[1, 1].bar(label, r, color=color)
                axes[1, 1].text(bar[0].get_x() + bar[0].get_width() / 2,
                                r + 0.02, f"{r:.3f}\n(n={len(sub)})",
                                ha="center", fontsize=8)
        axes[1, 1].set_ylim(0, 1.05)
        axes[1, 1].set_title("Recall by Overlap")
        axes[1, 1].set_ylabel("Recall")

    # (1,2) Overall precision-recall
    if not df_img.empty:
        total_tp = df_img["tp"].sum()
        total_fp = df_img["fp"].sum()
        total_fn = df_img["fn"].sum()
        prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        axes[1, 2].bar(["Precision", "Recall", "F1"],
                       [prec, rec, f1],
                       color=["#2196F3", "#4CAF50", "#9C27B0"])
        axes[1, 2].set_ylim(0, 1.05)
        axes[1, 2].set_title("Overall Metrics")
        for i, (val, label) in enumerate(
                zip([prec, rec, f1], ["Precision", "Recall", "F1"])):
            axes[1, 2].text(i, val + 0.02, f"{val:.3f}",
                            ha="center", fontsize=11)

    fig.suptitle(f"Evaluation Dashboard — {modality.capitalize()}",
                 fontsize=16, y=1.02)
    fig.tight_layout()
    save_plot(fig, f"12_eval_dashboard_{modality}")


def plot_detection_by_contrast(df_obj: pd.DataFrame, modality: str):
    """Plot 13: detection rate by foreground-background contrast."""
    print("[Plot 13] Detection rate by FG-BG contrast...")

    if df_obj.empty or "fg_bg_diff" not in df_obj.columns:
        print("  No fg_bg_diff data. Skipping.")
        return

    df = df_obj[df_obj["fg_bg_diff"].notna()].copy()
    if len(df) < 20:
        print("  Too few objects with quality features. Skipping.")
        return

    try:
        df["contrast_bin"] = pd.qcut(
            df["fg_bg_diff"], q=4,
            labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"],
            duplicates="drop",
        )
    except ValueError:
        print("  Cannot bin fg_bg_diff (insufficient variation). Skipping.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: recall by quartile
    bin_order = ["Q1 (low)", "Q2", "Q3", "Q4 (high)"]
    recalls = []
    counts = []
    for b in bin_order:
        sub = df[df["contrast_bin"] == b]
        counts.append(len(sub))
        recalls.append(sub["detected"].mean() if len(sub) > 0 else 0)

    colors = ["#F44336", "#FF9800", "#FFC107", "#4CAF50"]
    bars = axes[0].bar(bin_order, recalls, color=colors, edgecolor="black")
    axes[0].set_ylabel("Recall")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Recall by FG-BG Contrast Quartile", fontsize=13)
    axes[0].grid(axis="y", alpha=0.3)
    for bar, r, c in zip(bars, recalls, counts):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     r + 0.02, f"{r:.3f}\n(n={c})",
                     ha="center", fontsize=9)

    # Right: boxplot detected vs missed
    detected_vals = df[df["detected"] == True]["fg_bg_diff"]
    missed_vals = df[df["detected"] == False]["fg_bg_diff"]
    bp = axes[1].boxplot(
        [detected_vals.dropna(), missed_vals.dropna()],
        labels=["Detected", "Missed"],
        patch_artist=True,
    )
    bp["boxes"][0].set_facecolor("#4CAF50")
    bp["boxes"][1].set_facecolor("#F44336")
    axes[1].set_ylabel("FG-BG Brightness Difference")
    axes[1].set_title("Contrast Distribution: Detected vs Missed", fontsize=13)
    axes[1].grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"Object-Background Contrast Analysis -- {modality.capitalize()}",
        fontsize=15, y=1.02)
    fig.tight_layout()
    save_plot(fig, f"13_detection_by_contrast_{modality}")


def plot_detection_by_blur(df_obj: pd.DataFrame, modality: str):
    """Plot 14: detection rate by object blur (Laplacian variance)."""
    print("[Plot 14] Detection rate by object blur/sharpness...")

    if df_obj.empty or "object_blur" not in df_obj.columns:
        print("  No object_blur data. Skipping.")
        return

    df = df_obj[df_obj["object_blur"].notna()].copy()
    if len(df) < 20:
        print("  Too few objects with quality features. Skipping.")
        return

    try:
        df["blur_bin"] = pd.qcut(
            df["object_blur"], q=4,
            labels=["Q1 (blurry)", "Q2", "Q3", "Q4 (sharp)"],
            duplicates="drop",
        )
    except ValueError:
        print("  Cannot bin object_blur (insufficient variation). Skipping.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: recall by blur quartile
    bin_order = ["Q1 (blurry)", "Q2", "Q3", "Q4 (sharp)"]
    recalls = []
    counts = []
    for b in bin_order:
        sub = df[df["blur_bin"] == b]
        counts.append(len(sub))
        recalls.append(sub["detected"].mean() if len(sub) > 0 else 0)

    colors = ["#F44336", "#FF9800", "#FFC107", "#4CAF50"]
    bars = axes[0].bar(bin_order, recalls, color=colors, edgecolor="black")
    axes[0].set_ylabel("Recall")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Recall by Sharpness Quartile", fontsize=13)
    axes[0].grid(axis="y", alpha=0.3)
    for bar, r, c in zip(bars, recalls, counts):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     r + 0.02, f"{r:.3f}\n(n={c})",
                     ha="center", fontsize=9)

    # Right: boxplot detected vs missed
    detected_vals = df[df["detected"] == True]["object_blur"]
    missed_vals = df[df["detected"] == False]["object_blur"]
    bp = axes[1].boxplot(
        [detected_vals.dropna(), missed_vals.dropna()],
        labels=["Detected", "Missed"],
        patch_artist=True,
    )
    bp["boxes"][0].set_facecolor("#4CAF50")
    bp["boxes"][1].set_facecolor("#F44336")
    axes[1].set_ylabel("Laplacian Variance (higher = sharper)")
    axes[1].set_title("Sharpness Distribution: Detected vs Missed", fontsize=13)
    axes[1].grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"Object Sharpness Analysis -- {modality.capitalize()}",
        fontsize=15, y=1.02)
    fig.tight_layout()
    save_plot(fig, f"14_detection_by_blur_{modality}")


def plot_quality_dashboard(df_obj: pd.DataFrame, modality: str):
    """Plot 15: 2x2 image quality dashboard."""
    print("[Plot 15] Combined image quality dashboard...")

    quality_cols = [
        "fg_bg_diff", "object_blur",
        "object_edge_strength", "object_brightness",
    ]
    if df_obj.empty or not all(c in df_obj.columns for c in quality_cols):
        print("  Missing quality feature columns. Skipping.")
        return

    df = df_obj.dropna(subset=quality_cols).copy()
    if len(df) < 20:
        print("  Too few objects with quality features. Skipping.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    panels = [
        (axes[0, 0], "fg_bg_diff",
         "FG-BG Contrast", "FG-BG Brightness Difference"),
        (axes[0, 1], "object_blur",
         "Sharpness (Laplacian Var)", "Laplacian Variance"),
        (axes[1, 0], "object_edge_strength",
         "Edge Strength", "Mean Sobel Magnitude"),
        (axes[1, 1], "object_brightness",
         "Object Brightness", "Mean Pixel Intensity"),
    ]

    for ax, col, title, xlabel in panels:
        try:
            df[f"{col}_bin"] = pd.qcut(df[col], q=8, duplicates="drop")
        except ValueError:
            ax.text(0.5, 0.5, "Insufficient variation",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            continue

        bin_stats = (df.groupby(f"{col}_bin")["detected"]
                     .agg(["mean", "count"]).reset_index())

        midpoints = [interval.mid for interval in bin_stats[f"{col}_bin"]]
        recalls = bin_stats["mean"].tolist()
        counts = bin_stats["count"].tolist()

        sizes = [max(30, c * 2) for c in counts]
        ax.scatter(midpoints, recalls, s=sizes, c="#2196F3", alpha=0.7,
                   edgecolors="black", linewidths=0.5)

        if len(midpoints) > 2:
            z = np.polyfit(midpoints, recalls, deg=1)
            p = np.poly1d(z)
            x_line = np.linspace(min(midpoints), max(midpoints), 100)
            ax.plot(x_line, p(x_line), "--", color="#F44336", linewidth=1.5,
                    alpha=0.7, label=f"trend (slope={z[0]:.4f})")
            ax.legend(fontsize=8)

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Detection Rate")
        ax.set_ylim(0, 1.05)
        ax.set_title(title, fontsize=13)
        ax.grid(True, alpha=0.3)

        for mx, ry, cnt in zip(midpoints, recalls, counts):
            ax.annotate(f"n={cnt}", (mx, ry), textcoords="offset points",
                        xytext=(0, 8), fontsize=7, ha="center")

    fig.suptitle(
        f"Image Quality vs Detection -- {modality.capitalize()}",
        fontsize=16, y=1.02)
    fig.tight_layout()
    save_plot(fig, f"15_quality_dashboard_{modality}")


def generate_eval_plots(modality: str):
    """Generate all evaluation-related plots."""
    df_img, df_obj, df_sub = load_eval_results(modality)

    if df_img.empty and df_obj.empty:
        print("No evaluation results found. Run evaluate.py first.")
        return

    print(f"\nGenerating evaluation plots for {modality}...")
    print(f"  Per-image rows: {len(df_img)}")
    print(f"  Per-object rows: {len(df_obj)}")
    print()

    plot_subgroup_recall(df_sub, modality)
    plot_detection_by_size(df_obj, modality)
    plot_detection_by_daynight(df_obj, modality)
    plot_detection_by_occlusion(df_obj, modality)
    plot_detection_by_crowd(df_img, modality)
    plot_confusion_matrix(df_img, modality)
    plot_missed_objects(df_obj, modality)
    plot_eval_dashboard(df_img, df_obj, modality)
    plot_detection_by_contrast(df_obj, modality)
    plot_detection_by_blur(df_obj, modality)
    plot_quality_dashboard(df_obj, modality)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate training and evaluation plots"
    )
    parser.add_argument(
        "--modality", type=str, default="visible",
        choices=["visible", "thermal", "greyscale_inversion"],
    )
    parser.add_argument(
        "--training-only", action="store_true",
        help="Only generate training plots"
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Only generate evaluation plots"
    )
    args = parser.parse_args()

    if args.training_only:
        generate_training_plots(args.modality)
    elif args.eval_only:
        generate_eval_plots(args.modality)
    else:
        generate_training_plots(args.modality)
        generate_eval_plots(args.modality)

    print(f"\nAll plots saved to: {PLOTS_DIR}")
