"""
make_print_sheets.py
====================
Generates printable A4 PDF sheets from images_report folders.
Each metric gets its own page(s) with full-size images, metric name,
individual scores, and dataset label — ready to print and cut.

Usage:
    python OD/make_print_sheets.py --eval-dir OD/eval_results/PID_eval --label PID
    python OD/make_print_sheets.py --eval-dir OD/eval_results/PI-GAN_eval --label PI-GAN
    python OD/make_print_sheets.py --eval-dir OD/eval_results/visible_eval --label Visible
"""

import argparse
import glob
import os

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# ── A4 canvas at 150 DPI ─────────────────────────────────────────────────────
DPI = 150
A4_W = int(8.27 * DPI)   # 1240 px
A4_H = int(11.69 * DPI)  # 1754 px
MARGIN = 40
GAP = 20

# Font sizes
TITLE_SIZE = 36
LABEL_SIZE = 22
SCORE_SIZE = 18

BG_COLOR = (255, 255, 255)
TITLE_COLOR = (30, 30, 30)
LABEL_COLOR = (60, 60, 180)
SCORE_COLOR = (80, 80, 80)
LINE_COLOR = (200, 200, 200)

# Metric folder → CSV column mapping
METRIC_COL = {
    "contrast":       "contrast",
    "edge_strength":  "edge_density",
    "fg_bg_diff":     "fg_bg_diff",
    "ghost_score":    "ghost_score",
    "hallucination":  "hallucination_score",
    "lpips":          "lpips_mean",
    "sharpness":      "sharpness",
    "ssim":           "object_ssim",
    "brightness":     "brightness",
}

# Human-readable metric names
METRIC_LABEL = {
    "contrast":       "Contrast",
    "edge_strength":  "Edge Strength",
    "fg_bg_diff":     "Foreground–Background Difference",
    "ghost_score":    "Ghost Score",
    "hallucination":  "Hallucination Score",
    "lpips":          "LPIPS (Perceptual Distance)",
    "sharpness":      "Sharpness",
    "ssim":           "SSIM (Structural Similarity)",
    "brightness":     "Brightness",
}


def _load_font(size):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
        except Exception:
            return ImageFont.load_default()


def _parse_metric_and_level(folder_name):
    """Split e.g. 'contrast_high' → ('contrast', 'high')."""
    for suffix in ("_high", "_low"):
        if folder_name.endswith(suffix):
            metric = folder_name[: -len(suffix)]
            level = suffix[1:]
            return metric, level
    return folder_name, ""


def _load_scores(eval_dir):
    """Load per-image scores from the first CSV found in eval_dir."""
    csvs = glob.glob(os.path.join(eval_dir, "per_image_results_*.csv"))
    if not csvs:
        print("  WARNING: no per_image_results_*.csv found — scores will be N/A")
        return None
    df = pd.read_csv(csvs[0])
    df.columns = [c.strip() for c in df.columns]
    df["flat_name"] = df["flat_name"].astype(str).str.strip()
    return df


def _extract_flat_name(filename):
    """Extract flat_name (set##_V###_I#####) from the image filename."""
    base = os.path.splitext(os.path.basename(filename))[0]
    # strip trailing _pid_full / _visible_full / _pigan_full / _full etc.
    for suffix in ("_pid_full", "_visible_full", "_pigan_full", "_pi-gan_full",
                   "_PI-GAN_gen_full", "_full"):
        if base.endswith(suffix):
            return base[: -len(suffix)]
    return base


def _get_score(flat_name, metric, df_scores):
    if df_scores is None:
        return None
    col = METRIC_COL.get(metric)
    if col is None or col not in df_scores.columns:
        return None
    row = df_scores[df_scores["flat_name"] == flat_name]
    if row.empty:
        return None
    val = row[col].iloc[0]
    try:
        return float(val)
    except Exception:
        return None


def draw_page(images_info, metric, level, dataset_label):
    """
    images_info: list of (path, flat_name, score)
    Returns a PIL Image (A4).
    """
    font_title = _load_font(TITLE_SIZE)
    font_label = _load_font(LABEL_SIZE)
    font_score = _load_font(SCORE_SIZE)

    canvas = Image.new("RGB", (A4_W, A4_H), BG_COLOR)
    draw = ImageDraw.Draw(canvas)

    metric_human = METRIC_LABEL.get(metric, metric.replace("_", " ").title())
    level_human = level.upper() if level else ""

    # ── Header ──────────────────────────────────────────────────────────────
    title_text = f"{metric_human}  [{level_human}]"
    draw.text((MARGIN, MARGIN), title_text, font=font_title, fill=TITLE_COLOR)
    dataset_text = f"Dataset: {dataset_label}"
    draw.text((MARGIN, MARGIN + TITLE_SIZE + 6), dataset_text,
              font=font_label, fill=LABEL_COLOR)
    header_h = MARGIN + TITLE_SIZE + LABEL_SIZE + 20
    draw.line([(MARGIN, header_h), (A4_W - MARGIN, header_h)],
              fill=LINE_COLOR, width=2)
    y_cursor = header_h + GAP

    # ── Image grid ──────────────────────────────────────────────────────────
    n = len(images_info)
    cols = min(n, 3)          # max 3 per row
    rows = (n + cols - 1) // cols

    available_w = A4_W - 2 * MARGIN - (cols - 1) * GAP
    available_h = A4_H - y_cursor - MARGIN - rows * (LABEL_SIZE + SCORE_SIZE + GAP * 2)
    cell_w = available_w // cols
    cell_h = available_h // rows

    for idx, (img_path, flat_name, score) in enumerate(images_info):
        row_i = idx // cols
        col_i = idx % cols

        x = MARGIN + col_i * (cell_w + GAP)
        y = y_cursor + row_i * (cell_h + LABEL_SIZE + SCORE_SIZE + GAP * 2)

        # Image ID label
        draw.text((x, y), flat_name, font=font_label, fill=TITLE_COLOR)
        y += LABEL_SIZE + 4

        # Score label
        score_text = f"{metric_human}: {score:.4f}" if score is not None else f"{metric_human}: N/A"
        draw.text((x, y), score_text, font=font_score, fill=SCORE_COLOR)
        y += SCORE_SIZE + 6

        # Image
        try:
            img = Image.open(img_path).convert("RGB")
            img.thumbnail((cell_w, cell_h), Image.LANCZOS)
            canvas.paste(img, (x, y))
        except Exception as e:
            draw.rectangle([x, y, x + cell_w, y + cell_h], outline=(200, 0, 0))
            draw.text((x + 4, y + 4), f"Error: {e}", font=font_score, fill=(200, 0, 0))

    return canvas


def process_eval_dir(eval_dir, dataset_label, output_dir):
    report_dir = os.path.join(eval_dir, "images_report")
    if not os.path.isdir(report_dir):
        print(f"ERROR: images_report not found at {report_dir}")
        return

    df_scores = _load_scores(eval_dir)
    os.makedirs(output_dir, exist_ok=True)

    metric_folders = sorted([
        d for d in os.listdir(report_dir)
        if os.path.isdir(os.path.join(report_dir, d))
    ])

    print(f"\nProcessing {len(metric_folders)} metric folders for [{dataset_label}]")

    for folder_name in metric_folders:
        folder_path = os.path.join(report_dir, folder_name)
        metric, level = _parse_metric_and_level(folder_name)

        # Only full-sized images
        full_images = sorted(glob.glob(os.path.join(folder_path, "*_full.png")))
        if not full_images:
            print(f"  [{folder_name}] No _full.png images found, skipping.")
            continue

        images_info = []
        for img_path in full_images:
            flat_name = _extract_flat_name(img_path)
            score = _get_score(flat_name, metric, df_scores)
            images_info.append((img_path, flat_name, score))

        print(f"  [{folder_name}] {len(images_info)} images → ", end="")

        # Split into pages of 6 max
        pages = []
        page_size = 6
        for i in range(0, len(images_info), page_size):
            chunk = images_info[i: i + page_size]
            page = draw_page(chunk, metric, level, dataset_label)
            pages.append(page)

        # Save as multi-page PDF
        out_path = os.path.join(output_dir, f"{folder_name}.pdf")
        if len(pages) == 1:
            pages[0].save(out_path, "PDF", resolution=DPI)
        else:
            pages[0].save(out_path, "PDF", resolution=DPI,
                          save_all=True, append_images=pages[1:])
        print(f"saved → {out_path}")

    # Also save one combined PDF with all metrics
    all_pages = []
    for folder_name in metric_folders:
        folder_path = os.path.join(report_dir, folder_name)
        metric, level = _parse_metric_and_level(folder_name)
        full_images = sorted(glob.glob(os.path.join(folder_path, "*_full.png")))
        if not full_images:
            continue
        images_info = [
            (p, _extract_flat_name(p), _get_score(_extract_flat_name(p), metric, df_scores))
            for p in full_images
        ]
        for i in range(0, len(images_info), 6):
            chunk = images_info[i: i + 6]
            all_pages.append(draw_page(chunk, metric, level, dataset_label))

    if all_pages:
        combined_path = os.path.join(output_dir, f"ALL_{dataset_label}.pdf")
        all_pages[0].save(combined_path, "PDF", resolution=DPI,
                          save_all=True, append_images=all_pages[1:])
        print(f"\nCombined PDF → {combined_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate printable A4 report sheets")
    parser.add_argument("--eval-dir", required=True,
                        help="Path to eval folder, e.g. OD/eval_results/PID_eval")
    parser.add_argument("--label", required=True,
                        help="Dataset label printed on each page, e.g. PID")
    parser.add_argument("--output-dir", default=None,
                        help="Output folder (default: <eval-dir>/print_sheets)")
    args = parser.parse_args()

    out = args.output_dir or os.path.join(args.eval_dir, "print_sheets")
    process_eval_dir(args.eval_dir, args.label, out)
