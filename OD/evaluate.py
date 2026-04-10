"""
=============================================================================
evaluate.py — Main Evaluation Orchestrator
=============================================================================

Entry point for the full evaluation pipeline.  Runs YOLO inference on
validation images, computes detection metrics and image quality features,
then produces summary tables, correlation analyses, and plots.

Usage:
    python evaluate.py --modality pid           # single modality
    python evaluate.py --modality visible
    python evaluate.py --modality PI-GAN_gen
    python evaluate.py                          # all three + combined plots
    python evaluate.py --conf 0.3               # custom confidence threshold

The heavy lifting is split across four modules:
    eval_data.py      — data loading, detection matching, metadata
    eval_metrics.py   — quality metrics (per-object, ghost, LPIPS, FID)
    eval_analysis.py  — summary stats, correlations, FP analysis
    eval_plots.py     — all plotting functions
=============================================================================
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

from ultralytics import YOLO

# ── Sibling modules ──────────────────────────────────────────────────────
from eval_data import (
    infer_day_night, load_yolo_labels, yolo_to_xyxy,
    evaluate_image, match_gt_to_metadata, IOU_THRESHOLD,
)
from eval_metrics import (
    load_image, load_thermal_gt, compute_image_quality,
    compute_object_quality_features, compute_ghost_features,
    LPIPSScorer, FIDComputer,
)
from eval_analysis import (
    compute_subgroup_summary, print_full_summary,
)
from eval_plots import (
    plot_quality_vs_performance, plot_sharpness_size_heatmap,
    plot_fp_analysis, plot_fid_vs_map_bar, plot_lpips_vs_confidence,
    plot_example_crops,
)


# =============================================================================
# PATHS
# =============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DATASET_ROOT = os.path.join(PROJECT_ROOT, "datasets")
PID_DATASET_ROOT = os.path.join(PROJECT_ROOT, "datasets", "PID")
METADATA_DIR = os.path.join(DATASET_ROOT, "metadata")
THERMAL_GT_DIR = os.path.join(DATASET_ROOT, "images", "thermal")

FRAME_METADATA_CSV = os.path.join(METADATA_DIR, "frame_metadata.csv")
OBJECT_METADATA_CSV = os.path.join(METADATA_DIR, "object_metadata.csv")

RUNS_DIR = os.path.join(SCRIPT_DIR, "runs")
EVAL_BASE_DIR = os.path.join(SCRIPT_DIR, "eval_results")

EVAL_FOLDER = {
    "visible": "visible_eval",
    "pid": "PID_eval",
    "PI-GAN_gen": "PI-GAN_eval",
    "_all": "all_eval",
}

DEFAULT_CONF = 0.25

# Modalities for which we compare against thermal GT
# (visible images are real — no generation, so no thermal comparison)
SYNTHETIC_MODALITIES = {"pid", "PI-GAN_gen"}


def get_eval_dir(modality: str) -> str:
    folder = EVAL_FOLDER.get(modality, f"{modality}_eval")
    return os.path.join(EVAL_BASE_DIR, folder)


class _Tee:
    """Write stdout to both console and a log file simultaneously."""
    def __init__(self, path: str):
        self._file = open(path, "w", encoding="utf-8")
        self._stdout = sys.stdout
        sys.stdout = self
    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)
    def flush(self):
        self._stdout.flush()
        self._file.flush()
    def close(self):
        sys.stdout = self._stdout
        self._file.close()
    def __enter__(self):
        return self
    def __exit__(self, *_):
        self.close()


# =============================================================================
# MAIN EVALUATION PIPELINE
# =============================================================================

def run_evaluation(
    weights_path: str,
    modality: str = "visible",
    conf: float = DEFAULT_CONF,
    output_dir: str = None,
    lpips_scorer: object = None,
):
    """
    Orchestrates the full evaluation:
      1. Load model and predict on validation images
      2. Match predictions to ground truth (eval_data)
      3. Compute quality metrics per object and per image (eval_metrics)
      4. Build FP DataFrame for false positive analysis
      5. Export CSVs
      6. Print summary (eval_analysis)

    Returns (df_img, df_obj, df_fp) or None if validation dir not found.
    """
    print("=" * 70)
    print(f"EVALUATION: {modality.upper()}")
    print("=" * 70)

    os.makedirs(EVAL_BASE_DIR, exist_ok=True)
    if output_dir is None:
        output_dir = get_eval_dir(modality)
    os.makedirs(output_dir, exist_ok=True)

    is_synthetic = modality.lower() in SYNTHETIC_MODALITIES

    # ── Load model ───────────────────────────────────────────────
    model = YOLO(weights_path)
    print(f"Loaded model: {weights_path}")

    # ── Locate validation images and labels ──────────────────────
    if modality.lower() == "pid":
        val_img_dir = os.path.join(PID_DATASET_ROOT, "val", "images")
        labels_dir = os.path.join(PID_DATASET_ROOT, "val", "labels")
    else:
        val_img_dir = os.path.join(DATASET_ROOT, "images", modality, "val")
        labels_dir = os.path.join(DATASET_ROOT, "labels", modality, "val")

    if not os.path.exists(val_img_dir):
        print(f"ERROR: Validation directory not found: {val_img_dir}")
        return None

    img_files = sorted([
        f for f in os.listdir(val_img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    print(f"Found {len(img_files)} validation images in {val_img_dir}")

    # ── Load preprocessing metadata ──────────────────────────────
    def _pick_meta(df, target):
        if target in df["modality"].values:
            return target
        print(f"  NOTE: no metadata for '{target}'; falling back to 'visible'.")
        return "visible"

    frame_meta = {}
    if os.path.exists(FRAME_METADATA_CSV):
        df_fm = pd.read_csv(FRAME_METADATA_CSV)
        mm = _pick_meta(df_fm, modality)
        df_fm = df_fm[(df_fm["modality"] == mm) & (df_fm["split"] == "val")]
        frame_meta = {row["flat_name"]: row.to_dict() for _, row in df_fm.iterrows()}
        print(f"Loaded frame metadata: {len(frame_meta)} entries")

    object_meta = {}
    if os.path.exists(OBJECT_METADATA_CSV):
        df_om = pd.read_csv(OBJECT_METADATA_CSV)
        mm = _pick_meta(df_om, modality)
        df_om = df_om[(df_om["modality"] == mm) & (df_om["split"] == "val")]
        for fn, group in df_om.groupby("flat_name"):
            object_meta[fn] = group.to_dict("records")
        print(f"Loaded object metadata: {len(df_om)} objects")

    # ── Run predictions & evaluate ───────────────────────────────
    print(f"\nRunning predictions (conf={conf}, iou_thresh={IOU_THRESHOLD})...")

    per_image_rows = []
    per_object_rows = []
    per_fp_rows = []

    for img_file in img_files:
        flat_name = os.path.splitext(img_file)[0]
        img_path = os.path.join(val_img_dir, img_file)

        # ── Predict ──────────────────────────────────────────────
        results = model.predict(
            img_path, conf=conf, iou=0.45, imgsz=512, verbose=False,
        )
        result = results[0]

        pred_boxes = (result.boxes.xyxy.cpu().numpy()
                      if len(result.boxes) > 0 else np.empty((0, 4)))
        pred_confs = (result.boxes.conf.cpu().numpy()
                      if len(result.boxes) > 0 else np.empty(0))

        # ── Ground truth ─────────────────────────────────────────
        label_path = os.path.join(labels_dir, f"{flat_name}.txt")
        gt_labels = load_yolo_labels(label_path)

        # ── Detection matching ───────────────────────────────────
        ev = evaluate_image(pred_boxes, pred_confs, gt_labels)

        # ── Load images (BGR, 3-channel) ─────────────────────────
        img_bgr = load_image(img_path)
        img_h, img_w = img_bgr.shape[:2] if img_bgr is not None else (512, 512)

        # Load thermal GT for val set only (synthetic modalities only)
        thermal_bgr = None
        if is_synthetic:
            thermal_bgr = load_thermal_gt(flat_name, "val", THERMAL_GT_DIR)

        # ── Full-image quality metrics ───────────────────────────
        if img_bgr is not None:
            iq = compute_image_quality(img_bgr)
        else:
            iq = {k: float("nan") for k in
                  ["brightness", "contrast", "sharpness", "blur", "edge_density"]}

        _prec = ev["precision"]
        _rec = ev["recall"]
        _f1 = (2 * _prec * _rec / (_prec + _rec)) if (_prec + _rec) > 0 else 0.0
        mean_conf = float(np.mean(pred_confs)) if len(pred_confs) > 0 else float("nan")

        img_row = {
            "flat_name": flat_name,
            "modality": modality,
            "num_gt": len(gt_labels),
            "num_pred": len(pred_boxes),
            "tp": ev["tp"],
            "fp": ev["fp"],
            "fn": ev["fn"],
            "precision": _prec,
            "recall": _rec,
            "f1": _f1,
            "mean_conf": mean_conf,
            **iq,
        }

        # Merge frame-level metadata
        if flat_name in frame_meta:
            fm = frame_meta[flat_name]
            img_row["day_night"] = fm.get("day_night", infer_day_night(flat_name))
            img_row["num_people"] = fm.get("num_people", 0)
            img_row["num_overlapping_pairs"] = fm.get("num_overlapping_pairs", 0)
        else:
            img_row["day_night"] = infer_day_night(flat_name)

        # ── Per-object evaluation ────────────────────────────────
        meta_objs = object_meta.get(flat_name, [])
        matched_meta = match_gt_to_metadata(gt_labels, meta_objs)

        _ghost_scores = []
        _halluc_scores = []
        _obj_ssims = []
        _lpips_scores = []

        for gi, (cls, cx, cy, w, h) in enumerate(gt_labels):
            detected = (ev["gt_matched"][gi]
                        if gi < len(ev["gt_matched"]) else False)

            # Confidence of TP prediction matching this GT
            tp_conf = ev["tp_confs"][gi] if detected and gi < len(ev["tp_confs"]) else float("nan")

            obj_row = {
                "flat_name": flat_name,
                "modality": modality,
                "gt_idx": gi,
                "detected": detected,
                "tp_conf": tp_conf,
                "gt_cx": round(cx, 4),
                "gt_cy": round(cy, 4),
                "gt_w": round(w, 4),
                "gt_h": round(h, 4),
            }

            # ── Metadata (size, occlusion, etc.) ─────────────────
            mo = matched_meta[gi]
            if mo is not None:
                obj_row["size_category"] = mo.get("size_category", "unknown")
                obj_row["bbox_height_px"] = mo.get("bbox_height_px", 0)
                obj_row["bbox_width_px"] = mo.get("bbox_width_px", 0)
                obj_row["bbox_area_px"] = mo.get("bbox_area_px", 0)
                obj_row["occlusion"] = mo.get("occlusion", 0)
                obj_row["truncated"] = mo.get("truncated", 0)
                obj_row["has_overlap"] = mo.get("has_overlap", False)
                obj_row["dist_from_center"] = mo.get("dist_from_center", 0)
                obj_row["day_night"] = mo.get("day_night", infer_day_night(flat_name))
                obj_row["aspect_ratio"] = mo.get("aspect_ratio", 0)
            else:
                obj_row["day_night"] = img_row.get("day_night", infer_day_night(flat_name))
                bbox_h_px = h * img_h
                bbox_w_px = w * img_w
                if bbox_h_px <= 32:
                    size_cat = "small"
                elif bbox_h_px <= 96:
                    size_cat = "medium"
                else:
                    size_cat = "large"
                obj_row["size_category"] = size_cat
                obj_row["bbox_height_px"] = round(bbox_h_px, 1)
                obj_row["bbox_width_px"] = round(bbox_w_px, 1)
                obj_row["bbox_area_px"] = round(bbox_h_px * bbox_w_px, 1)
                obj_row["aspect_ratio"] = round(bbox_w_px / bbox_h_px, 4) if bbox_h_px > 0 else 0

            # ── Per-object quality features (on BGR image) ───────
            if img_bgr is not None:
                px1, py1, px2, py2 = yolo_to_xyxy(cx, cy, w, h, img_w)
                px1 = max(0, int(round(px1)))
                py1 = max(0, int(round(py1)))
                px2 = min(img_w, int(round(px2)))
                py2 = min(img_h, int(round(py2)))
                qf = compute_object_quality_features(
                    img_bgr, px1, py1, px2, py2, img_h, img_w)
                obj_row.update(qf)

                # ── Ghost / SSIM (compare vs thermal GT) ─────────
                if thermal_bgr is not None:
                    gen_crop = img_bgr[py1:py2, px1:px2]
                    th, tw = thermal_bgr.shape[:2]
                    tx1 = max(0, int(round((cx - w/2) * tw)))
                    ty1 = max(0, int(round((cy - h/2) * th)))
                    tx2 = min(tw, int(round((cx + w/2) * tw)))
                    ty2 = min(th, int(round((cy + h/2) * th)))
                    thermal_crop = thermal_bgr[ty1:ty2, tx1:tx2]
                    gf = compute_ghost_features(gen_crop, thermal_crop)
                    obj_row.update(gf)

                    # ── LPIPS ────────────────────────────────────
                    if lpips_scorer is not None:
                        lp = lpips_scorer.score(gen_crop, thermal_crop)
                        obj_row["object_lpips"] = lp
                    else:
                        obj_row["object_lpips"] = float("nan")
                else:
                    for gk in ("object_ssim", "ghost_score",
                               "hallucination_score", "object_lpips"):
                        obj_row[gk] = float("nan")
            else:
                for key in ("object_brightness", "object_contrast",
                            "bg_brightness", "bg_contrast", "fg_bg_diff",
                            "object_edge_strength", "object_blur",
                            "object_ssim", "ghost_score",
                            "hallucination_score", "object_lpips"):
                    obj_row[key] = float("nan")

            per_object_rows.append(obj_row)

            # Collect for image-level averaging
            for _k, _lst in [("ghost_score", _ghost_scores),
                              ("hallucination_score", _halluc_scores),
                              ("object_ssim", _obj_ssims),
                              ("object_lpips", _lpips_scores)]:
                v = obj_row.get(_k)
                if v is not None and not np.isnan(v):
                    _lst.append(v)

        # ── Image-level ghost/LPIPS averages ─────────────────────
        img_row["ghost_score"] = float(np.mean(_ghost_scores)) if _ghost_scores else float("nan")
        img_row["hallucination_score"] = float(np.mean(_halluc_scores)) if _halluc_scores else float("nan")
        img_row["object_ssim"] = float(np.mean(_obj_ssims)) if _obj_ssims else float("nan")
        img_row["lpips_mean"] = float(np.mean(_lpips_scores)) if _lpips_scores else float("nan")
        per_image_rows.append(img_row)

        # ── Collect FP data ──────────────────────────────────────
        for fp_info in ev["fp_boxes"]:
            fp_row = {
                "flat_name": flat_name,
                "modality": modality,
                "conf": fp_info["conf"],
                "day_night": img_row.get("day_night", "unknown"),
            }
            # Extract quality features at FP box location
            if img_bgr is not None:
                bx = fp_info["box"]
                fx1 = max(0, int(round(bx[0])))
                fy1 = max(0, int(round(bx[1])))
                fx2 = min(img_w, int(round(bx[2])))
                fy2 = min(img_h, int(round(bx[3])))
                fp_qf = compute_object_quality_features(
                    img_bgr, fx1, fy1, fx2, fy2, img_h, img_w)
                fp_row["fp_brightness"] = fp_qf.get("object_brightness", float("nan"))
                fp_row["fp_contrast"] = fp_qf.get("object_contrast", float("nan"))
                fp_row["fp_edge_strength"] = fp_qf.get("object_edge_strength", float("nan"))
                fp_row["fp_blur"] = fp_qf.get("object_blur", float("nan"))
            per_fp_rows.append(fp_row)

    # ── Export DataFrames ────────────────────────────────────────
    df_img = pd.DataFrame(per_image_rows)
    df_obj = pd.DataFrame(per_object_rows)
    df_fp = pd.DataFrame(per_fp_rows) if per_fp_rows else pd.DataFrame()

    img_csv = os.path.join(output_dir, f"per_image_results_{modality}.csv")
    df_img.to_csv(img_csv, index=False)
    print(f"\nSaved per-image results: {img_csv} ({len(df_img)} rows)")

    obj_csv = os.path.join(output_dir, f"per_object_results_{modality}.csv")
    df_obj.to_csv(obj_csv, index=False)
    print(f"Saved per-object results: {obj_csv} ({len(df_obj)} rows)")

    if len(df_fp) > 0:
        fp_csv = os.path.join(output_dir, f"per_fp_results_{modality}.csv")
        df_fp.to_csv(fp_csv, index=False)
        print(f"Saved FP results: {fp_csv} ({len(df_fp)} rows)")

    # ── Subgroup summary & print ─────────────────────────────────
    summary_csv = compute_subgroup_summary(df_img, df_obj, modality, output_dir)
    print(f"Saved subgroup summary: {summary_csv}")

    print_full_summary(df_img, df_obj, df_fp, modality)

    # ── Plots ────────────────────────────────────────────────────
    print("\nGenerating plots...")
    plot_sharpness_size_heatmap(df_obj, output_dir)
    plot_fp_analysis(df_fp, df_img, output_dir)

    if is_synthetic:
        plot_lpips_vs_confidence(df_obj, os.path.join(output_dir, "appendix"))

    # Example crops for thesis — 3 images per quality-metric extreme
    # For visible modality: show PID examples (same images, better quality for visual examples)
    if modality.lower() == "visible":
        example_img_dir = os.path.join(PID_DATASET_ROOT, "val", "images")
    else:
        example_img_dir = val_img_dir

    thermal_gt_val = os.path.join(THERMAL_GT_DIR, "val") if is_synthetic else None
    plot_example_crops(
        df_obj,
        image_dir=example_img_dir,
        output_dir=os.path.join(output_dir, "examples"),
        thermal_dir=thermal_gt_val,
        is_synthetic=is_synthetic,
        n_examples=3,
        report_dir=os.path.join(output_dir, "images_report"),
    )

    print(f"\nAll results saved to: {output_dir}")
    return df_img, df_obj, df_fp


# =============================================================================
# LOAD mAP50 FROM TRAINING RESULTS
# =============================================================================

def _load_map50(modality: str) -> float:
    """Read mAP50 from the last epoch of the training results.csv."""
    run_name = "pid" if modality.lower() == "pid" else f"kaist_{modality}"
    csv_path = os.path.join(RUNS_DIR, run_name, "results.csv")
    if not os.path.exists(csv_path):
        return float("nan")
    try:
        df = pd.read_csv(csv_path)
        col = [c for c in df.columns if "mAP50" in c and "mAP50-95" not in c]
        if col:
            return float(df[col[0]].iloc[-1])
    except Exception:
        pass
    return float("nan")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate YOLOv8 with subgroup analysis"
    )
    parser.add_argument(
        "--modality", type=str, default=None,
        choices=["visible", "PI-GAN_gen", "pid"],
        help="Modality to evaluate. Omit to evaluate all three together.",
    )
    parser.add_argument(
        "--weights", type=str, default=None,
        help="Path to model weights (single modality only)."
    )
    parser.add_argument(
        "--conf", type=float, default=DEFAULT_CONF,
        help=f"Confidence threshold (default: {DEFAULT_CONF})"
    )
    args = parser.parse_args()

    def _weights_for(mod):
        run_name = "pid" if mod.lower() == "pid" else f"kaist_{mod}"
        return os.path.join(RUNS_DIR, run_name, "weights", "best.pt")

    # ── Initialise LPIPS scorer (once for all modalities) ────────
    try:
        lpips_scorer = LPIPSScorer()
        print("LPIPS scorer loaded.")
    except Exception as e:
        print(f"LPIPS not available: {e}")
        lpips_scorer = None

    # ── Single modality ──────────────────────────────────────────
    if args.modality is not None:
        out_dir = get_eval_dir(args.modality)
        os.makedirs(out_dir, exist_ok=True)
        log_path = os.path.join(out_dir, f"eval_log_{args.modality}.txt")
        weights = args.weights or _weights_for(args.modality)

        with _Tee(log_path):
            if not os.path.exists(weights):
                print(f"ERROR: Weights not found: {weights}")
                print("Train the model first with train.py")
            else:
                result = run_evaluation(
                    weights, args.modality, args.conf,
                    output_dir=out_dir, lpips_scorer=lpips_scorer,
                )
                if result is not None:
                    df_img, df_obj, df_fp = result
                    plot_quality_vs_performance(
                        {args.modality: df_img}, out_dir,
                        appendix_dir=os.path.join(out_dir, "appendix"),
                    )
                    print(f"\nLog saved to: {log_path}")

    # ── All three modalities ─────────────────────────────────────
    else:
        ALL_MODALITIES = ["visible", "PI-GAN_gen", "pid"]
        collected_img = {}
        collected_obj = {}
        collected_fp = {}
        fid_results = {}
        map_results = {}

        # Initialise FID computer once
        try:
            fid_computer = FIDComputer()
            print("FID computer loaded.\n")
        except Exception as e:
            print(f"FID not available: {e}\n")
            fid_computer = None

        for mod in ALL_MODALITIES:
            out_dir = get_eval_dir(mod)
            os.makedirs(out_dir, exist_ok=True)
            log_path = os.path.join(out_dir, f"eval_log_{mod}.txt")
            w = _weights_for(mod)

            with _Tee(log_path):
                if not os.path.exists(w):
                    print(f"SKIP {mod}: weights not found at {w}")
                else:
                    result = run_evaluation(
                        w, mod, args.conf,
                        output_dir=out_dir, lpips_scorer=lpips_scorer,
                    )
                    if result is not None:
                        df_img, df_obj, df_fp = result
                        collected_img[mod] = df_img
                        collected_obj[mod] = df_obj
                        collected_fp[mod] = df_fp

                        # Single-modality quality plots
                        plot_quality_vs_performance(
                            {mod: df_img}, out_dir,
                            appendix_dir=os.path.join(out_dir, "appendix"),
                        )

                        # FID: compare synthetic val images vs thermal val
                        if mod.lower() in SYNTHETIC_MODALITIES and fid_computer is not None:
                            if mod.lower() == "pid":
                                gen_dir = os.path.join(PID_DATASET_ROOT, "val", "images")
                            else:
                                gen_dir = os.path.join(DATASET_ROOT, "images", mod, "val")
                            thermal_dir = os.path.join(THERMAL_GT_DIR, "val")
                            fid_val = fid_computer.compute(gen_dir, thermal_dir)
                            fid_results[mod] = fid_val
                            print(f"  FID ({mod}): {fid_val:.2f}")

                        map_results[mod] = _load_map50(mod)
                        print(f"  mAP50 ({mod}): {map_results[mod]:.4f}")
                        print(f"\nLog saved to: {log_path}")

        if collected_img:
            all_dir = get_eval_dir("_all")
            os.makedirs(all_dir, exist_ok=True)
            log_all = os.path.join(all_dir, "eval_log_all.txt")
            with _Tee(log_all):
                print("Generating combined quality-vs-performance plots...")
                plot_quality_vs_performance(
                    collected_img, all_dir,
                    appendix_dir=os.path.join(all_dir, "appendix"),
                )

                if fid_results and map_results:
                    print("\nGenerating FID vs mAP plot...")
                    plot_fid_vs_map_bar(fid_results, map_results, all_dir)

                    print("\n" + "=" * 60)
                    print("FID vs mAP50 COMPARISON")
                    print("=" * 60)
                    print(f"  {'Modality':<15}  {'FID':>8}  {'mAP50':>8}")
                    print(f"  {'-'*15}  {'-'*8}  {'-'*8}")
                    for mod in ALL_MODALITIES:
                        fid_val = fid_results.get(mod, float("nan"))
                        map_val = map_results.get(mod, float("nan"))
                        fid_str = f"{fid_val:.2f}" if not np.isnan(fid_val) else "n/a"
                        map_str = f"{map_val:.4f}" if not np.isnan(map_val) else "n/a"
                        print(f"  {mod:<15}  {fid_str:>8}  {map_str:>8}")

                print(f"\nCombined plots saved to: {all_dir}")
                print(f"Log saved to: {log_all}")
        else:
            print("No modalities could be evaluated.")
