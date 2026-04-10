"""
=============================================================================
eval_analysis.py — Statistical Analysis and Summary Tables
=============================================================================

Produces all statistical analyses for the thesis:

  Section 1: Overall Detection Performance (GT, TP, FP, FN, P, R, F1)
  Section 2: Day/Night/Size Recall Breakdown
  Section 3: Image Quality Metrics (scaled 0–1)
  Section 3b: Ghost & Hallucination Metrics (detected vs missed, day vs night)
  Section 3c: LPIPS Metrics (detected vs missed, day vs night)
  Section 4: FID Summary (overall + day/night) vs mAP50
  Section 5: Correlation Table (all metrics vs recall, F1, confidence)
  Section 6: Sharpness × Size Cross-Table
  Section 7: False Positive Analysis (confidence breakdown, image conditions)
  Section 8: Performance by Condition (low/high splits)
  Section 9: Root Cause Analysis (feature importance via logistic regression)

Functions:
    compute_subgroup_summary(df_img, df_obj, modality, output_dir)
    print_full_summary(df_img, df_obj, df_fp, modality, fid_results)
    compute_sharpness_size_table(df_obj)
    analyze_false_positives(df_fp, df_img)
    root_cause_analysis(df_obj)
=============================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional


# =============================================================================
# SUBGROUP SUMMARY
#   Aggregates detection metrics by category for structured comparison.
#   Each row = one subgroup level with count, detected, missed, recall.
# =============================================================================

def compute_subgroup_summary(
    df_img: pd.DataFrame,
    df_obj: pd.DataFrame,
    modality: str,
    output_dir: str,
) -> str:
    """
    Compute detection metrics broken down by subgroup.

    Subgroups: day_night, size_category, occlusion, truncated,
    has_overlap, crowd_density.

    Returns path to saved CSV.
    """
    import os
    rows = []

    def add_group(group_name, col, df):
        if col not in df.columns:
            return
        for val, sub in df.groupby(col):
            n = len(sub)
            det = int(sub["detected"].sum())
            rows.append({
                "subgroup_type": group_name,
                "subgroup_value": str(val),
                "count": n,
                "detected": det,
                "missed": n - det,
                "recall": round(det / n, 4) if n > 0 else 0.0,
            })

    add_group("day_night", "day_night", df_obj)
    add_group("size_category", "size_category", df_obj)
    add_group("occlusion", "occlusion", df_obj)
    add_group("truncated", "truncated", df_obj)
    add_group("has_overlap", "has_overlap", df_obj)

    # Crowd density bins from per-image data
    if "num_people" in df_img.columns:
        df_c = df_img.copy()
        df_c["crowd_bin"] = df_c["num_people"].apply(
            lambda x: str(int(x)) if x <= 4 else "5+"
        )
        for val, sub in df_c.groupby("crowd_bin"):
            tp_sum = int(sub["tp"].sum())
            fn_sum = int(sub["fn"].sum())
            total_gt = tp_sum + fn_sum
            rec = tp_sum / total_gt if total_gt > 0 else 0
            rows.append({
                "subgroup_type": "crowd_density",
                "subgroup_value": val,
                "count": total_gt,
                "detected": tp_sum,
                "missed": fn_sum,
                "recall": round(rec, 4),
            })

    path = os.path.join(output_dir, f"subgroup_summary_{modality}.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


# =============================================================================
# FULL SUMMARY PRINTOUT
#   Prints all evaluation sections to stdout (captured by _Tee into log file).
# =============================================================================

def print_full_summary(
    df_img: pd.DataFrame,
    df_obj: pd.DataFrame,
    df_fp: pd.DataFrame,
    modality: str = "",
    fid_results: Optional[Dict] = None,
):
    """
    Print comprehensive evaluation summary covering all thesis analyses.
    """
    from scipy import stats

    tag = modality.upper() if modality else "MODEL"
    sep = "=" * 60

    # ── Section 1: OVERALL DETECTION PERFORMANCE ─────────────────────────
    # Precision = TP / (TP + FP) — of all predictions, how many are correct?
    # Recall    = TP / (TP + FN) — of all GT objects, how many are found?
    # F1        = 2 * P * R / (P + R) — harmonic mean of P and R
    total_tp = int(df_img["tp"].sum())
    total_fp = int(df_img["fp"].sum())
    total_fn = int(df_img["fn"].sum())
    total_gt = total_tp + total_fn
    prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    rec = total_tp / total_gt if total_gt > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    print(f"\n{sep}")
    print(f"1. OVERALL DETECTION PERFORMANCE — {tag}")
    print(sep)
    print(f"  {'GT objects':<20}: {total_gt}")
    print(f"  {'True Positives':<20}: {total_tp}")
    print(f"  {'False Positives':<20}: {total_fp}")
    print(f"  {'False Negatives':<20}: {total_fn}")
    print(f"  {'Precision':<20}: {prec:.4f}")
    print(f"  {'Recall':<20}: {rec:.4f}")
    print(f"  {'F1 Score':<20}: {f1:.4f}")
    print(f"")
    print(f"  How each value is computed:")
    print(f"  GT objects     = total ground-truth pedestrian bounding boxes in the validation set")
    print(f"  True Positive  = a predicted box that overlaps a GT box by IoU ≥ 0.5 at conf ≥ 0.25;")
    print(f"                   each GT box can only be matched once (highest-conf prediction wins)")
    print(f"  False Positive = a predicted box that did NOT match any GT box (spurious detection)")
    print(f"  False Negative = a GT box that was NOT matched by any prediction (missed pedestrian)")
    print(f"  Precision      = TP / (TP + FP)  — of all predictions made, what fraction were correct")
    print(f"  Recall         = TP / (TP + FN)  — of all real pedestrians, what fraction were found")
    print(f"  F1 Score       = 2 × Precision × Recall / (Precision + Recall)  — harmonic mean")

    # ── Section 2: DAY / NIGHT / SIZE RECALL ─────────────────────────────
    print(f"\n{sep}")
    print(f"2. DAY / NIGHT / SIZE RECALL — {tag}")
    print(sep)

    if "day_night" in df_obj.columns:
        for label in ["day", "night"]:
            sub = df_obj[df_obj["day_night"] == label]
            n = len(sub)
            det = int(sub["detected"].sum()) if n > 0 else 0
            val = f"{det/n:.4f}" if n > 0 else "n/a"
            print(f"  {'Recall ' + label:<20}: {val}  ({det}/{n})")

    if "size_category" in df_obj.columns:
        for cat in ["small", "medium", "large"]:
            sub = df_obj[df_obj["size_category"] == cat]
            n = len(sub)
            det = int(sub["detected"].sum()) if n > 0 else 0
            val = f"{det/n:.4f}" if n > 0 else "n/a"
            print(f"  {'Recall ' + cat:<20}: {val}  ({det}/{n})")
        print(f"  (Size defined by bbox height at 512px resolution: "
              f"small ≤ 32px, medium 33–96px, large >96px)")
    print(f"")
    print(f"  How each value is computed:")
    print(f"  Recall [group] = pedestrians detected in that group / all GT pedestrians in that group")
    print(f"  (det/n) shown  = detected count / total GT count in that group")
    print(f"  day/night      = inferred from filename (set 01–11 = day, set 12–20 = night in KAIST)")

    # ── Section 3: IMAGE QUALITY METRICS ─────────────────────────────────
    # All metrics scaled to [0, 1] for comparability:
    # brightness, contrast: divide by 255 (natural max)
    # sharpness, blur, edge_density: min-max normalisation within dataset
    quality_cols = ["brightness", "contrast", "sharpness", "edge_density", "blur"]
    ghost_cols = ["ghost_score", "hallucination_score", "object_ssim"]
    lpips_cols = ["lpips_mean"]

    available = [c for c in quality_cols if c in df_img.columns
                 and df_img[c].notna().any()]
    ghost_avail = [c for c in ghost_cols if c in df_img.columns
                   and df_img[c].notna().any()]
    lpips_avail = [c for c in lpips_cols if c in df_img.columns
                   and df_img[c].notna().any()]

    def _scale(s, col):
        s = s.dropna()
        if col in ("brightness", "contrast"):
            return s / 255.0
        if col in ("ghost_score", "hallucination_score", "object_ssim",
                    "lpips_mean", "object_lpips"):
            return s  # already [0, 1]
        mn, mx = s.min(), s.max()
        return (s - mn) / (mx - mn + 1e-9)

    # ── Section 3b: GHOST / HALLUCINATION ────────────────────────────────
    print(f"\n{sep}")
    print(f"3b. GHOST & HALLUCINATION METRICS — {tag}")
    print(sep)
    print(f"  All values shown as percentages (metric × 100).")
    print(f"  ghost_score: pixel-level artefact between generated and thermal GT (0%=identical, 100%=completely different)")
    print(f"  hallucination_score: fraction of generated edges absent from thermal GT (0%=none invented, 100%=all invented)")
    print(f"  object_ssim: structural similarity to thermal GT (0%=different, 100%=identical)")
    if ghost_avail:
        print()
        for col in ghost_avail:
            s = df_img[col].dropna() * 100
            print(f"  {col:<25}: mean={s.mean():.1f}%  std={s.std():.1f}%")

        # Per-object level
        obj_ghost = [c for c in ghost_cols if c in df_obj.columns
                     and df_obj[c].notna().any()]
        if obj_ghost:
            print(f"\n  Per-object level (all pedestrian bounding boxes):")
            for col in obj_ghost:
                s = df_obj[col].dropna() * 100
                print(f"    {col:<25}: mean={s.mean():.1f}%  std={s.std():.1f}%  "
                      f"median={s.median():.1f}%")

        # By day/night
        if "day_night" in df_img.columns:
            print(f"\n  By day/night (per-image averages):")
            for dn in ["day", "night"]:
                sub = df_img[df_img["day_night"] == dn]
                if len(sub) == 0:
                    continue
                for col in ghost_avail:
                    s = sub[col].dropna() * 100
                    if len(s) > 0:
                        print(f"    {dn.capitalize():<8} {col:<25}: "
                              f"mean={s.mean():.1f}%  std={s.std():.1f}%")

        # Detected vs missed
        if "ghost_score" in df_obj.columns and "detected" in df_obj.columns:
            det_obj = df_obj[df_obj["detected"] == True]
            mis_obj = df_obj[df_obj["detected"] == False]

            METRIC_PLAIN = {
                "ghost_score": (
                    "Ghost Score",
                    "how much the generated image differs from the original thermal "
                    "(pixel-level artefacts). 0% = perfect match, 100% = completely different. "
                    "Lower = better quality, easier to detect.",
                ),
                "hallucination_score": (
                    "Hallucination Score",
                    "fraction of edges in the generated image that do NOT exist in the real "
                    "thermal. 0% = all edges are real, 100% = all edges are invented by the "
                    "model. High score means the model added fake structure.",
                ),
                "object_ssim": (
                    "SSIM (Structural Similarity)",
                    "how structurally close the generated pedestrian crop is to the real "
                    "thermal image. 0% = completely different, 100% = identical. "
                    "Higher = better quality, easier to detect.",
                ),
            }

            print(f"\n  ── Detected vs Missed Pedestrians ──────────────────────────────")
            for col in obj_ghost:
                if col not in METRIC_PLAIN:
                    continue
                short_name, description = METRIC_PLAIN[col]
                d_vals = det_obj[col].dropna()
                m_vals = mis_obj[col].dropna()
                if len(d_vals) == 0 or len(m_vals) == 0:
                    continue

                d_mean_pct = d_vals.mean() * 100
                d_med_pct  = d_vals.median() * 100
                m_mean_pct = m_vals.mean() * 100
                m_med_pct  = m_vals.median() * 100
                delta      = d_mean_pct - m_mean_pct

                print(f"\n  {short_name}")
                print(f"  Definition: {description}")
                print(f"    Detected pedestrians : mean {d_mean_pct:.1f}%  "
                      f"(median {d_med_pct:.1f}%)   n={len(d_vals):,}")
                print(f"    Missed   pedestrians : mean {m_mean_pct:.1f}%  "
                      f"(median {m_med_pct:.1f}%)   n={len(m_vals):,}")
                print(f"    Δ (detected − missed): {delta:+.1f} pp")
    else:
        print("  (no thermal GT available — ghost metrics skipped)")

    # ── Section 3c: LPIPS ────────────────────────────────────────────────
    if lpips_avail:
        print(f"\n{sep}")
        print(f"3c. LPIPS (PERCEPTUAL SIMILARITY) — {tag}")
        print(sep)
        print(f"  LPIPS = Learned Perceptual Image Patch Similarity.")
        print(f"  Measures how different the generated image LOOKS compared to the real thermal,")
        print(f"  using deep neural-network features (not just pixel differences).")
        print(f"  Scale: 0.0 = perceptually identical, ~0.5+ = very different.")
        print(f"  Lower LPIPS = higher perceptual quality = should be easier to detect.")
        print()
        for col in lpips_avail:
            s = df_img[col].dropna()
            print(f"  Per-image mean LPIPS : {s.mean():.3f}  (std {s.std():.3f})")

        if "object_lpips" in df_obj.columns:
            s = df_obj["object_lpips"].dropna()
            if len(s) > 0:
                print(f"  Per-object mean LPIPS: {s.mean():.3f}  "
                      f"(std {s.std():.3f},  median {s.median():.3f})")

            if "detected" in df_obj.columns:
                det_s = df_obj[df_obj["detected"] == True]["object_lpips"].dropna()
                mis_s = df_obj[df_obj["detected"] == False]["object_lpips"].dropna()
                if len(det_s) > 0 and len(mis_s) > 0:
                    delta = det_s.mean() - mis_s.mean()
                    print(f"\n  LPIPS — Detected vs Missed pedestrians:")
                    print(f"  (Lower = generated image looks more like real thermal = easier to detect)")
                    print(f"    Detected pedestrians : mean {det_s.mean():.3f}  "
                          f"(median {det_s.median():.3f})   n={len(det_s):,}")
                    print(f"    Missed   pedestrians : mean {mis_s.mean():.3f}  "
                          f"(median {mis_s.median():.3f})   n={len(mis_s):,}")
                    print(f"    Δ (detected − missed): {delta:+.3f}  "
                          f"({'detected look more realistic' if delta < 0 else 'no quality advantage for detected'})")

    # ── Section 4: FID SUMMARY ───────────────────────────────────────────
    if fid_results:
        print(f"\n{sep}")
        print(f"4. FID (FRÉCHET INCEPTION DISTANCE) — {tag}")
        print(sep)
        for key, val in fid_results.items():
            if not np.isnan(val):
                print(f"  {key:<25}: {val:.2f}")

    # ── Section 5: CORRELATION TABLE ─────────────────────────────────────
    # Pearson correlation: measures linear relationship between metric and
    # detection performance. r ∈ [-1, 1].
    # r > 0 → metric increases with performance (good quality helps)
    # r < 0 → metric increases as performance drops (bad quality hurts)
    # * = statistically significant (p < 0.05)
    all_corr_cols = [c for c in ["sharpness", "ghost_score", "object_ssim", "lpips_mean"]
                     if c in (available + ghost_avail + lpips_avail)]
    perf_cols = [c for c in ["recall", "f1", "mean_conf"] if c in df_img.columns]

    print(f"\n{sep}")
    print(f"5. CORRELATION WITH DETECTION PERFORMANCE — {tag}")
    print(sep)
    print(f"  Pearson r ∈ [-1, 1].  +r → metric increases with better performance;  -r → metric increases as performance drops.")
    print(f"  Recall = detected GT objects / total GT objects.  F1 = harmonic mean of Precision & Recall.")
    print(f"  mean_conf = mean YOLO confidence score on TP predictions.  (* = statistically significant at p < 0.05)")
    print(f"  Metrics: sharpness=Laplacian variance; ghost_score=pixel residual vs thermal GT [0-1];"
          f" object_ssim=SSIM vs thermal GT [0-1]; lpips_mean=perceptual distance vs thermal GT [0-1]")
    print()
    if all_corr_cols and perf_cols:
        header = f"  {'Metric':<25}"
        for pc in perf_cols:
            header += f"  {'Corr w/ ' + pc:>16}"
        print(header)
        print(f"  {'-'*23}" + f"  {'-'*16}" * len(perf_cols))

        df_valid = df_img.dropna(subset=perf_cols)
        for col in all_corr_cols:
            if col not in df_valid.columns:
                continue
            sub = df_valid.dropna(subset=[col])
            if len(sub) < 10 or sub[col].std() < 1e-9:
                continue
            parts = f"  {col:<25}"
            for pc in perf_cols:
                r, p = stats.pearsonr(sub[col], sub[pc])
                sig = "*" if p < 0.05 else ""
                parts += f"  {r:>+.4f}{sig:<1}{'':>10}"
            print(parts)
        print("  (* p < 0.05)")
    else:
        print("  (insufficient data)")
    print(f"")
    print(f"  How each value is computed:")
    print(f"  Each row = Pearson correlation between that metric (per image) and the performance column.")
    print(f"  Pearson r = cov(X, Y) / (std(X) × std(Y))  — measures linear association, range −1 to +1.")
    print(f"  Unit of analysis = one row per validation IMAGE (metrics averaged over all objects in that image).")
    print(f"  p-value tests whether the correlation is statistically different from zero; * means p < 0.05.")

    # ── Section 6: SHARPNESS × SIZE TABLE ────────────────────────────────
    _print_sharpness_size_table(df_obj, tag, sep)

    # ── Section 7: FALSE POSITIVE ANALYSIS ───────────────────────────────
    _print_fp_analysis(df_fp, df_img, tag, sep)

    # ── Section 8: PERFORMANCE BY CONDITION (low/high splits) ────────────
    all_condition_cols = ["sharpness", "hallucination_score", "object_ssim"]
    if "lpips_mean" in df_img.columns:
        all_condition_cols.append("lpips_mean")
    for col in all_condition_cols:
        _condition_breakdown(df_img, col, tag, sep)

    # ── Section 9: ROOT CAUSE ANALYSIS ───────────────────────────────────
    _print_root_cause(df_obj, tag, sep)

    print(f"\n{sep}\n")


# =============================================================================
# SHARPNESS × SIZE CROSS-TABLE
#   Does sharpness affect detection differently for small vs large objects?
#
#   Sharpness = Var(Laplacian(luminance_crop)):
#     High → many sharp edges → crisp object
#     Low  → smooth gradients → blurry object
#
#   Split at the MEDIAN sharpness into "low" and "high" groups.
#   For each cell (sharpness_level × size_category), compute recall.
#
#   Thesis hypothesis: small objects have few pixels, so blur destroys
#   discriminative features → sharpness matters MORE for small objects.
# =============================================================================

def _print_sharpness_size_table(df_obj, tag, sep):
    """Print the sharpness × size × day/night cross-table."""
    if "object_blur" not in df_obj.columns or "size_category" not in df_obj.columns:
        return
    if df_obj["object_blur"].isna().all():
        return

    print(f"\n{sep}")
    print(f"6. SHARPNESS × SIZE RECALL TABLE — {tag}")
    print(sep)

    # "object_blur" is actually Laplacian variance = SHARPNESS
    median_sharp = df_obj["object_blur"].median()
    df = df_obj.copy()
    df["sharp_level"] = df["object_blur"].apply(
        lambda x: "high" if x >= median_sharp else "low"
    )

    subsets = [("Overall", df)]

    print(f"  (median sharpness = {median_sharp:.2f}  |  sharpness = Laplacian variance of luminance)")
    print(f"  (size: small = bbox height ≤ 32px, medium = 33–96px, large = >96px, at 512px resolution)")

    for label, sub in subsets:
        print(f"\n  {label}:")
        print(f"    {'':>15}  {'Small (≤32px)':>14}  {'Medium (33-96px)':>16}  {'Large (>96px)':>14}")
        print(f"    {'':>15}  {'-'*14}  {'-'*16}  {'-'*14}")

        for slevel in ["low", "high"]:
            parts = f"    {slevel + ' sharpness':>15}"
            for scat in ["small", "medium", "large"]:
                s = sub[(sub["sharp_level"] == slevel) & (sub["size_category"] == scat)]
                n = len(s)
                det = int(s["detected"].sum()) if n > 0 else 0
                r = f"{det/n:.4f}" if n > 0 else "n/a"
                parts += f"  {r + f' ({n})':>14}"
            print(parts)

    print(f"")
    print(f"  How each value is computed:")
    print(f"  Cell value     = detected / total GT objects in that (sharpness, size) group")
    print(f"  (n)            = total GT pedestrians in that cell")
    print(f"  Sharpness      = Laplacian variance of the object crop luminance channel;")
    print(f"                   low/high split at the MEDIAN sharpness across all objects")
    print(f"  Object size    = bounding-box height at 512×512px: small ≤ 32px, medium 33–96px, large > 96px")


# =============================================================================
# FALSE POSITIVE ANALYSIS
#   WHY does the detector hallucinate?
#
#   We analyse:
#   1. Confidence distribution of FPs:
#      - Low conf (0.25–0.5): detector is uncertain, less worrying
#      - High conf (>0.5): detector is strongly fooled
#      high_conf_rate = count(conf > 0.5) / total_FP
#
#   2. Image conditions that produce FPs:
#      - Pearson r(FP_count_per_image, image_quality_metric)
#      - Day vs Night FP rate
#
#   3. Per-FP crop properties:
#      - Brightness, contrast, edge_strength at FP box locations
#      - Compare with TP crop properties: are FPs brighter? More edges?
# =============================================================================

def _print_fp_analysis(df_fp, df_img, tag, sep):
    """Print false positive summary: FPPI and brief observations."""
    if df_fp is None or len(df_fp) == 0:
        return

    print(f"\n{sep}")
    print(f"7. FALSE POSITIVE ANALYSIS — {tag}")
    print(sep)

    total_fp = len(df_fp)
    n_images = len(df_img)
    fppi = total_fp / n_images if n_images > 0 else 0.0
    print(f"  Total FP        : {total_fp}")
    print(f"  Total images    : {n_images}")
    print(f"  FPPI            : {fppi:.4f}  (false positives per image)")

    # Day vs Night FPPI
    if "day_night" in df_img.columns:
        print(f"\n  FPPI by day/night:")
        for dn in ["day", "night"]:
            sub = df_img[df_img["day_night"] == dn]
            fp_count = int(sub["fp"].sum()) if "fp" in sub.columns else 0
            n = len(sub)
            fppi_dn = fp_count / n if n > 0 else 0
            print(f"    {dn.capitalize():<8}: {fppi_dn:.4f}  ({fp_count} FPs / {n} images)")

    # Confidence observation
    if "conf" in df_fp.columns:
        high_conf = (df_fp["conf"] > 0.5).sum()
        pct = 100 * high_conf / total_fp if total_fp > 0 else 0
        print(f"\n  {pct:.1f}% of FPs had confidence > 0.5 (high-confidence errors)")

    print(f"")
    print(f"  How each value is computed:")
    print(f"  FPPI           = total false positives / total validation images")
    print(f"                   (average number of spurious detections per image)")
    print(f"  FPPI day/night = sum of FPs in day (or night) images / number of day (or night) images")
    print(f"  FP             = a predicted box with conf ≥ 0.25 that did not overlap any GT box at IoU ≥ 0.5")
    print(f"  High-conf FPs  = FPs where the model score > 0.5 (the detector was strongly but wrongly confident)")


# =============================================================================
# PERFORMANCE BY CONDITION (LOW/HIGH SPLIT)
#   For each quality metric, split images at the median into "low" and "high".
#   Report Recall and F1 for each half, broken down by Overall / Day / Night.
#
#   This answers: if image quality X is low, does detection get worse?
# =============================================================================

def _condition_breakdown(df, col, tag, sep):
    """Print performance breakdown by low/high split of a metric."""
    if col not in df.columns or df[col].isna().all():
        return

    print(f"\n{sep}")
    print(f"8. PERFORMANCE BY {col.upper()} — {tag}")
    print(sep)

    median_val = df[col].median()
    df_copy = df.copy()
    df_copy["_bin"] = df_copy[col].apply(lambda x: "high" if x >= median_val else "low")

    subsets = [("Overall", df_copy)]
    if "day_night" in df_copy.columns:
        subsets += [
            ("Day", df_copy[df_copy["day_night"] == "day"]),
            ("Night", df_copy[df_copy["day_night"] == "night"]),
        ]

    print(f"  (median={median_val:.2f}; 'low'=below, 'high'=at or above)")
    print(f"  {'Group':<10}  {'Low Recall':>10}  {'High Recall':>11}  "
          f"{'Low F1':>8}  {'High F1':>8}  {'N low':>6}  {'N high':>7}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*11}  {'-'*8}  {'-'*8}  {'-'*6}  {'-'*7}")

    for label, sub in subsets:
        _low = _high = None
        for flag in ["low", "high"]:
            s = sub[sub["_bin"] == flag]
            if len(s) == 0:
                continue
            tp_ = int(s["tp"].sum())
            fp_ = int(s["fp"].sum())
            fn_ = int(s["fn"].sum())
            p_ = tp_ / (tp_ + fp_) if (tp_ + fp_) > 0 else 0
            r_ = tp_ / (tp_ + fn_) if (tp_ + fn_) > 0 else 0
            f_ = 2 * p_ * r_ / (p_ + r_) if (p_ + r_) > 0 else 0
            if flag == "low":
                _low = (r_, f_, len(s))
            else:
                _high = (r_, f_, len(s))

        try:
            if _low and _high:
                print(f"  {label:<10}  {_low[0]:>10.4f}  {_high[0]:>11.4f}  "
                      f"{_low[1]:>8.4f}  {_high[1]:>8.4f}  "
                      f"{_low[2]:>6}  {_high[2]:>7}")
        except Exception:
            pass

    print(f"")
    print(f"  How each value is computed:")
    print(f"  Images are split into two halves at the MEDIAN value of the metric.")
    print(f"  Low  = images where the metric is BELOW the median.")
    print(f"  High = images where the metric is AT OR ABOVE the median.")
    print(f"  Recall  = sum(TP in group) / sum(TP + FN in group)  — fraction of pedestrians found")
    print(f"  F1      = 2×P×R / (P+R) where P = sum(TP) / sum(TP+FP) in that group")
    print(f"  N low / N high = number of IMAGES in each half (not objects)")


# =============================================================================
# ROOT CAUSE ANALYSIS
#   Uses logistic regression to identify which features best predict
#   whether an object is detected or missed.
#
#   Model: P(detected | features) = sigmoid(w^T * x + b)
#   Features are standardised (mean=0, std=1) so coefficients are
#   directly comparable as feature importance.
#
#   Positive coefficient → higher value of this feature → more likely detected
#   Negative coefficient → higher value → more likely missed
# =============================================================================

def _print_root_cause(df_obj, tag, sep):
    """Logistic regression feature importance for detection prediction."""
    feature_cols = [c for c in [
        "object_brightness", "object_contrast", "fg_bg_diff",
        "object_edge_strength", "object_blur",
        "object_ssim", "ghost_score", "hallucination_score",
        "object_lpips", "bbox_area_px",
    ] if c in df_obj.columns and df_obj[c].notna().any()]

    if len(feature_cols) < 3 or "detected" not in df_obj.columns:
        return

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return

    print(f"\n{sep}")
    print(f"9. ROOT CAUSE ANALYSIS (Feature Importance) — {tag}")
    print(sep)
    print("  Logistic regression: P(detected=1 | object features)")
    print("  Features are standardised (mean=0, std=1) so coefficients are directly comparable.")
    print("  Positive coeff → higher value of this feature → object MORE likely to be detected.")
    print("  Negative coeff → higher value → object LESS likely to be detected.")
    print("  NOTE: coefficients are CONDITIONAL effects (all other features held constant).")
    print("        When features are correlated, individual signs may differ from bivariate")
    print("        correlations in Section 5 — this is a known suppressor-variable effect.")

    df_clean = df_obj.dropna(subset=feature_cols + ["detected"])
    if len(df_clean) < 50:
        print("  (too few complete rows for analysis)")
        return

    X = df_clean[feature_cols].values
    y = df_clean["detected"].astype(int).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_scaled, y)

    # Sort by absolute coefficient (most important first)
    coefs = model.coef_[0]
    order = np.argsort(-np.abs(coefs))

    print(f"\n  {'Feature':<25}  {'Coefficient':>12}  {'Direction':>10}")
    print(f"  {'-'*25}  {'-'*12}  {'-'*10}")
    for idx in order:
        c = coefs[idx]
        direction = "helps ↑" if c > 0 else "hurts ↓"
        print(f"  {feature_cols[idx]:<25}  {c:>+12.4f}  {direction:>10}")

    acc = model.score(X_scaled, y)
    print(f"\n  Model accuracy: {acc:.4f} (on training data)")
    print(f"  N samples: {len(df_clean)}")
    print(f"")
    print(f"  How each value is computed:")
    print(f"  Unit of analysis = one row per GT pedestrian bounding box.")
    print(f"  Features are standardised: each is centred at 0 and scaled to std=1")
    print(f"  so that coefficients from different scales are directly comparable.")
    print(f"  Coefficient = weight in the logistic regression equation:")
    print(f"    P(detected=1) = sigmoid( w1×feat1 + w2×feat2 + ... + bias )")
    print(f"  Larger absolute coefficient = stronger predictor of detection outcome.")
    print(f"  Model accuracy = fraction of objects correctly classified as detected/missed")
    print(f"  (measured on the same training data, so it is an optimistic upper bound).")
