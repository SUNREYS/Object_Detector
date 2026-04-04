"""
=============================================================================
MAIN PIPELINE: KAIST Dataset Preprocessing for Object Detection
=============================================================================

This is the main entry point that runs the entire preprocessing pipeline
in sequence. Each step depends on the output of the previous step.

Pipeline overview:
    Step 1: Parse XML annotations, crop images to 512x512, convert
            bounding boxes to YOLO format, organize into train/val/test.
            Day/night labels come from the KAIST set IDs directly.
    Step 2: Extract per-image and per-object features for analysis
            (brightness, contrast, size, occlusion, crowd density, etc.)
    Step 3: Export all metadata to structured CSV files.
    Step 4: Generate analysis plots and visualizations.

Usage:
    python main.py              # Run full pipeline
    python main.py --step 1     # Run only Step 1
    python main.py --step 2-4   # Run Steps 2-4 (requires Step 1 output)
    python main.py --plots-only # Only regenerate plots from existing CSVs

Output structure:
    datasets/kaist_processed/
    ├── images/
    │   ├── visible/
    │   │   ├── train/
    │   │   ├── val/
    │   │   └── test/
    │   └── thermal/
    │       ├── train/
    │       ├── val/
    │       └── test/
    ├── labels/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── metadata/
    │   ├── frame_metadata.csv
    │   ├── object_metadata.csv
    │   ├── dataset_summary.csv
    │   ├── size_distribution.csv
    │   └── crowd_analysis.csv
    ├── plots/
    │   ├── 01_dataset_composition.png
    │   ├── 02_size_distribution.png
    │   ├── ... (12 analysis plots)
    │   ├── 13_training_loss_curves.png  (post-training)
    │   └── 14_training_metrics.png      (post-training)
    ├── kaist_visible.yaml
    └── kaist_thermal.yaml
=============================================================================
"""

import argparse
import os
import sys
import time
import pickle

from config import OUTPUT_ROOT, OUTPUT_METADATA


def save_intermediate(data, name: str):
    """Save intermediate results for resuming pipeline."""
    path = os.path.join(OUTPUT_METADATA, f"_intermediate_{name}.pkl")
    os.makedirs(OUTPUT_METADATA, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"  [Checkpoint] Saved {name} to: {path}")


def load_intermediate(name: str):
    """Load intermediate results to resume pipeline."""
    path = os.path.join(OUTPUT_METADATA, f"_intermediate_{name}.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        data = pickle.load(f)
    print(f"  [Checkpoint] Loaded {name} from: {path}")
    return data


def run_full_pipeline(start_step: int = 1, end_step: int = 4,
                      results_csv: str = None, modality: str = "visible",
                      newsplit: bool = False):
    """
    Run the preprocessing pipeline from start_step to end_step.

    Args:
        start_step: First step to run (1-4).
        end_step: Last step to run (1-4).
        results_csv: Optional path to YOLO results.csv for post-training plots.
    """
    total_start = time.time()

    print("=" * 70)
    print("KAIST DATASET PREPROCESSING PIPELINE")
    print(f"Running Steps {start_step} through {end_step}")
    print(f"Output directory: {OUTPUT_ROOT}")
    print("=" * 70)

    frame_results = None
    all_features = None

    # ---- Step 1: Convert annotations and process images ----
    if start_step <= 1 <= end_step:
        from step1_convert import run_step1
        frame_results = run_step1()
        save_intermediate(frame_results, "step1_results")
    elif start_step > 1:
        frame_results = load_intermediate("step1_results")
        if frame_results is None:
            print("ERROR: Step 1 results not found. Run Step 1 first.")
            sys.exit(1)

    # ---- Step 2: Extract features ----
    if start_step <= 2 <= end_step:
        from step2_features import run_step2
        all_features = run_step2(frame_results, modality=modality)
        save_intermediate(all_features, "step2_features")
    elif start_step > 2:
        all_features = load_intermediate("step2_features")
        if all_features is None:
            print("ERROR: Step 2 results not found. Run Step 2 first.")
            sys.exit(1)

    # ---- Step 3: Export metadata to CSV ----
    if start_step <= 3 <= end_step:
        from step4_export import run_step4
        csv_paths = run_step4(all_features, modalities=[modality])

    # ---- Step 4: Generate plots ----
    if start_step <= 4 <= end_step:
        from step5_plots import run_step5
        run_step5(results_csv)

    elapsed = time.time() - total_start
    print("\n" + "=" * 70)
    print(f"PIPELINE COMPLETE — Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Output directory: {OUTPUT_ROOT}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="KAIST Dataset Preprocessing Pipeline for Object Detection"
    )
    parser.add_argument(
        "--step", type=str, default="1-4",
        help="Step(s) to run. Examples: '1', '1-4', '2-4', '4'. Default: 1-4"
    )
    parser.add_argument(
        "--modality", type=str, default="visible",
        choices=["visible", "thermal", "greyscale_inversion", "PI-GAN_gen"],
        help="Which modality to process. Default: visible"
    )
    parser.add_argument(
        "--newsplit", action="store_true",
        help="Use newsplit mode (train/val split from --newsplit.py)"
    )
    parser.add_argument(
        "--plots-only", action="store_true",
        help="Only regenerate plots from existing CSV files (runs Step 4 only)"
    )
    parser.add_argument(
        "--results-csv", type=str, default=None,
        help="Path to YOLO training results.csv for post-training plots"
    )

    args = parser.parse_args()

    if args.plots_only:
        from step5_plots import run_step5
        run_step5(args.results_csv)
        return

    # Parse step range
    if "-" in args.step:
        parts = args.step.split("-")
        start_step = int(parts[0])
        end_step = int(parts[1])
    else:
        start_step = int(args.step)
        end_step = start_step

    run_full_pipeline(start_step, end_step, args.results_csv, 
                      args.modality, args.newsplit)


if __name__ == "__main__":
    main()
