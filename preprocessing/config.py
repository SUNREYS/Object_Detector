"""
=============================================================================
Configuration Module for KAIST Dataset Preprocessing Pipeline
=============================================================================

Step 0: CONFIGURATION
    Central configuration file that holds all paths, constants, and parameters
    used across the preprocessing pipeline. Edit paths here to match your
    local setup. All other scripts import from this module.

Classes:
    The KAIST dataset labels people as "person", "people", or "cyclist".
    For the object detector, all are mapped to a single class (class 0)
    since we only care about detecting humans/pedestrians.
=============================================================================
"""

import os

# =============================================================================
# Step 0.1: DATASET PATHS
#   Root paths for the raw KAIST dataset and output directories.
# =============================================================================

KAIST_ROOT = r"C:\Users\poopi\Downloads\kaist-cvpr15"
KAIST_IMAGES = os.path.join(KAIST_ROOT, "images")
KAIST_ANNOTATIONS = os.path.join(KAIST_ROOT, "annotations-xml-new-sanitized")
KAIST_IMAGE_SETS = os.path.join(KAIST_ROOT, "imageSets")

# Output root for all processed data
OUTPUT_ROOT = r"C:\Users\poopi\Documents\programming projectzx\exjobb\datasets\kaist_processed"

# Processed output subdirectories
OUTPUT_IMAGES_VISIBLE = os.path.join(OUTPUT_ROOT, "images", "visible")
OUTPUT_IMAGES_THERMAL = os.path.join(OUTPUT_ROOT, "images", "thermal")
OUTPUT_IMAGES_GREYSCALE_INVERSION = os.path.join(OUTPUT_ROOT, "images", "greyscale_inversion")
OUTPUT_IMAGES_PIGAN = os.path.join(OUTPUT_ROOT, "images", "PI-GAN_gen")

# Lookup: modality name -> processed image directory
MODALITY_IMAGE_DIRS = {
    "visible": OUTPUT_IMAGES_VISIBLE,
    "thermal": OUTPUT_IMAGES_THERMAL,
    "greyscale_inversion": OUTPUT_IMAGES_GREYSCALE_INVERSION,
    "PI-GAN_gen": OUTPUT_IMAGES_PIGAN,
}

OUTPUT_LABELS = os.path.join(OUTPUT_ROOT, "labels")
OUTPUT_METADATA = os.path.join(OUTPUT_ROOT, "metadata")
OUTPUT_PLOTS = os.path.join(OUTPUT_ROOT, "plots")

# Train/val split subdirectories (created under images and labels)
# No test split — user has a separate test set (authentic thermal data)
SPLITS = ["train", "val"]

# =============================================================================
# Step 0.2: IMAGE PROCESSING PARAMETERS
#   The KAIST images are 640x512. We crop to 512x512 (center crop) and
#   then the images are already at the target size. If a different target
#   size is needed, set TARGET_SIZE accordingly.
# =============================================================================

ORIGINAL_WIDTH = 640
ORIGINAL_HEIGHT = 512
TARGET_SIZE = 512  # Final square image size (512x512)

# Center crop: remove (640-512)/2 = 64 pixels from left and right
CROP_LEFT = (ORIGINAL_WIDTH - ORIGINAL_HEIGHT) // 2   # 64
CROP_RIGHT = ORIGINAL_WIDTH - CROP_LEFT                # 576
CROP_TOP = 0
CROP_BOTTOM = ORIGINAL_HEIGHT                          # 512

# After cropping, the image is 512x512 already, so no resize needed
# unless TARGET_SIZE differs from ORIGINAL_HEIGHT
CROPPED_WIDTH = CROP_RIGHT - CROP_LEFT    # 512
CROPPED_HEIGHT = CROP_BOTTOM - CROP_TOP   # 512

# =============================================================================
# Step 0.3: CLASS MAPPING
#   All human-related classes in KAIST are mapped to a single YOLO class 0.
#   "person", "people", "cyclist" all represent humans in the scene.
# =============================================================================

# Source labels in the KAIST XML annotations
HUMAN_CLASSES = {"person", "people", "cyclist"}

# YOLO class mapping: all human classes -> class 0
CLASS_MAP = {
    "person": 0,
    "people": 0,
    "cyclist": 0,
}

# Class names list for YOLO .yaml config
CLASS_NAMES = ["person"]
NUM_CLASSES = len(CLASS_NAMES)

# =============================================================================
# Step 0.4: ANNOTATION ATTRIBUTES
#   Thresholds and flags for filtering annotations.
# =============================================================================

# Minimum bounding box area (in pixels, after crop) to keep an annotation.
# Very small boxes are often labeling noise or extremely distant pedestrians
# that are impossible to detect.
MIN_BBOX_AREA_PX = 100  # pixels squared (e.g., 10x10)

# Minimum fraction of the original bounding box that must remain visible
# after cropping. If less than this fraction is visible, discard the annotation.
MIN_VISIBLE_FRACTION = 0.25

# =============================================================================
# Step 0.5: DAY / NIGHT SET MAPPING
#   Based on the KAIST imageSets split files:
#   - Day sets: set00, set01, set02, set06, set07, set08
#   - Night sets: set03, set04, set05, set09, set10, set11
#   All sets are used for training (no KAIST test split needed — user
#   has a separate authentic thermal test set).
# =============================================================================

DAY_SETS = {"set00", "set01", "set02", "set06", "set07", "set08"}
NIGHT_SETS = {"set03", "set04", "set05", "set09", "set10", "set11"}

# =============================================================================
# Step 0.6: FEATURE EXTRACTION PARAMETERS
#   Parameters for the analysis feature extraction step.
# =============================================================================

# Bounding box size categories (based on pixel height after crop/resize)
SIZE_SMALL_MAX = 32     # 0 to 32px height -> "small" (far away)
SIZE_MEDIUM_MAX = 96    # 33 to 96px height -> "medium"
                        # > 96px height -> "large" (close)

# Occlusion levels from KAIST annotations
OCCLUSION_NONE = 0
OCCLUSION_PARTIAL = 1
OCCLUSION_HEAVY = 2

# Overlap IoU threshold for counting overlapping annotations
OVERLAP_IOU_THRESHOLD = 0.3

# =============================================================================
# Step 0.7: SAMPLING PARAMETERS
#   The full KAIST dataset has ~95K frames. We combine both the original
#   train and test lists to use ALL frames for training the OD.
#   Every-other-frame sampling reduces redundancy from consecutive video
#   frames. The -02 suffix means every 2nd frame.
# =============================================================================

# imageSets files to load — we combine both into one training pool
TRAIN_LIST_FILE = "train-all-02.txt"   # set00-05, every 2nd frame
TEST_LIST_FILE = "test-all-01.txt"     # set06-11, every frame (also used for training)

# Fraction of the combined frame pool to actually use (discard the rest)
# Set to 1.0 to use everything, 0.8 to use 80%, etc.
DATASET_FRACTION = 1

# Validation split: fraction of the used data to hold out for validation
VAL_FRACTION = 0.20

# Random seed for reproducibility
RANDOM_SEED = 42
