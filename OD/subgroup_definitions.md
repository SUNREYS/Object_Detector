# Evaluation Subgroup Definitions

Reference for the thresholds and categories used in evaluation and plotting.
All definitions come from `preprocessing/config.py`.

---

## Object Size Categories

Based on **bounding box pixel height** after cropping to 512x512.

| Category | Height Range | Description |
|----------|-------------|-------------|
| Small | 0 -- 32 px | Far-away pedestrians, often hard to detect |
| Medium | 33 -- 96 px | Mid-range pedestrians |
| Large | > 96 px | Close-up pedestrians |

Defined by:
```
SIZE_SMALL_MAX = 32   (config.py:118)
SIZE_MEDIUM_MAX = 96  (config.py:119)
```

---

## Occlusion Levels

From the original KAIST XML annotations (attribute on each bounding box).

| Level | Value | Description |
|-------|-------|-------------|
| None | 0 | Fully visible, no occlusion |
| Partial | 1 | Partially hidden behind another object |
| Heavy | 2 | Mostly hidden, only a small part visible |

Defined by:
```
OCCLUSION_NONE = 0       (config.py:123)
OCCLUSION_PARTIAL = 1    (config.py:124)
OCCLUSION_HEAVY = 2      (config.py:125)
```

Note: annotations with less than 25% of their original area visible
after cropping are **discarded entirely** during preprocessing
(`MIN_VISIBLE_FRACTION = 0.25`).

---

## Day / Night Classification

Based on the KAIST recording set ID (each set was filmed at a specific time).

| Condition | Sets |
|-----------|------|
| Day | set00, set01, set02, set06, set07, set08 |
| Night | set03, set04, set05, set09, set10, set11 |

Defined at `config.py:109-110`.

---

## Annotation Overlap

Two bounding boxes are considered "overlapping" when their
Intersection-over-Union (IoU) exceeds the threshold.

```
OVERLAP_IOU_THRESHOLD = 0.3    (config.py:128)
```

---

## Truncation

A bounding box is marked as **truncated** if part of the person extends
beyond the image boundary (i.e., the box was clipped during cropping).
This is a binary flag: 0 = not truncated, 1 = truncated.

---

## Filtering During Preprocessing

Before any image enters the dataset, annotations are filtered:

| Filter | Threshold | Effect |
|--------|-----------|--------|
| Minimum bbox area | 100 px^2 | Removes tiny noise boxes (< ~10x10 px) |
| Minimum visible fraction | 25% | Removes boxes mostly outside the crop region |

---

## Evaluation Matching

During `evaluate.py`, predictions are matched to ground-truth boxes using:

- **IoU threshold**: 0.5 (a prediction must overlap >= 50% with a GT box to count as TP)
- **Confidence threshold**: 0.25 (default, adjustable via `--conf`)
- Each GT box can be matched to at most one prediction (greedy, highest IoU first)

---

## Per-Object Image Quality Features

Computed during evaluation from the actual image pixels (works for all modalities).
These features help answer: **does the visual quality of a person's appearance
in the image affect whether the detector finds them?**

### Feature Table

| Feature | What it measures | Scale |
|---------|-----------------|-------|
| `object_brightness` | Average brightness of the person region | 0 = black, 255 = white |
| `object_contrast` | How much brightness varies within the person region | Higher = more texture/detail |
| `bg_brightness` | Average brightness of the area around the person | Same scale as above |
| `bg_contrast` | How much brightness varies in the background | Higher = busier background |
| `fg_bg_diff` | How much the person stands out from the background | Higher = person "pops" more |
| `object_edge_strength` | How crisp/defined the person's edges are | Higher = sharper edges |
| `object_blur` | Overall sharpness of the person region | Higher = sharper, lower = blurrier |

### Plain-Language Explanations

**fg_bg_diff (Foreground-Background Difference)**:
Take the average brightness inside the bounding box (the person) and subtract
the average brightness of the 15px border around it (the background). The
absolute value is the FG-BG difference. If a person wearing dark clothes
stands against a bright wall, this value is high. If the person and
background have similar brightness, this value is low and the person
"blends in."

#### Interpreting FG-BG Brightness Difference Values

The scale is 0-255 (greyscale pixel intensity difference).

| Range | Interpretation | Example |
|-------|---------------|---------|
| 0-5 | Very low contrast -- person blends into background | Person in grey clothes against grey pavement |
| 5-15 | Low contrast -- slight difference | Typical daytime scene with moderate lighting |
| 15-30 | Moderate contrast | Person in dark clothing against lighter wall |
| 30-60 | High contrast -- person clearly stands out | Dark silhouette against bright sky |
| 60+ | Very high contrast | Thermal hotspot against cold background |

A value of ~12 means the person's average brightness differs from the local
background by only ~5% of the full 0-255 range. This is low contrast.

**object_edge_strength (Sobel Magnitude)**:
The Sobel operator slides a small 3x3 filter across the person region to
detect where brightness changes sharply (i.e., edges). The mean of all
these edge responses gives the "edge strength." A crisp photo of a person
with clear clothing boundaries has high edge strength. A blurred or
low-contrast silhouette has low edge strength.

**object_blur (Laplacian Variance)**:
The Laplacian operator detects rapid brightness changes in all directions.
Taking the variance of this response across the person region gives a
single sharpness number. A sharp, in-focus image has many strong
responses => high variance. A blurry, out-of-focus image has weak,
uniform responses => low variance. This is the standard "blur detector"
used in computer vision.

### Background Region

The background is sampled from a 15px-wide border around the bounding box
(clipped to image boundaries), **excluding** the object pixels themselves.
This captures the local surroundings of the person, not distant scenery.

Objects smaller than 3x3 pixels get NaN for all quality features (too small
for meaningful edge/blur computation).

### How to Read the Plots

**Plot 13 (Contrast Analysis)**:
- Left panel: Each dot is a group of ~1800 objects binned by their
  FG-BG contrast. X-axis = contrast value, Y-axis = what fraction of
  those objects were detected. The dashed red line shows the trend.
  A positive slope means higher contrast helps detection.
- Right panel: Average contrast for detected vs missed objects.
  If the bars are similar, contrast doesn't affect detection.

**Plot 14 (Sharpness Analysis)**:
- Same layout as Plot 13, but for sharpness (Laplacian variance).
  A positive slope means sharper objects are easier to detect.

**Plot 15 (Quality Dashboard)**:
- Four scatter plots, one per feature. Each dot = a group of objects
  with similar feature values. X-axis = feature value, Y-axis = detection rate.
  The red dashed trend line shows correlation.
  A flat line (slope ~0) means the feature doesn't affect detection.
  A rising line means higher values help detection.
