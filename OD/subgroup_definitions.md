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

| Feature | Computation | Interpretation |
|---------|-------------|----------------|
| `object_brightness` | Mean pixel intensity within bbox (greyscale) | 0=black, 255=white |
| `object_contrast` | Std dev of pixels within bbox | Higher = more internal texture variation |
| `bg_brightness` | Mean pixel intensity in 15px border around bbox (excluding object) | Background appearance |
| `bg_contrast` | Std dev in border region | Background complexity |
| `fg_bg_diff` | abs(object_brightness - bg_brightness) | How much the object "pops" from its surroundings |
| `object_edge_strength` | Mean Sobel magnitude within bbox | Sharpness of edges (higher = crisper) |
| `object_blur` | Variance of Laplacian within bbox | Classic blur metric (higher = sharper, lower = blurrier) |

**Background region**: 15px border around the bounding box, excluding the object itself.
Objects smaller than 3x3 pixels get NaN for all quality features.

**Plots using these features**:
- Plot 13: Recall by FG-BG contrast quartile
- Plot 14: Recall by sharpness quartile
- Plot 15: 2x2 dashboard (contrast, blur, edge strength, brightness vs detection rate)
