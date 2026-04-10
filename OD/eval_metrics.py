"""
=============================================================================
eval_metrics.py — Image Quality Metrics and Perceptual Similarity
=============================================================================

Computes image quality features that measure how faithful the synthetic
thermal-to-visible translation is, and how that quality affects detection.

ALL images are loaded and processed as 3-channel BGR (as stored on disk).
The YOLO detector sees RGB — we must evaluate quality on the SAME data
the detector receives. Greyscale conversion is only done internally
inside metrics that mathematically require a single channel (Laplacian
sharpness, Canny edge detection).

──────────────────────────────────────────────────────────────────────
METRIC OVERVIEW
──────────────────────────────────────────────────────────────────────

PER-OBJECT QUALITY (computed on the bounding-box crop):
  brightness        = mean(crop) / 255                         [0, 1]
  contrast          = std(crop) / 255                          [0, 1]
  fg_bg_diff        = |mean(foreground) - mean(background)|    [0, 255]
  edge_strength     = mean(Sobel_magnitude(luminance_crop))    [0, ~360]
  blur (sharpness)  = Var(Laplacian(luminance_crop))           [0, ∞)

PER-OBJECT GHOST / HALLUCINATION (generated crop vs thermal GT crop):
  object_ssim           = SSIM(gen, thermal) per-channel avg   [-1, 1]
  ghost_score           = mean(GaussBlur(|gen - thermal|))     [0, 1]
  hallucination_score   = |gen_edges - thermal_edges| / |gen_edges|  [0, 1]

PER-OBJECT PERCEPTUAL (requires lpips package):
  object_lpips          = LPIPS(gen_crop, thermal_crop)        [0, ~1]

PER-DISTRIBUTION (one number per modality):
  FID                   = Fréchet Inception Distance            [0, ∞)

Functions:
    load_image(path)           → 3-channel BGR uint8
    load_thermal_gt(name, split, thermal_dir) → 3-channel BGR or None
    compute_object_quality_features(img_bgr, x1, y1, x2, y2, ...)
    compute_ghost_features(gen_bgr, thermal_bgr)
    compute_image_quality(img_bgr)  → dict of full-image metrics

Classes:
    LPIPSScorer    — wraps lpips library (loaded once, reused)
    FIDComputer    — InceptionV3 feature extraction + Fréchet distance
=============================================================================
"""

import os
import numpy as np
import cv2
from typing import Dict, Optional, Tuple

# Background border size for fg/bg contrast computation
BG_BORDER_PX = 15


# =============================================================================
# IMAGE LOADING
#   All images are loaded as 3-channel BGR uint8 — no greyscale conversion.
#   The YOLO detector sees RGB images, so quality metrics must reflect
#   what the detector actually receives.
# =============================================================================

def load_image(img_path: str) -> Optional[np.ndarray]:
    """
    Load an image as 3-channel BGR uint8 (as-is from disk).
    Returns None if the file cannot be read.
    """
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    return img


def load_thermal_gt(flat_name: str, split: str, thermal_dir: str) -> Optional[np.ndarray]:
    """
    Load the corresponding thermal ground truth image (3-channel BGR).
    Tries the given split first; returns None if not found.
    """
    path = os.path.join(thermal_dir, split, f"{flat_name}.jpg")
    if not os.path.exists(path):
        return None
    return load_image(path)


def _to_gray(img_bgr: np.ndarray) -> np.ndarray:
    """
    Internal helper: convert BGR → single-channel greyscale.
    Only used inside metrics that require single-channel input
    (Laplacian, Sobel, Canny).
    """
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


# =============================================================================
# PER-IMAGE QUALITY METRICS
#   Computed on the full image (not just object crops).
#   These give an overall picture of image quality.
#
#   brightness    = mean pixel value across all channels / 255     [0, 1]
#   contrast      = std of pixel values across all channels / 255  [0, 1]
#   sharpness     = Var(Laplacian(luminance))  — higher = sharper
#   blur          = 1 / (sharpness + ε)        — higher = blurrier
#   edge_density  = mean(Sobel magnitude on luminance)
# =============================================================================

def compute_image_quality(img_bgr: np.ndarray) -> Dict[str, float]:
    """
    Compute full-image quality metrics from a 3-channel BGR image.

    Returns dict with: brightness, contrast, sharpness, blur, edge_density.
    """
    # ── Brightness ──────────────────────────────────────────────
    # Mean pixel intensity across all 3 channels.
    # brightness = (1 / (H * W * 3)) * sum(all_pixels)
    # Divided by 255 to scale to [0, 1].
    brightness = float(np.mean(img_bgr))

    # ── Contrast ────────────────────────────────────────────────
    # Standard deviation of pixel intensities across all channels.
    # High contrast = wide spread of pixel values.
    contrast = float(np.std(img_bgr))

    # ── Sharpness (via Laplacian variance) ──────────────────────
    # The Laplacian operator detects edges / rapid intensity changes:
    #   L(x,y) = ∂²I/∂x² + ∂²I/∂y²  (sum of second derivatives)
    # Sharpness = Var(L) over the image.
    # High variance → many sharp edges → sharp image.
    # Low variance → smooth gradients → blurry image.
    gray = _to_gray(img_bgr)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = float(np.var(lap))

    # ── Blur ────────────────────────────────────────────────────
    # Inverse of sharpness: blur = 1 / (sharpness + ε)
    # ε = 1e-6 prevents division by zero.
    blur = float(1.0 / (sharpness + 1e-6))

    # ── Edge Density ────────────────────────────────────────────
    # Mean Sobel gradient magnitude on the luminance channel.
    # Sobel computes first derivatives:
    #   Gx = ∂I/∂x,  Gy = ∂I/∂y
    #   magnitude = sqrt(Gx² + Gy²)
    # edge_density = mean(magnitude) — how much edge content overall.
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_density = float(np.mean(np.sqrt(sx**2 + sy**2)))

    return {
        "brightness": brightness,
        "contrast": contrast,
        "sharpness": sharpness,
        "blur": blur,
        "edge_density": edge_density,
    }


# =============================================================================
# PER-OBJECT QUALITY FEATURES
#   For each GT bounding box, measure local image quality.
#   These tell us about the specific object region the detector must recognise.
#
#   object_brightness   = mean(crop_pixels) across all 3 channels
#   object_contrast     = std(crop_pixels) across all 3 channels
#   bg_brightness       = mean(background_border_pixels)
#   bg_contrast         = std(background_border_pixels)
#   fg_bg_diff          = |object_brightness - bg_brightness|
#                         Low fg_bg_diff → object blends into background
#                         (hard to detect because there's no visual separation)
#   object_edge_strength = mean Sobel magnitude on luminance crop
#   object_blur          = Laplacian variance on luminance crop (= sharpness)
# =============================================================================

def compute_object_quality_features(
    img_bgr: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    img_h: int, img_w: int,
    border_px: int = BG_BORDER_PX,
) -> Dict[str, float]:
    """
    Compute image-quality features for a single object bounding box.

    Args:
        img_bgr:   Full image as 3-channel BGR uint8.
        x1, y1, x2, y2: Bounding box in pixel coordinates.
        img_h, img_w:    Image dimensions.
        border_px:       Background ring width around the bbox (default 15px).

    Returns dict with 7 quality features (see module docstring).
    """
    nan_result = {
        "object_brightness": float("nan"),
        "object_contrast": float("nan"),
        "bg_brightness": float("nan"),
        "bg_contrast": float("nan"),
        "fg_bg_diff": float("nan"),
        "object_edge_strength": float("nan"),
        "object_blur": float("nan"),
    }

    # Bbox must be at least 3×3 for Sobel/Laplacian to work
    if (x2 - x1) < 3 or (y2 - y1) < 3:
        return nan_result

    crop = img_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return nan_result

    # ── Brightness & Contrast (on full RGB crop) ────────────────
    # Mean and std across ALL pixel values in the crop (H × W × 3).
    obj_brightness = float(np.mean(crop))
    obj_contrast = float(np.std(crop))

    # ── Background ring ─────────────────────────────────────────
    # Expand bbox by border_px in all directions, then mask out the
    # object region to get background-only pixels.
    bx1 = max(0, x1 - border_px)
    by1 = max(0, y1 - border_px)
    bx2 = min(img_w, x2 + border_px)
    by2 = min(img_h, y2 + border_px)

    bg_region = img_bgr[by1:by2, bx1:bx2].copy()
    mask = np.ones(bg_region.shape[:2], dtype=bool)
    mask[y1 - by1:y2 - by1, x1 - bx1:x2 - bx1] = False

    bg_pixels = bg_region[mask]
    if bg_pixels.size > 0:
        bg_brightness = float(np.mean(bg_pixels))
        bg_contrast = float(np.std(bg_pixels))
    else:
        bg_brightness = float("nan")
        bg_contrast = float("nan")

    # ── Foreground-Background Difference ────────────────────────
    # fg_bg_diff = |mean(object) - mean(background)|
    # Low value → object blends into background → hard to detect.
    fg_bg_diff = (abs(obj_brightness - bg_brightness)
                  if not np.isnan(bg_brightness) else float("nan"))

    # ── Edge Strength & Blur (on luminance channel) ─────────────
    # These require single-channel input → convert internally.
    crop_gray = _to_gray(crop)

    # Edge strength = mean(Sobel_magnitude)
    sobel_x = cv2.Sobel(crop_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(crop_gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_strength = float(np.mean(np.sqrt(sobel_x**2 + sobel_y**2)))

    # Blur = Laplacian variance (higher = sharper, confusingly named)
    laplacian = cv2.Laplacian(crop_gray, cv2.CV_64F)
    blur_var = float(np.var(laplacian))

    return {
        "object_brightness": round(obj_brightness, 4),
        "object_contrast": round(obj_contrast, 4),
        "bg_brightness": round(bg_brightness, 4),
        "bg_contrast": round(bg_contrast, 4),
        "fg_bg_diff": round(fg_bg_diff, 4),
        "object_edge_strength": round(edge_strength, 4),
        "object_blur": round(blur_var, 4),
    }


# =============================================================================
# GHOST PATTERN & HALLUCINATION DETECTION
#   Compares each generated object crop against the thermal ground truth
#   to find artifacts the translation model introduced.
#
#   Three complementary metrics:
#
#   1. object_ssim (Structural Similarity Index)
#      SSIM(x, y) = [f(luminance) * f(contrast) * f(structure)]
#      Computed per-channel and averaged.  Range: [-1, 1], 1 = identical.
#      Measures overall structural preservation.
#
#   2. ghost_score (Structured Residual)
#      R(x,y) = |gen(x,y) - thermal(x,y)|  per-pixel absolute difference
#      S = GaussianBlur(R, ksize)            suppress random noise
#      ghost_score = mean(S) / 255           scale to [0, 1]
#      High → large structured differences → visible ghost artifacts.
#
#   3. hallucination_score (Edge Hallucination)
#      E_gen = Canny(gen_crop),  E_thm = Canny(thermal_crop)
#      Hallucinated edges = gen edges not covered by (dilated) thermal edges
#      hallucination_score = |hallucinated| / |E_gen|
#      Range [0, 1].  High → generated image has edges that don't exist
#      in reality → the model invented visual features.
# =============================================================================

def compute_ghost_features(
    gen_crop_bgr: np.ndarray,
    thermal_crop_bgr: np.ndarray,
) -> Dict[str, float]:
    """
    Compare a generated image crop against the thermal ground truth
    to detect ghost patterns and hallucinated features.

    Both inputs should be 3-channel BGR crops of the same object region.

    Returns dict with: object_ssim, ghost_score, hallucination_score.
    """
    from skimage.metrics import structural_similarity as ssim

    nan_result = {
        "object_ssim": float("nan"),
        "ghost_score": float("nan"),
        "hallucination_score": float("nan"),
    }

    if gen_crop_bgr.size == 0 or thermal_crop_bgr.size == 0:
        return nan_result
    if gen_crop_bgr.shape[0] < 7 or gen_crop_bgr.shape[1] < 7:
        return nan_result

    # Resize thermal crop to match generated crop dimensions
    if gen_crop_bgr.shape[:2] != thermal_crop_bgr.shape[:2]:
        thermal_crop_bgr = cv2.resize(
            thermal_crop_bgr,
            (gen_crop_bgr.shape[1], gen_crop_bgr.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    # ── 1. SSIM (per-channel average) ──────────────────────────
    # Compute SSIM on each BGR channel separately, then average.
    # This preserves colour-specific quality differences.
    win_size = min(7, gen_crop_bgr.shape[0], gen_crop_bgr.shape[1])
    if win_size % 2 == 0:
        win_size -= 1
    if win_size < 3:
        return nan_result

    ssim_values = []
    for ch in range(3):
        s_val = ssim(
            thermal_crop_bgr[:, :, ch],
            gen_crop_bgr[:, :, ch],
            win_size=win_size,
            data_range=255,
        )
        ssim_values.append(s_val)
    ssim_avg = float(np.mean(ssim_values))

    # ── 2. Ghost Score (structured residual) ────────────────────
    # Convert to float [0, 1], take absolute difference across all channels,
    # average channels, Gaussian blur to emphasise structure over noise.
    gen_f = gen_crop_bgr.astype(np.float32) / 255.0
    thm_f = thermal_crop_bgr.astype(np.float32) / 255.0
    residual = np.mean(np.abs(gen_f - thm_f), axis=2)  # average across channels
    ksize = max(3, min(gen_crop_bgr.shape[0], gen_crop_bgr.shape[1]) // 4)
    if ksize % 2 == 0:
        ksize += 1
    structured = cv2.GaussianBlur(residual, (ksize, ksize), 0)
    ghost_score = float(np.mean(structured))

    # ── 3. Hallucination Score (edge comparison) ────────────────
    # Canny on luminance channel of both crops.
    gen_gray = _to_gray(gen_crop_bgr)
    thm_gray = _to_gray(thermal_crop_bgr)

    med_gen = max(1, int(np.median(gen_gray)))
    med_thm = max(1, int(np.median(thm_gray)))
    edges_gen = cv2.Canny(gen_gray, int(0.66 * med_gen), int(1.33 * med_gen + 1))
    edges_thm = cv2.Canny(thm_gray, int(0.66 * med_thm), int(1.33 * med_thm + 1))

    # Dilate thermal edges to allow small alignment tolerance
    kernel = np.ones((3, 3), np.uint8)
    edges_thm_dil = cv2.dilate(edges_thm, kernel, iterations=1)

    gen_edge_count = int(np.sum(edges_gen > 0))
    if gen_edge_count > 0:
        hallucinated = np.sum((edges_gen > 0) & (edges_thm_dil == 0))
        hallucination_score = float(hallucinated) / gen_edge_count
    else:
        hallucination_score = 0.0

    return {
        "object_ssim": round(ssim_avg, 4),
        "ghost_score": round(ghost_score, 4),
        "hallucination_score": round(hallucination_score, 4),
    }


# =============================================================================
# LPIPS — LEARNED PERCEPTUAL IMAGE PATCH SIMILARITY
#
#   Measures perceptual distance between two image crops using deep features
#   from a pretrained neural network (AlexNet).
#
#   Math:
#     d(x, y) = sum_l  w_l * || norm(f_l(x)) - norm(f_l(y)) ||²
#   where:
#     f_l = feature activations at layer l of pretrained AlexNet
#     norm = unit-length normalisation in the channel dimension
#     w_l  = learned weights calibrated against human perceptual judgments
#
#   Range: 0 = perceptually identical, ~0.5+ = very different
#
#   Why for this thesis:
#     SSIM measures structural similarity (pixel-level statistics).
#     LPIPS measures whether the image LOOKS right to a human.
#     A generated pedestrian might have correct brightness/edges (good SSIM)
#     but unrealistic textures or colours (bad LPIPS). If LPIPS correlates
#     with detection failure, it means the detector is sensitive to
#     perceptual quality — the pedestrian needs to "look real" to be detected.
#
#   Only computed for synthetic modalities (PID, PI-GAN) where thermal
#   ground truth exists. Skipped for visible (real images).
# =============================================================================

class LPIPSScorer:
    """
    Wrapper around the lpips library. Loads the network once and reuses
    it for all crops during evaluation.
    """

    def __init__(self, net: str = "alex", device: str = "cuda"):
        """
        Args:
            net:    Backbone network ("alex" or "vgg"). AlexNet is faster.
            device: "cuda" or "cpu".
        """
        try:
            import lpips as _lpips
            import torch
        except ImportError:
            print("WARNING: lpips package not installed. "
                  "Run: pip install lpips")
            self._model = None
            return

        self._device = device if torch.cuda.is_available() else "cpu"
        self._model = _lpips.LPIPS(net=net).to(self._device)
        self._model.eval()
        self._torch = torch

    def score(self, gen_crop_bgr: np.ndarray,
              thermal_crop_bgr: np.ndarray) -> float:
        """
        Compute LPIPS distance between a generated crop and thermal GT crop.

        Both inputs: BGR uint8 numpy arrays.
        Returns float (0 = identical, higher = more different).
        Returns NaN if LPIPS is unavailable or crops are too small.
        """
        if self._model is None:
            return float("nan")

        if gen_crop_bgr.size == 0 or thermal_crop_bgr.size == 0:
            return float("nan")
        if gen_crop_bgr.shape[0] < 16 or gen_crop_bgr.shape[1] < 16:
            return float("nan")

        # ── Preprocessing ──────────────────────────────────────
        # 1. BGR → RGB
        # 2. Resize to 64×64 (normalises scale, avoids OOM on tiny crops)
        # 3. Convert to float tensor, scale to [-1, 1]
        # 4. Shape: (1, 3, 64, 64)
        def _prep(crop_bgr):
            rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (64, 64), interpolation=cv2.INTER_LINEAR)
            t = self._torch.from_numpy(resized).permute(2, 0, 1).float()
            t = t / 127.5 - 1.0  # scale [0, 255] → [-1, 1]
            return t.unsqueeze(0).to(self._device)

        t_gen = _prep(gen_crop_bgr)
        t_thm = _prep(thermal_crop_bgr)

        with self._torch.no_grad():
            d = self._model(t_gen, t_thm)

        return round(float(d.item()), 4)


# =============================================================================
# FID — FRÉCHET INCEPTION DISTANCE
#
#   Measures the distance between two IMAGE DISTRIBUTIONS, not individual
#   images. One FID number per modality = "how realistic is this generator?"
#
#   Math:
#     FID = ||mu_gen - mu_real||² + Tr(S_gen + S_real - 2 * sqrtm(S_gen @ S_real))
#   where:
#     mu, S = mean vector and covariance matrix of 2048-dim InceptionV3
#             pool3 features extracted from each image set
#     sqrtm = matrix square root (scipy.linalg.sqrtm)
#
#   Lower FID = generated distribution closer to reference. Typical range:
#     FID ≈ 0–50   → very close distributions
#     FID ≈ 50–150 → moderate difference
#     FID > 200    → quite different
#
#   Why for this thesis:
#     Per-image metrics (SSIM, LPIPS) tell us about individual images.
#     FID tells us about the MODEL — does it systematically produce
#     realistic images? Comparing FID vs mAP across modalities answers:
#     "Does a more realistic generator produce a better detector?"
#
#   Uses torchvision's InceptionV3 (already installed). No extra dependency.
# =============================================================================

class FIDComputer:
    """
    Compute Fréchet Inception Distance between two sets of images.

    Usage:
        fid = FIDComputer()
        score = fid.compute(gen_image_dir, real_image_dir)
        # score is a float (lower = better)
    """

    def __init__(self, device: str = "cuda", batch_size: int = 32):
        import torch
        import torchvision.models as models
        import torchvision.transforms as T

        self._device = device if torch.cuda.is_available() else "cpu"
        self._batch_size = batch_size
        self._torch = torch

        # ── Load InceptionV3 ───────────────────────────────────
        # We remove the final classification layer and use the
        # 2048-dim average-pooling output as image features.
        inception = models.inception_v3(pretrained=True, transform_input=False)
        inception.fc = torch.nn.Identity()  # remove classifier
        inception.eval()
        self._model = inception.to(self._device)

        # ── Preprocessing ──────────────────────────────────────
        # InceptionV3 expects 299×299 RGB images normalised with
        # ImageNet mean/std.
        self._transform = T.Compose([
            T.ToPILImage(),
            T.Resize((299, 299)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def extract_features(self, image_dir: str,
                         filenames: list = None) -> np.ndarray:
        """
        Extract 2048-dim InceptionV3 features for all images in a directory.

        Args:
            image_dir: Path to directory containing .jpg images.
            filenames: Optional list of specific filenames to process.
                       If None, processes all .jpg/.png files.

        Returns:
            np.ndarray of shape (N, 2048).
        """
        if filenames is None:
            filenames = sorted([
                f for f in os.listdir(image_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ])

        features = []
        batch = []

        for fname in filenames:
            img_bgr = cv2.imread(os.path.join(image_dir, fname))
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            tensor = self._transform(img_rgb)
            batch.append(tensor)

            if len(batch) >= self._batch_size:
                features.append(self._run_batch(batch))
                batch = []

        if batch:
            features.append(self._run_batch(batch))

        if not features:
            return np.empty((0, 2048))
        return np.concatenate(features, axis=0)

    def _run_batch(self, batch: list) -> np.ndarray:
        """Forward a batch of tensors through InceptionV3."""
        x = self._torch.stack(batch).to(self._device)
        with self._torch.no_grad():
            feats = self._model(x)
        return feats.cpu().numpy()

    def compute(self, gen_dir: str, real_dir: str,
                gen_filenames: list = None,
                real_filenames: list = None) -> float:
        """
        Compute FID between generated and real image directories.

        FID = ||mu1 - mu2||² + Tr(S1 + S2 - 2 * sqrtm(S1 @ S2))

        Returns float (lower = better).
        """
        from scipy import linalg

        feats_gen = self.extract_features(gen_dir, gen_filenames)
        feats_real = self.extract_features(real_dir, real_filenames)

        if len(feats_gen) < 2 or len(feats_real) < 2:
            print("WARNING: Not enough images for FID computation.")
            return float("nan")

        # ── Compute statistics ─────────────────────────────────
        # mu = mean feature vector (2048-dim)
        # sigma = covariance matrix (2048 × 2048)
        mu1 = np.mean(feats_gen, axis=0)
        mu2 = np.mean(feats_real, axis=0)
        sigma1 = np.cov(feats_gen, rowvar=False)
        sigma2 = np.cov(feats_real, rowvar=False)

        # ── Fréchet distance ───────────────────────────────────
        # d² = ||mu1 - mu2||² + Tr(sigma1 + sigma2 - 2 * sqrtm(sigma1 @ sigma2))
        diff = mu1 - mu2
        covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)

        # sqrtm can return complex values due to numerical issues
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = float(
            diff @ diff
            + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        )
        return round(fid, 2)
