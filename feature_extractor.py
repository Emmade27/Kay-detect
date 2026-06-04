from __future__ import annotations

from typing import Dict, List, Tuple
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

FEATURE_NAMES: List[str] = [
    "rgb_mean_r", "rgb_mean_g", "rgb_mean_b",
    "rgb_std_r", "rgb_std_g", "rgb_std_b",
    "hsv_mean_h", "hsv_mean_s", "hsv_mean_v",
    "hsv_std_h", "hsv_std_s", "hsv_std_v",
    "glcm_contrast", "glcm_homogeneity", "glcm_energy", "glcm_correlation", "glcm_asm",
    "area_ratio", "perimeter_ratio", "circularity", "aspect_ratio", "extent",
    "dark_spot_ratio", "edge_density", "redness_index", "green_to_red_ratio",
]


def allowed_extensions() -> Tuple[str, ...]:
    return ("png", "jpg", "jpeg", "bmp", "webp")


def decode_image_bytes(data: bytes) -> np.ndarray:
    array = np.frombuffer(data, np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Unable to decode image data.")
    return image


def read_image(path: str) -> np.ndarray:
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Unable to read image: {path}")
    return image


def _safe_stats(arr: np.ndarray) -> Tuple[float, float]:
    if arr.size == 0:
        return 0.0, 0.0
    return float(np.mean(arr)), float(np.std(arr))


def preprocess_image(image_bgr: np.ndarray, target_size: Tuple[int, int] = (256, 256)) -> Tuple[np.ndarray, np.ndarray]:
    resized = cv2.resize(image_bgr, target_size, interpolation=cv2.INTER_AREA)
    denoised = cv2.bilateralFilter(resized, d=7, sigmaColor=60, sigmaSpace=60)

    hsv = cv2.cvtColor(denoised, cv2.COLOR_BGR2HSV)
    # Mask red/orange tomato regions. This also catches mature fruit tones.
    lower_red1 = np.array([0, 40, 30])
    upper_red1 = np.array([15, 255, 255])
    lower_red2 = np.array([160, 40, 30])
    upper_red2 = np.array([179, 255, 255])
    lower_orange = np.array([10, 40, 30])
    upper_orange = np.array([40, 255, 255])

    mask = cv2.inRange(hsv, lower_red1, upper_red1)
    mask |= cv2.inRange(hsv, lower_red2, upper_red2)
    mask |= cv2.inRange(hsv, lower_orange, upper_orange)

    if np.count_nonzero(mask) < (mask.size * 0.02):
        gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    if np.count_nonzero(mask) == 0:
        mask = np.ones(mask.shape, dtype=np.uint8) * 255

    return denoised, mask


def extract_features_from_image(image_bgr: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    processed, mask = preprocess_image(image_bgr)
    mask_bool = mask > 0
    rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(processed, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

    features: Dict[str, float] = {}

    # Color features
    for idx, channel in enumerate([rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]]):
        mean, std = _safe_stats(channel[mask_bool])
        features[["rgb_mean_r", "rgb_mean_g", "rgb_mean_b"][idx]] = mean
        features[["rgb_std_r", "rgb_std_g", "rgb_std_b"][idx]] = std

    for idx, channel in enumerate([hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]]):
        mean, std = _safe_stats(channel[mask_bool])
        features[["hsv_mean_h", "hsv_mean_s", "hsv_mean_v"][idx]] = mean
        features[["hsv_std_h", "hsv_std_s", "hsv_std_v"][idx]] = std

    # Texture features
    gray_small = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    glcm = graycomatrix(gray_small, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    features["glcm_contrast"] = float(graycoprops(glcm, "contrast")[0, 0])
    features["glcm_homogeneity"] = float(graycoprops(glcm, "homogeneity")[0, 0])
    features["glcm_energy"] = float(graycoprops(glcm, "energy")[0, 0])
    features["glcm_correlation"] = float(graycoprops(glcm, "correlation")[0, 0])
    features["glcm_asm"] = float(graycoprops(glcm, "ASM")[0, 0])

    # Shape features
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(contour))
        perimeter = float(cv2.arcLength(contour, True))
        x, y, w, h = cv2.boundingRect(contour)
        bbox_area = float(max(w * h, 1))
        extent = area / bbox_area
        aspect_ratio = w / max(h, 1)
        circularity = (4 * np.pi * area) / max(perimeter ** 2, 1e-6)
    else:
        area = float(np.count_nonzero(mask))
        perimeter = 0.0
        extent = 0.0
        aspect_ratio = 1.0
        circularity = 0.0

    total_pixels = float(mask.size)
    features["area_ratio"] = area / total_pixels
    features["perimeter_ratio"] = perimeter / max(sum(mask.shape), 1)
    features["circularity"] = float(circularity)
    features["aspect_ratio"] = float(aspect_ratio)
    features["extent"] = float(extent)

    # Defect proxies
    roi_gray = gray[mask_bool]
    roi_rgb = rgb[mask_bool]
    dark_threshold = max(25, np.percentile(roi_gray, 18)) if roi_gray.size else 25
    features["dark_spot_ratio"] = float(np.mean(roi_gray < dark_threshold)) if roi_gray.size else 0.0

    edges = cv2.Canny(gray, 80, 160)
    features["edge_density"] = float(np.mean(edges[mask_bool] > 0)) if np.count_nonzero(mask_bool) else 0.0

    if roi_rgb.size:
        r = roi_rgb[:, 0].astype(np.float32)
        g = roi_rgb[:, 1].astype(np.float32)
        b = roi_rgb[:, 2].astype(np.float32)
        redness_index = np.mean((r - (g + b) / 2.0))
        green_to_red = np.mean((g + 1.0) / (r + 1.0))
    else:
        redness_index = 0.0
        green_to_red = 1.0
    features["redness_index"] = float(redness_index)
    features["green_to_red_ratio"] = float(green_to_red)

    vector = np.array([features[name] for name in FEATURE_NAMES], dtype=np.float32)
    return vector, features


def explain_prediction(multiclass_label: str, binary_label: str, features: Dict[str, float]) -> List[str]:
    reasons: List[str] = []
    if features.get("dark_spot_ratio", 0.0) > 0.15:
        reasons.append("noticeable dark regions suggest bruising, rot, or surface spoilage")
    if features.get("redness_index", 0.0) < 12:
        reasons.append("lower red color intensity suggests the fruit may be unripe or not market ready")
    if features.get("glcm_contrast", 0.0) > 220:
        reasons.append("surface texture variation indicates a rough or damaged skin pattern")
    if features.get("circularity", 1.0) < 0.70:
        reasons.append("the shape profile is less regular, which may indicate physical damage")
    if features.get("green_to_red_ratio", 0.0) > 0.95 and multiclass_label == "Unripe":
        reasons.append("the color balance still leans away from mature red tones")
    if multiclass_label == "Old" and not reasons:
        reasons.append("the color and texture pattern is consistent with an aged tomato")
    if multiclass_label == "Damaged" and not reasons:
        reasons.append("surface pattern and darker regions suggest damage")
    if multiclass_label == "Ripe" and not reasons:
        reasons.append("strong red coloration and smoother surface pattern are consistent with ripe fruit")
    if not reasons:
        reasons.append(f"the extracted color, texture, and shape features support the {binary_label.lower()} decision")
    return reasons[:3]
