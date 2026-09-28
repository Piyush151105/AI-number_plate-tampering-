"""Feature extraction helpers for tampering heuristics."""

from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import local_binary_pattern


def normalize_plate(image: np.ndarray, size: tuple[int, int] = (224, 64)) -> np.ndarray:
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def edge_density(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    edges = cv2.Canny(gray, 40, 120)
    return float(np.mean(edges > 0))


def texture_variance(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    hist = hist.astype(float)
    hist /= hist.sum() + 1e-8
    return float(np.var(hist))


def _plate_gray(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    if len(image.shape) == 3:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        blue_mask = cv2.inRange(hsv, (85, 35, 35), (145, 255, 255))
        gray[blue_mask > 0] = 255
    return gray


def _character_contours(gray: np.ndarray) -> list[tuple[float, tuple[int, int, int, int], np.ndarray]]:
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    plate_area = gray.shape[0] * gray.shape[1]
    min_area = plate_area * 0.0015
    max_area = plate_area * 0.10

    boxes: list[tuple[float, tuple[int, int, int, int], np.ndarray]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            boxes.append((area, cv2.boundingRect(contour), contour))
    return boxes


def character_alignment_score(image: np.ndarray) -> float:
    gray = _plate_gray(image)
    boxes = _character_contours(gray)
    if len(boxes) < 4:
        return 0.12

    centers_y = [y + h / 2 for _, (_, y, _, h), _ in boxes]
    spread = np.std(centers_y) / max(image.shape[0], 1)
    return float(min(1.0, spread * 8))


def color_inconsistency(image: np.ndarray) -> float:
    if len(image.shape) != 3:
        return 0.0

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    split_x = int(hsv.shape[1] * 0.18)
    left = hsv[:, split_x : hsv.shape[1] // 2]
    right = hsv[:, hsv.shape[1] // 2 :]
    diff = np.abs(left.mean(axis=(0, 1)) - right.mean(axis=(0, 1)))
    return float(np.clip(diff.mean() / 45.0, 0.0, 1.0))


def smudge_patch_score(image: np.ndarray) -> float:
    """Detect oversized irregular dark blobs beyond normal character size."""
    gray = _plate_gray(image)
    boxes = _character_contours(gray)
    if len(boxes) < 4:
        return 0.0

    areas = np.array([item[0] for item in boxes], dtype=float)
    median_area = float(np.median(areas))
    if median_area <= 0:
        return 0.0

    plate_area = gray.shape[0] * gray.shape[1]
    blob_scores: list[float] = []

    for area, (x, y, bw, bh), contour in boxes:
        relative_size = area / median_area
        if relative_size < 2.4:
            continue

        hull = cv2.convexHull(contour)
        solidity = area / (cv2.contourArea(hull) + 1e-8)
        aspect = bw / max(bh, 1)
        size_ratio = area / plate_area

        irregular_block = solidity < 0.78 or aspect > 2.2
        much_larger = relative_size >= 3.0
        if much_larger or (relative_size >= 2.4 and irregular_block):
            score = min(
                1.0,
                (relative_size - 2.0) / 3.5 + (1.0 - solidity) * 0.45 + size_ratio * 2.0,
            )
            blob_scores.append(score)

    if not blob_scores:
        return 0.0
    return float(min(1.0, max(blob_scores)))


def background_smudge_score(image: np.ndarray) -> float:
    """Detect gray residue and uneven patches on the plate's white background."""
    gray = _plate_gray(image)
    white_mask = gray > 165
    if int(white_mask.sum()) < 100:
        return 0.0

    residue_mask = (gray > 95) & (gray < 210) & white_mask
    residue_ratio = float(residue_mask.sum() / (white_mask.sum() + 1e-8))

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    texture_noise = float(np.var(laplacian[white_mask]) / 900.0) if white_mask.any() else 0.0

    return float(np.clip(residue_ratio * 4.0 + min(1.0, texture_noise) * 0.35, 0.0, 1.0))
