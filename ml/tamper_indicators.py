"""Forensic Tampering Indicators for vehicle number plates.

Detects plates with hidden, damaged, blurred, missing, or altered characters:
1. Occlusion (patches, tape, mud, stickers)
2. Broken Characters (fragmented contours, disconnected strokes)
3. Blur / Defocus (selective smearing, low Laplacian variance)
4. Altered Font / Spacing (non-uniform character pitch, kerning variance, baseline jitter)
5. Low OCR Confidence (character confidence drops, unreadable glyphs)
"""

from __future__ import annotations

import cv2
import numpy as np

from ml.feature_extractor import (
    background_smudge_score,
    character_alignment_score,
    smudge_patch_score,
)
from ml.schemas import CharConfidence, OCRResult, TamperAlert


def detect_occlusion(plate_image: np.ndarray) -> TamperAlert:
    """Detect stickers, tape, dark paint patches, or physical occlusion covering plate."""
    if plate_image is None or plate_image.size == 0:
        return TamperAlert(
            indicator="Occlusion",
            score=0.0,
            severity="low",
            description="No image provided for occlusion analysis.",
            detected=False,
        )

    patch_score = smudge_patch_score(plate_image)
    smudge_score = background_smudge_score(plate_image)
    combined = float(np.clip(patch_score * 0.7 + smudge_score * 0.3, 0.0, 1.0))

    if combined >= 0.50:
        severity = "high"
        desc = "High probability of physical occlusion (tape, sticker, or paint mask detected over characters)."
        detected = True
    elif combined >= 0.28:
        severity = "medium"
        desc = "Possible partial occlusion or localized dark blotch on the plate face."
        detected = True
    else:
        severity = "low"
        desc = "No abnormal opaque occlusion or masking patches detected."
        detected = False

    return TamperAlert(
        indicator="Occlusion",
        score=round(combined, 4),
        severity=severity,
        description=desc,
        detected=detected,
    )


def detect_broken_characters(plate_image: np.ndarray) -> TamperAlert:
    """Detect fragmented strokes, scratched paint, or discontinuous character contours."""
    if plate_image is None or plate_image.size == 0:
        return TamperAlert(
            indicator="Broken Characters",
            score=0.0,
            severity="low",
            description="No image provided for contour analysis.",
            detected=False,
        )

    gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY) if len(plate_image.shape) == 3 else plate_image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape[:2]
    plate_area = h * w

    char_contours = []
    tiny_fragments = 0

    for c in contours:
        area = cv2.contourArea(c)
        bx, by, bw, bh = cv2.boundingRect(c)
        ar = bw / max(bh, 1)

        # True character size range
        if plate_area * 0.003 < area < plate_area * 0.12 and 0.15 <= ar <= 1.8:
            char_contours.append((bx, by, bw, bh, area, c))
        elif plate_area * 0.0003 < area <= plate_area * 0.003 and bh > 5:
            # Small fragments in character zone
            if 0.15 * h < by < 0.85 * h:
                tiny_fragments += 1

    # Ratio of broken fragments to full characters
    num_chars = len(char_contours)
    if num_chars >= 4:
        fragment_ratio = tiny_fragments / max(num_chars, 1)
        # Check aspect ratio anomalies (e.g. split halves of an 8 or B)
        unusual_splits = sum(1 for _, _, bw, bh, _, _ in char_contours if bh < 0.35 * h or bw < 0.03 * w)
        split_ratio = unusual_splits / max(num_chars, 1)
        score = float(np.clip(fragment_ratio * 0.25 + split_ratio * 0.5, 0.0, 1.0))
    else:
        score = 0.15

    if score >= 0.45:
        severity = "high"
        desc = "Disconnected strokes or fragmented character outlines indicating physical damage or paint scratching."
        detected = True
    elif score >= 0.25:
        severity = "medium"
        desc = "Minor stroke discontinuity or edge irregularities detected around character edges."
        detected = True
    else:
        severity = "low"
        desc = "Character strokes exhibit normal continuous typography."
        detected = False

    return TamperAlert(
        indicator="Broken Characters",
        score=round(score, 4),
        severity=severity,
        description=desc,
        detected=detected,
    )


def detect_blur(plate_image: np.ndarray) -> TamperAlert:
    """Detect optical defocus, motion blur, or intentional digital smearing on characters."""
    if plate_image is None or plate_image.size == 0:
        return TamperAlert(
            indicator="Blur",
            score=0.0,
            severity="low",
            description="No image provided for sharpness analysis.",
            detected=False,
        )

    gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY) if len(plate_image.shape) == 3 else plate_image

    # Global Laplacian variance
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Check local horizontal band blur (where characters reside)
    h = gray.shape[0]
    char_band = gray[int(h * 0.2) : int(h * 0.8), :]
    band_var = float(cv2.Laplacian(char_band, cv2.CV_64F).var()) if char_band.size > 0 else laplacian_var

    # Sharp plates typically have variance > 400.
    # Blurred plates have variance < 120.
    min_var = min(laplacian_var, band_var)

    if min_var < 80:
        blur_score = 0.85
        severity = "high"
        desc = f"Severe blur detected (sharpness metric {min_var:.1f}). Characters are obscured by heavy blur."
        detected = True
    elif min_var < 180:
        blur_score = 0.55
        severity = "medium"
        desc = f"Moderate blur detected (sharpness metric {min_var:.1f}). Character legibility is compromised."
        detected = True
    elif min_var < 300:
        blur_score = 0.25
        severity = "low"
        desc = f"Slight softness detected (sharpness metric {min_var:.1f}). Characters remain legible."
        detected = False
    else:
        blur_score = 0.05
        severity = "low"
        desc = f"Plate image is sharp and in focus (sharpness metric {min_var:.1f})."
        detected = False

    return TamperAlert(
        indicator="Blur",
        score=round(blur_score, 4),
        severity=severity,
        description=desc,
        detected=detected,
    )


def detect_altered_font_spacing(plate_image: np.ndarray) -> TamperAlert:
    """Detect irregular character kerning/spacing, anomalous widths, or baseline misalignment."""
    if plate_image is None or plate_image.size == 0:
        return TamperAlert(
            indicator="Altered Font/Spacing",
            score=0.0,
            severity="low",
            description="No image provided for font/spacing analysis.",
            detected=False,
        )

    gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY) if len(plate_image.shape) == 3 else plate_image
    alignment_score = character_alignment_score(plate_image)

    # Contour-based inter-character gap analysis
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape[:2]
    plate_area = h * w

    boxes = []
    for c in contours:
        area = cv2.contourArea(c)
        bx, by, bw, bh = cv2.boundingRect(c)
        if plate_area * 0.003 < area < plate_area * 0.12 and 0.25 * h < bh < 0.9 * h:
            boxes.append((bx, by, bw, bh))

    boxes.sort(key=lambda b: b[0])  # Sort left-to-right

    spacing_score = 0.1
    if len(boxes) >= 5:
        # Calculate gaps between consecutive bounding boxes
        gaps = []
        widths = [b[2] for b in boxes]
        for i in range(len(boxes) - 1):
            gap = boxes[i + 1][0] - (boxes[i][0] + boxes[i][2])
            gaps.append(max(0, gap))

        gap_std = float(np.std(gaps)) / max(np.mean(widths), 1.0)
        width_std = float(np.std(widths)) / max(np.mean(widths), 1.0)

        # High variance in spacing or width indicates non-standard fonts or inserted characters
        spacing_score = float(np.clip(gap_std * 0.4 + width_std * 0.3 + alignment_score * 0.3, 0.0, 1.0))
    else:
        spacing_score = alignment_score

    if spacing_score >= 0.45:
        severity = "high"
        desc = "Noticeable font irregularity, irregular character spacing, or baseline jitter detected."
        detected = True
    elif spacing_score >= 0.28:
        severity = "medium"
        desc = "Mild spacing or character width variation across the registration row."
        detected = True
    else:
        severity = "low"
        desc = "Standard uniform character spacing and font alignment observed."
        detected = False

    return TamperAlert(
        indicator="Altered Font/Spacing",
        score=round(spacing_score, 4),
        severity=severity,
        description=desc,
        detected=detected,
    )


def detect_low_ocr_confidence(
    ocr_confidence: float,
    char_confidences: list[CharConfidence] | None = None,
    raw_text: str = "",
) -> TamperAlert:
    """Flag OCR confidence drops, unreadable wildcard positions (*, ?, _), or low character confidence."""
    has_wildcard = any(c in (raw_text or "") for c in ("*", "?", "_", "#"))

    low_chars = 0
    total_chars = len(char_confidences) if char_confidences else len(raw_text)

    if char_confidences:
        low_chars = sum(1 for c in char_confidences if c.confidence < 0.60 or c.status != "CONFIRMED")

    # Score calculation
    conf_deficit = max(0.0, 1.0 - ocr_confidence)
    wildcard_penalty = 0.40 if has_wildcard else 0.0
    low_char_penalty = (low_chars / max(total_chars, 1)) * 0.50

    combined = float(np.clip(conf_deficit * 0.4 + wildcard_penalty + low_char_penalty, 0.0, 1.0))

    if combined >= 0.50:
        severity = "high"
        desc = f"Severe OCR confidence drop (overall {ocr_confidence:.1%}, {low_chars} uncertain/occluded characters)."
        detected = True
    elif combined >= 0.25:
        severity = "medium"
        desc = f"Moderate OCR uncertainty detected (overall {ocr_confidence:.1%})."
        detected = True
    else:
        severity = "low"
        desc = f"High OCR confidence across registration characters ({ocr_confidence:.1%})."
        detected = False

    return TamperAlert(
        indicator="Low OCR Confidence",
        score=round(combined, 4),
        severity=severity,
        description=desc,
        detected=detected,
    )


class TamperIndicatorAnalyzer:
    """Orchestrates multi-signal forensic indicator analysis."""

    def analyze(
        self,
        plate_image: np.ndarray | None,
        ocr_result: OCRResult | None = None,
        char_confidences: list[CharConfidence] | None = None,
    ) -> list[TamperAlert]:
        alerts: list[TamperAlert] = []

        # 1. Occlusion
        if plate_image is not None:
            alerts.append(detect_occlusion(plate_image))
            # 2. Broken Characters
            alerts.append(detect_broken_characters(plate_image))
            # 3. Blur
            alerts.append(detect_blur(plate_image))
            # 4. Altered Font / Spacing
            alerts.append(detect_altered_font_spacing(plate_image))
        else:
            for ind in ["Occlusion", "Broken Characters", "Blur", "Altered Font/Spacing"]:
                alerts.append(
                    TamperAlert(
                        indicator=ind,
                        score=0.0,
                        severity="low",
                        description="No image provided for visual inspection.",
                        detected=False,
                    )
                )

        # 5. Low OCR Confidence
        overall_conf = ocr_result.confidence if ocr_result else 0.0
        raw_text = ocr_result.raw_text if ocr_result and ocr_result.raw_text else (ocr_result.text if ocr_result else "")
        chars = char_confidences or (ocr_result.char_confidences if ocr_result else [])
        alerts.append(detect_low_ocr_confidence(overall_conf, chars, raw_text))

        return alerts


tamper_indicator_analyzer = TamperIndicatorAnalyzer()
