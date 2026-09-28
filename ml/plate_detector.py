"""License plate localization using classical computer vision + text-guided fallback."""

from __future__ import annotations

import cv2
import numpy as np

from ml.schemas import PlateRegion


class PlateDetector:
    """
    Detect likely plate regions using edge density, contour filtering,
    and text-guided fallback for two-wheelers and high-contrast plates.
    """

    def __init__(self) -> None:
        self._ocr_reader = None

    def _get_ocr_reader(self):
        if self._ocr_reader is None:
            try:
                import easyocr
                self._ocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
            except Exception:
                self._ocr_reader = None
        return self._ocr_reader

    def detect(self, image: np.ndarray) -> PlateRegion:
        height, width = image.shape[:2]
        aspect_ratio = width / max(height, 1)

        # Close-up plate photo: the whole image is the plate
        if 1.1 <= aspect_ratio <= 6.5 and self._looks_like_plate_closeup(image):
            return PlateRegion(0, 0, width, height, 0.95)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.bilateralFilter(gray, 11, 17, 17)
        edges = cv2.Canny(blur, 30, 200)

        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:50]

        image_area = height * width
        candidates: list[PlateRegion] = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            ar = w / max(h, 1)

            # Support both square 2-wheeler plates (AR ~1.1-2.2) and wide car plates (AR ~2.5-6.0)
            if area < image_area * 0.003 or area > image_area * 0.45:
                continue
            if not 1.05 <= ar <= 6.5:
                continue
            if w < 70 or h < 25:
                continue

            roi = gray[y : y + h, x : x + w]
            edge_density = float(np.mean(cv2.Canny(roi, 50, 150) > 0))
            roi_brightness = float(np.mean(roi))
            
            # Position score: plates are rarely in the extreme top 20%
            vertical_center = (y + h / 2) / height
            pos_weight = 1.0 if 0.25 <= vertical_center <= 0.95 else 0.4
            
            # Plates have bright background with high-contrast text
            bright_weight = 1.2 if roi_brightness > 110 else 0.8

            confidence = min(0.95, (0.35 + edge_density * 2.2) * pos_weight * bright_weight)
            candidates.append(PlateRegion(x, y, w, h, confidence))

        best_candidate = max(candidates, key=lambda region: region.confidence) if candidates else None

        # Try text-guided localization first as it is highly accurate for Indian plates
        text_region = self._detect_via_text(image)
        if text_region:
            return text_region

        if best_candidate:
            return best_candidate

        return self._fallback_center_crop(width, height)

    def _detect_via_text(self, image: np.ndarray) -> PlateRegion | None:
        reader = self._get_ocr_reader()
        if reader is None:
            return None

        try:
            results = reader.readtext(image, detail=1, paragraph=False)
            if not results:
                return None

            import re
            plate_boxes = []
            for box, text, conf in results:
                cleaned = re.sub(r"[^A-Z0-9]", "", text.upper())
                # State prefixes or alphanumeric plate chunks
                state_prefixes = ("MH", "DL", "KA", "HR", "UP", "GJ", "TN", "KL", "WB", "RJ", "MP", "AP", "TS", "PB")
                looks_like_plate = (
                    any(cleaned.startswith(p) for p in state_prefixes)
                    or any(p in cleaned for p in state_prefixes)
                    or (len(cleaned) >= 4 and any(c.isdigit() for c in cleaned) and any(c.isalpha() for c in cleaned))
                )
                if looks_like_plate and conf > 0.20:
                    plate_boxes.append(box)

            if not plate_boxes:
                # If no specific state prefix, take boxes located in middle/lower half with digits
                for box, text, conf in results:
                    cleaned = re.sub(r"[^A-Z0-9]", "", text.upper())
                    if len(cleaned) >= 3 and any(c.isdigit() for c in cleaned) and conf > 0.30:
                        plate_boxes.append(box)

            if not plate_boxes:
                return None

            h_img, w_img = image.shape[:2]
            all_x = [pt[0] for box in plate_boxes for pt in box]
            all_y = [pt[1] for box in plate_boxes for pt in box]

            min_x, max_x = max(0, int(min(all_x))), min(w_img, int(max(all_x)))
            min_y, max_y = max(0, int(min(all_y))), min(h_img, int(max(all_y)))

            # Expand margins slightly around the text to capture the full plate rectangle
            pad_x = max(10, int((max_x - min_x) * 0.10))
            pad_y = max(10, int((max_y - min_y) * 0.15))

            x1 = max(0, min_x - pad_x)
            y1 = max(0, min_y - pad_y)
            x2 = min(w_img, max_x + pad_x)
            y2 = min(h_img, max_y + pad_y)

            return PlateRegion(x1, y1, x2 - x1, y2 - y1, 0.94)
        except Exception:
            return None

    def _fallback_center_crop(self, width: int, height: int) -> PlateRegion:
        crop_w = int(width * 0.55)
        crop_h = int(crop_w / 3.0)
        x = (width - crop_w) // 2
        y = int(height * 0.55)
        return PlateRegion(x, y, crop_w, crop_h, 0.35)

    def _looks_like_plate_closeup(self, image: np.ndarray) -> bool:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_val = float(np.mean(gray))
        bright_ratio = float(np.mean(gray > 140))
        return bright_ratio > 0.60 and mean_val > 130
