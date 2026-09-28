"""End-to-end inference pipeline."""

from __future__ import annotations

import time
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from config.settings import settings
from ml.ocr_engine import PlateOCR
from ml.plate_detector import PlateDetector
from ml.schemas import DetectionResult, OCRResult, PlateRegion, TamperAnalysis
from ml.tampering_detector import TamperingDetector


class PlateTamperingPipeline:
    def __init__(self) -> None:
        self.detector = PlateDetector()
        self.ocr: PlateOCR | None = None
        self.tamper_detector = TamperingDetector()

    def _get_ocr(self) -> PlateOCR:
        if self.ocr is None:
            self.ocr = PlateOCR()
        return self.ocr

    def run(self, image_path: Path) -> DetectionResult:
        started = time.perf_counter()
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Unable to read image: {image_path}")

        plate_region = self.detector.detect(image)
        plate_crop = self._crop(image, plate_region)

        ocr_result = self._get_ocr().recognize(plate_crop)
        tamper_result = self.tamper_detector.analyze(plate_crop, ocr_result)

        run_id = uuid4().hex[:8]
        annotated_path = settings.output_dir / f"annotated_{run_id}.jpg"
        crop_path = settings.output_dir / f"plate_{run_id}.jpg"

        annotated = self._annotate(image, plate_region, ocr_result, tamper_result)
        cv2.imwrite(str(annotated_path), annotated)
        cv2.imwrite(str(crop_path), plate_crop)

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return DetectionResult(
            plate_region=plate_region,
            ocr=ocr_result,
            tampering=tamper_result,
            candidate_analysis=None,
            annotated_image_path=str(annotated_path),
            plate_crop_path=str(crop_path),
            processing_ms=elapsed_ms,
        )

    def _crop(self, image: np.ndarray, region: PlateRegion) -> np.ndarray:
        x1, y1, x2, y2 = region.bbox
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        return image[y1:y2, x1:x2]

    def _annotate(
        self,
        image: np.ndarray,
        region: PlateRegion,
        ocr: OCRResult,
        tamper: TamperAnalysis,
    ) -> np.ndarray:
        output = image.copy()
        x1, y1, x2, y2 = region.bbox
        color = (0, 0, 255) if tamper.is_tampered else (0, 200, 0)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

        label = f"{ocr.text or 'UNKNOWN'} | {tamper.tamper_score:.2f}"
        cv2.putText(
            output,
            label,
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
        return output
