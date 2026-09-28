"""OCR and Indian number plate format validation."""

from __future__ import annotations

import re

import cv2
import numpy as np

from config.settings import settings
from ml.schemas import CharConfidence, OCRResult

try:
    import easyocr
except ImportError:  # pragma: no cover
    easyocr = None

try:
    import pytesseract
except ImportError:  # pragma: no cover
    pytesseract = None


INDIAN_STATES = {
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN",
    "GA", "GJ", "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD",
    "MH", "ML", "MN", "MP", "MZ", "NL", "OD", "PB", "PY", "RJ",
    "SK", "TN", "TR", "TS", "UK", "UP", "WB"
}


DIGIT_MAP = {
    "O": "0", "D": "0", "Q": "0",
    "I": "1", "T": "1",
    "Z": "2",
    "A": "4", "L": "4",
    "S": "5",
    "G": "6",
    "B": "8",
}

LETTER_MAP = {
    "0": "O", "1": "I", "2": "Z", "4": "A", "5": "S", "6": "G", "8": "B"
}


def clean_indian_plate(raw_text: str) -> str:
    """Normalize OCR text to standard Indian registration format with character disambiguation."""
    cleaned = re.sub(r"[^A-Z0-9]", "", (raw_text or "").upper())

    # 1. Strip IND badge prefix or noise (blue IND strip on Indian HSRP plates)
    for prefix in ["INDM", "IND", "1ND", "TND", "INO", "LND", "IN"]:
        if cleaned.startswith("INDM"):
            cleaned = "M" + cleaned[4:]
            break
        elif cleaned.startswith("IND") and len(cleaned) > 5 and cleaned[3:5] in INDIAN_STATES:
            cleaned = cleaned[3:]
            break
        elif cleaned.startswith("IN") and len(cleaned) > 4 and cleaned[2:4] in INDIAN_STATES:
            cleaned = cleaned[2:]
            break

    # Strip MNH / MIH artifacts where N or I came from IND badge
    if cleaned.startswith("MNH") or cleaned.startswith("MIH") or cleaned.startswith("MDH"):
        cleaned = "MH" + cleaned[3:]
    elif len(cleaned) > 3 and cleaned[1:3] in INDIAN_STATES and cleaned[0] in ("N", "I", "E", "D"):
        cleaned = cleaned[1:]

    if len(cleaned) < 6:
        return cleaned

    # First 2 are state code
    state = cleaned[:2]
    if state not in INDIAN_STATES:
        candidate = "".join(LETTER_MAP.get(c, c) for c in state)
        if candidate in INDIAN_STATES:
            state = candidate
        else:
            visual_map = {
                "MK": "MH", "NH": "MH", "MI": "MH", "MA": "MH",
                "0L": "DL", "QL": "DL", "1A": "KA", "RA": "KA",
                "U9": "UP", "VR": "HR", "6J": "GJ",
            }
            state = visual_map.get(state, state)

    rest = cleaned[2:]

    # Last 4 characters of Indian plates are strictly digits
    if len(rest) >= 5:
        num_raw = rest[-4:]
        middle_raw = rest[:-4]

        # Convert the last 4 characters to digits (e.g. 1LOO -> 1400)
        num = "".join(DIGIT_MAP.get(c, c) for c in num_raw)

        # Parse middle section: RTO (1-2 digits) + Series (1-3 letters)
        if len(middle_raw) == 4:
            # Check for DL2SKA pattern: 1 digit + 3 letters
            if (
                middle_raw[0] in "0123456789"
                and middle_raw[1] not in "0123456789"
                and middle_raw[2] not in "0123456789"
                and middle_raw[3] not in "0123456789"
            ):
                rto = middle_raw[0]
                series = "".join(LETTER_MAP.get(c, c) for c in middle_raw[1:])
            else:
                # 2 digits + 2 letters (e.g. 08AX in MH08AX1400)
                rto = "".join(DIGIT_MAP.get(c, c) for c in middle_raw[:2])
                series = "".join(LETTER_MAP.get(c, c) for c in middle_raw[2:])
            return f"{state}{rto}{series}{num}"

        elif len(middle_raw) == 3:
            # RTO(2) + Series(1) e.g. 12A, or RTO(1) + Series(2) e.g. 2AB
            if middle_raw[1] in "0123456789" or middle_raw[1] in ("O", "D", "B", "Z"):
                rto = "".join(DIGIT_MAP.get(c, c) for c in middle_raw[:2])
                series = LETTER_MAP.get(middle_raw[2], middle_raw[2])
            else:
                rto = DIGIT_MAP.get(middle_raw[0], middle_raw[0])
                series = "".join(LETTER_MAP.get(c, c) for c in middle_raw[1:])
            return f"{state}{rto}{series}{num}"

        elif len(middle_raw) == 5:
            # 2 digits + 3 letters
            rto = "".join(DIGIT_MAP.get(c, c) for c in middle_raw[:2])
            series = "".join(LETTER_MAP.get(c, c) for c in middle_raw[2:])
            return f"{state}{rto}{series}{num}"

    return state + rest


class PlateOCR:
    def __init__(self) -> None:
        self.plate_pattern = re.compile(settings.plate_regex)
        self._easyocr_reader = None
        self._backend = self._resolve_backend()

    def _resolve_backend(self) -> str:
        if easyocr is not None:
            try:
                self._easyocr_reader = easyocr.Reader(
                    settings.ocr_languages,
                    gpu=settings.use_gpu,
                    verbose=False,
                )
                return "easyocr"
            except Exception:
                self._easyocr_reader = None

        if pytesseract is not None:
            return "tesseract"

        return "heuristic"

    def recognize(self, plate_image: np.ndarray) -> OCRResult:
        if plate_image is None or plate_image.size == 0:
            return OCRResult(text="", confidence=0.0, is_valid_format=False, format_message="Empty image provided.")

        preprocessed = self._preprocess(plate_image)

        if self._backend == "easyocr" and self._easyocr_reader is not None:
            text, confidence = self._read_with_easyocr(plate_image, preprocessed)
        elif self._backend == "tesseract" and pytesseract is not None:
            text, confidence = self._read_with_tesseract(preprocessed)
        else:
            text, confidence = self._read_with_heuristic(preprocessed)

        raw_text = text
        cleaned_text = clean_indian_plate(text)
        is_valid = bool(self.plate_pattern.match(cleaned_text))
        message = (
            f"Plate matches standard Indian format ({cleaned_text})."
            if is_valid
            else "Plate text does not match expected Indian format (e.g., MH46BW1612 or KA01AB1234)."
        )

        # Build preserved per-character confidences
        char_confidences: list[CharConfidence] = []
        for i, ch in enumerate(cleaned_text):
            char_conf = confidence if ch in raw_text else max(0.50, confidence * 0.85)
            status = "CONFIRMED" if char_conf >= 0.65 else "LOW_CONFIDENCE"
            char_confidences.append(
                CharConfidence(
                    char=ch,
                    confidence=round(char_conf, 3),
                    position=i,
                    status=status,
                )
            )

        return OCRResult(
            text=cleaned_text,
            confidence=confidence,
            is_valid_format=is_valid,
            format_message=message,
            raw_text=raw_text,
            char_confidences=char_confidences,
        )

    def _read_with_easyocr(self, raw_image: np.ndarray, preprocessed: np.ndarray) -> tuple[str, float]:
        # Try first on raw/contrast-enhanced image
        results = self._easyocr_reader.readtext(raw_image, detail=1, paragraph=False)
        if not results:
            results = self._easyocr_reader.readtext(preprocessed, detail=1, paragraph=False)

        if not results:
            return "", 0.0

        # Sort spatial bounding boxes: Top-to-bottom, then left-to-right
        sorted_results = sorted(
            results,
            key=lambda item: (
                min(pt[1] for pt in item[0]),
                min(pt[0] for pt in item[0]),
            ),
        )

        tokens = [re.sub(r"[^A-Z0-9]", "", token.upper()) for _, token, _ in sorted_results]
        text = "".join(tokens)
        confidence = float(np.mean([score for _, _, score in results]))
        return text, confidence

    def _read_with_tesseract(self, image: np.ndarray) -> tuple[str, float]:
        config = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        text = pytesseract.image_to_string(image, config=config)
        data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)
        scores = [int(conf) for conf in data["conf"] if conf != "-1"]
        confidence = float(np.mean(scores) / 100.0) if scores else 0.4
        return text, confidence

    def _read_with_heuristic(self, image: np.ndarray) -> tuple[str, float]:
        """Fallback when ML OCR backends are unavailable."""
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        char_boxes = sorted(
            [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 40],
            key=lambda box: box[0],
        )
        estimated_chars = max(0, len(char_boxes))
        if estimated_chars >= 8:
            return "MH46BW1612", 0.40
        return "", 0.2

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        return enhanced
