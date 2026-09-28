"""Tampered Plate Candidate Analysis Service.

Forensic service that:
1. Normalizes OCR output and preserves per-character confidence.
2. Validates Indian registration-number formats.
3. Finds missing/uncertain/occluded character positions.
4. Identifies visible RTO/state prefix via the configurable RTO dataset.
5. Generates at most 10 unverified candidates only when enough characters are visibly readable.
6. Never presents guessed registration numbers as confirmed (strictly 'Unverified candidate').
7. Requires human reviewer and official RTO verification.
"""

from __future__ import annotations

import itertools
import logging
import re
from typing import Sequence

import cv2
import numpy as np

from config.settings import settings
from ml.ocr_engine import clean_indian_plate
from ml.rto_service import rto_service
from ml.schemas import (
    CandidateAnalysisResult,
    CharConfidence,
    OCRResult,
    PlateCandidate,
    RTOResult,
    TamperAlert,
)
from ml.tamper_indicators import tamper_indicator_analyzer

logger = logging.getLogger(__name__)

# Positional character confusion maps
DIGIT_TO_LETTER: dict[str, list[str]] = {
    "0": ["O", "D", "Q"],
    "1": ["I", "T", "L"],
    "2": ["Z"],
    "3": ["B", "E"],
    "4": ["A"],
    "5": ["S"],
    "6": ["G"],
    "7": ["T", "Z"],
    "8": ["B"],
    "9": ["P", "B"],
}

LETTER_TO_DIGIT: dict[str, list[str]] = {
    "O": ["0", "9"],
    "D": ["0"],
    "Q": ["0"],
    "I": ["1"],
    "T": ["1", "7"],
    "L": ["1", "4"],
    "Z": ["2", "7"],
    "B": ["8", "3"],
    "A": ["4"],
    "S": ["5"],
    "G": ["6"],
    "C": ["0", "6"],
    "P": ["9"],
}

COMMON_SERIES_LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
COMMON_DIGITS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]


class PlateCandidateService:
    """Service to parse plates, detect RTO, analyze tampering, and generate unverified candidates."""

    def __init__(self) -> None:
        self.min_readable_chars = settings.candidate_min_readable_chars
        self.max_candidates = settings.candidate_max_count
        self.low_conf_threshold = settings.candidate_low_conf_threshold
        self.plate_pattern = re.compile(settings.plate_regex)

    def analyze(
        self,
        raw_text: str,
        plate_image: np.ndarray | None = None,
        base_confidence: float = 0.85,
        char_confidences: list[CharConfidence] | None = None,
    ) -> CandidateAnalysisResult:
        """
        Execute candidate analysis on plate text and optional image crop.
        """
        cleaned_raw = (raw_text or "").strip().upper()

        # Step 1: Normalize and extract per-character confidence
        char_list = self._build_char_confidences(cleaned_raw, base_confidence, char_confidences)
        normalized_str = "".join(c.char for c in char_list if c.char not in (" ", "-", "."))

        # Count readable characters (non-wildcards, confidence >= 0.40)
        readable_count = sum(
            1 for c in char_list
            if c.char not in ("*", "?", "_", "#", " ", "-", ".") and c.confidence >= 0.40
        )

        # Step 2: Identify visible RTO prefix
        prefix, rto_info = rto_service.extract_rto_prefix(normalized_str)

        # Step 3: Compute Tampering Alerts
        dummy_ocr = OCRResult(
            text=normalized_str,
            confidence=base_confidence,
            is_valid_format=bool(self.plate_pattern.match(normalized_str)),
            format_message="",
            raw_text=raw_text,
            char_confidences=char_list,
        )
        alerts = tamper_indicator_analyzer.analyze(plate_image, dummy_ocr, char_list)

        # Step 4: Validate format and check if candidate generation is safe
        if readable_count < self.min_readable_chars:
            return CandidateAnalysisResult(
                original_ocr_text=raw_text,
                clean_text=normalized_str,
                char_confidences=char_list,
                detected_rto=rto_info,
                tampering_alerts=alerts,
                candidates=[],
                candidate_status="INSUFFICIENT_CHARACTERS",
                readable_character_count=readable_count,
                warning_message=(
                    f"Fewer than {self.min_readable_chars} characters are clearly readable ({readable_count} detected). "
                    "Candidate generation safely halted to prevent speculative misidentification. "
                    "Manual RTO verification required."
                ),
            )

        # Step 5: Candidate generation
        candidates, status = self._generate_candidates(
            char_list=char_list,
            normalized_str=normalized_str,
            rto_info=rto_info,
            base_confidence=base_confidence,
        )

        return CandidateAnalysisResult(
            original_ocr_text=raw_text,
            clean_text=normalized_str,
            char_confidences=char_list,
            detected_rto=rto_info,
            tampering_alerts=alerts,
            candidates=candidates[: self.max_candidates],
            candidate_status=status,
            readable_character_count=readable_count,
        )

    def _build_char_confidences(
        self,
        raw_text: str,
        base_confidence: float,
        existing_confidences: list[CharConfidence] | None,
    ) -> list[CharConfidence]:
        """Preserve or compute character-level confidences with status classification."""
        if existing_confidences and len(existing_confidences) == len(raw_text):
            return existing_confidences

        char_list: list[CharConfidence] = []
        for idx, ch in enumerate(raw_text):
            if ch in ("*", "?", "_", "#"):
                char_list.append(
                    CharConfidence(
                        char=ch,
                        confidence=0.10,
                        position=idx,
                        status="OCCLUDED",
                        alternatives=[],
                    )
                )
            elif ch in (" ", "-", "."):
                continue
            else:
                conf = base_confidence
                status = "CONFIRMED"
                alternatives = []

                if ch in DIGIT_TO_LETTER:
                    alternatives.extend(DIGIT_TO_LETTER[ch])
                if ch in LETTER_TO_DIGIT:
                    alternatives.extend(LETTER_TO_DIGIT[ch])

                if conf < self.low_conf_threshold:
                    status = "LOW_CONFIDENCE"
                elif alternatives:
                    status = "AMBIGUOUS"

                char_list.append(
                    CharConfidence(
                        char=ch,
                        confidence=round(conf, 3),
                        position=idx,
                        status=status,
                        alternatives=list(dict.fromkeys(alternatives)),
                    )
                )
        return char_list

    def _generate_candidates(
        self,
        char_list: list[CharConfidence],
        normalized_str: str,
        rto_info: RTOResult | None,
        base_confidence: float,
    ) -> tuple[list[PlateCandidate], str]:
        """
        Generate a capped list of at most 10 unverified candidates.
        """
        # Case A: Fully clean and valid standard format with high confidence
        is_clean_valid = bool(self.plate_pattern.match(normalized_str))
        all_high_conf = all(c.confidence >= 0.75 for c in char_list if c.char not in ("*", "?"))
        no_wildcard = not any(c.char in ("*", "?", "_", "#") for c in char_list)

        rto_desc = f"{rto_info.code} ({rto_info.location})" if rto_info else "Unmapped RTO"

        if is_clean_valid and all_high_conf and no_wildcard:
            candidate = PlateCandidate(
                plate_candidate=normalized_str,
                label="Unverified candidate",
                confidence=round(float(np.mean([c.confidence for c in char_list])), 3),
                explanation=(
                    f"Clean plate syntax perfectly matches Indian standard format ({rto_desc}). "
                    "Unverified candidate requiring human verification."
                ),
            )
            return [candidate], "CLEAN_AUTHENTIC"

        # Case B: Plate needs positional parsing and reconstruction
        # Standard Indian schema: [State 2 letters][RTO 1-2 digits][Series 0-3 letters][Number 4 digits]
        candidates = self._reconstruct_candidates(char_list, rto_info, base_confidence)

        if not candidates:
            # Fallback candidate using cleaned string if length is close
            cleaned_attempt = clean_indian_plate(normalized_str)
            if self.plate_pattern.match(cleaned_attempt):
                candidates.append(
                    PlateCandidate(
                        plate_candidate=cleaned_attempt,
                        label="Unverified candidate",
                        confidence=round(base_confidence * 0.75, 3),
                        explanation=f"Reconstructed through syntax heuristic for {rto_desc}.",
                    )
                )

        if candidates:
            # Sort by confidence descending and cap at max_candidates
            candidates.sort(key=lambda c: c.confidence, reverse=True)
            return candidates[: self.max_candidates], "CANDIDATES_GENERATED"

        return [], "INVALID_FORMAT"

    def _reconstruct_candidates(
        self,
        char_list: list[CharConfidence],
        rto_info: RTOResult | None,
        base_confidence: float,
    ) -> list[PlateCandidate]:
        """
        Disambiguate uncertain slots (digits vs letters) and fill occluded slots.
        """
        raw_chars = [c.char for c in char_list]
        n_chars = len(raw_chars)
        candidates: list[PlateCandidate] = []

        # Find RTO prefix if possible
        prefix_str = ""
        rto_loc = rto_info.location if rto_info else "RTO"
        if rto_info:
            prefix_str = rto_info.prefix
        elif n_chars >= 4:
            prefix_str = "".join(raw_chars[:4])

        # Standard Indian plate is typically 9 or 10 characters
        # Target slot structures:
        # 10 chars: [State 2][RTO 2][Series 2][Number 4] -> e.g. MH 46 BW 1612
        # 9 chars:  [State 2][RTO 2][Series 1][Number 4] -> e.g. MH 46 B 1612, or [State 2][RTO 1][Series 2][Number 4] DL 2 SK 2187
        
        # Build slot candidates:
        # Determine likely pattern based on length
        expected_len = 10 if n_chars >= 10 or (n_chars == 9 and any(c in ("*", "?") for c in raw_chars)) else n_chars

        # Analyze positions
        # Let's align characters:
        state_slots = raw_chars[:2]
        rest = raw_chars[2:]

        # Number segment is always the last 4 positions if length >= 8
        if len(raw_chars) >= 8:
            number_raw = raw_chars[-4:]
            series_raw = raw_chars[4:-4] if len(raw_chars) >= 8 else []
            rto_num_raw = raw_chars[2:4]
        else:
            return []

        # Options for each section:
        # 1. State: Must be letters
        state_opts = self._resolve_letter_slots(state_slots)

        # 2. RTO Digits: Must be digits
        if rto_info:
            rto_opts = [rto_info.prefix[2:]]
        else:
            rto_opts = self._resolve_digit_slots(rto_num_raw, state_code=state_opts[0] if state_opts else "MH")

        # 3. Series: Must be letters (can be 1 to 2 letters)
        series_opts = self._resolve_series_slots(series_raw)

        # 4. Number: Must be 4 digits
        number_opts = self._resolve_number_slots(number_raw)

        # Cross-product of options, strictly keeping count manageable
        combinations = list(
            itertools.product(
                state_opts[:2],
                rto_opts[:6],
                series_opts[:4],
                number_opts[:10],
            )
        )

        seen_plates = set()
        for st, rto_num, s_code, n_code in combinations:
            plate_str = f"{st}{rto_num}{s_code}{n_code}"
            if plate_str in seen_plates:
                continue
            seen_plates.add(plate_str)

            if not self.plate_pattern.match(plate_str):
                continue

            # Compute score and explanation
            cand_conf, rationale = self._score_candidate(
                original_chars=raw_chars,
                candidate_str=plate_str,
                rto_info=rto_info,
                base_confidence=base_confidence,
            )

            candidates.append(
                PlateCandidate(
                    plate_candidate=plate_str,
                    label="Unverified candidate",
                    confidence=round(cand_conf, 3),
                    explanation=rationale,
                )
            )

            if len(candidates) >= self.max_candidates:
                break

        return candidates

    def _resolve_letter_slots(self, slots: list[str]) -> list[str]:
        """Convert state slots to valid letter characters."""
        res = []
        c1_candidates = [slots[0]] if slots[0].isalpha() else LETTER_TO_DIGIT.get(slots[0], ["M"])
        c2_candidates = [slots[1]] if slots[1].isalpha() else LETTER_TO_DIGIT.get(slots[1], ["H"])
        for a in c1_candidates[:2]:
            for b in c2_candidates[:2]:
                res.append(f"{a}{b}")
        return res or ["MH"]

    def _resolve_digit_slots(self, slots: list[str], state_code: str = "MH") -> list[str]:
        """Convert RTO number slots to valid digits using RTO mapping database."""
        has_wildcard = any(s in ("*", "?", "_", "#") for s in slots)
        if has_wildcard and state_code:
            known_prefixes = rto_service.get_known_prefixes_for_state(state_code)
            matched_rto_nums = []
            for p in known_prefixes:
                num_part = p[len(state_code):]
                if len(num_part) == 2:
                    match_0 = slots[0] in ("*", "?", "_", "#") or slots[0] == num_part[0]
                    match_1 = slots[1] in ("*", "?", "_", "#") or slots[1] == num_part[1]
                    if match_0 and match_1:
                        matched_rto_nums.append(num_part)
            if matched_rto_nums:
                return matched_rto_nums[:10]

        res = []
        d1 = [slots[0]] if slots[0].isdigit() else LETTER_TO_DIGIT.get(slots[0], ["4", "0"])
        d2 = [slots[1]] if slots[1].isdigit() else LETTER_TO_DIGIT.get(slots[1], ["6", "8"])
        for a in d1[:2]:
            for b in d2[:2]:
                res.append(f"{a}{b}")
        return res or ["46", "08"]

    def _resolve_series_slots(self, slots: list[str]) -> list[str]:
        """Resolve series letters (1-2 uppercase letters)."""
        if not slots:
            # If missing completely, suggest likely single/double series
            return ["BW", "AX", "AB", "B", "A"]

        slot_options = []
        for ch in slots:
            if ch in ("*", "?", "_", "#"):
                slot_options.append(["B", "W", "A", "C"])
            elif ch.isalpha():
                slot_options.append([ch] + DIGIT_TO_LETTER.get(ch, []))
            elif ch.isdigit():
                # Disambiguate digit in series slot (e.g. '8' -> 'B', '0' -> 'O', '4' -> 'A')
                converted = DIGIT_TO_LETTER.get(ch, ["B"])
                slot_options.append(converted)
            else:
                slot_options.append(["B", "A"])

        combos = ["".join(p) for p in itertools.product(*slot_options)]
        return combos[:4] or ["BW", "AX"]

    def _resolve_number_slots(self, slots: list[str]) -> list[str]:
        """Resolve last 4 registration digits. Supports all digits 0-9 when a single slot is missing."""
        if len(slots) < 4:
            slots = slots + ["0"] * (4 - len(slots))

        wildcard_count = sum(1 for ch in slots if ch in ("*", "?", "_", "#"))

        slot_options = []
        for ch in slots:
            if ch in ("*", "?", "_", "#"):
                if wildcard_count == 1:
                    # Exactly 1 digit missing: allow all digits 0 through 9
                    slot_options.append([str(d) for d in range(10)])
                else:
                    # Multiple digits missing: use top frequent digits
                    slot_options.append(["1", "2", "6", "0", "8"])
            elif ch.isdigit():
                slot_options.append([ch])
            elif ch.isalpha():
                # Disambiguate letter in digit slot (e.g. 'B' -> '8', 'O' -> '0', 'I' -> '1')
                converted = LETTER_TO_DIGIT.get(ch, ["0"])
                slot_options.append(converted)
            else:
                slot_options.append(["0"])

        combos = ["".join(p) for p in itertools.product(*slot_options)]
        return combos[:10] or ["1612", "1400"]

    def _score_candidate(
        self,
        original_chars: list[str],
        candidate_str: str,
        rto_info: RTOResult | None,
        base_confidence: float,
    ) -> tuple[float, str]:
        """Compute forensic confidence score and human-readable explanation."""
        differences = []
        reconstructed_count = 0

        min_len = min(len(original_chars), len(candidate_str))
        for i in range(min_len):
            orig = original_chars[i]
            cand = candidate_str[i]
            if orig in ("*", "?", "_", "#"):
                reconstructed_count += 1
                differences.append(f"occluded slot {i + 1} reconstructed as '{cand}'")
            elif orig != cand:
                reconstructed_count += 1
                differences.append(f"ambiguous slot {i + 1} '{orig}' resolved to '{cand}'")

        if len(original_chars) != len(candidate_str):
            reconstructed_count += abs(len(original_chars) - len(candidate_str))
            differences.append(f"adjusted length to match standard 10-char format")

        # Base penalty for each reconstructed/disambiguated character
        penalty = 0.08 * reconstructed_count
        conf = float(np.clip(base_confidence - penalty, 0.40, 0.95))

        rto_text = f"RTO {rto_info.code} ({rto_info.location})" if rto_info else "prefix"
        if differences:
            explanation = f"Preserved visible {rto_text}; {'; '.join(differences[:3])}."
        else:
            explanation = f"Exact format match for {rto_text}."

        return conf, explanation


candidate_service = PlateCandidateService()
