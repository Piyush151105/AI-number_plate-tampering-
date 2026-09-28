from dataclasses import dataclass, field


@dataclass
class PlateRegion:
    x: int
    y: int
    width: int
    height: int
    confidence: float

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.x + self.width, self.y + self.height)


@dataclass
class CharConfidence:
    char: str
    confidence: float
    position: int
    status: str = "CONFIRMED"  # CONFIRMED, AMBIGUOUS, LOW_CONFIDENCE, OCCLUDED, MISSING
    alternatives: list[str] = field(default_factory=list)


@dataclass
class OCRResult:
    text: str
    confidence: float
    is_valid_format: bool
    format_message: str
    raw_text: str = ""
    char_confidences: list[CharConfidence] = field(default_factory=list)


@dataclass
class TamperSignal:
    name: str
    score: float
    description: str
    severity: str


@dataclass
class TamperAnalysis:
    tamper_score: float
    is_tampered: bool
    verdict: str
    signals: list[TamperSignal] = field(default_factory=list)


@dataclass
class RTOResult:
    prefix: str
    code: str
    state_code: str
    state_name: str
    location: str
    district: str
    is_known: bool = True
    region_zone: str = ""


@dataclass
class TamperAlert:
    indicator: str  # Occlusion, Broken Characters, Blur, Altered Font/Spacing, Low OCR Confidence
    score: float
    severity: str  # high, medium, low
    description: str
    detected: bool = True


@dataclass
class RTOAreaConstraint:
    rto_code: str
    prefix: str
    state_code: str
    state_name: str
    location: str
    district: str
    region_zone: str = ""
    allowed_series: list[str] = field(default_factory=list)
    prohibited_letters: list[str] = field(default_factory=lambda: ["I", "O", "Q"])
    common_series_by_class: dict[str, list[str]] = field(default_factory=dict)
    standard_format: str = "SS RR CC NNNN"


@dataclass
class PredictedCandidate:
    complete_plate: str
    raw_plate: str
    match_confidence: float
    match_confidence_pct: str
    predicted_chars: dict[int, str]
    vehicle_class: str
    rto_area: str
    validation_status: str
    explanation: str
    is_verified_truth: bool = False
    vehicle_details: dict[str, str] = field(default_factory=dict)


@dataclass
class PlatePredictionResult:
    input_pattern: str
    normalized_pattern: str
    state_code: str | None
    rto_code: str | None
    rto_area_info: RTOAreaConstraint | None
    candidates: list[PredictedCandidate] = field(default_factory=list)
    total_candidates_found: int = 0
    validation_notes: list[str] = field(default_factory=list)
    status: str = "SUCCESS"
    error_message: str | None = None


@dataclass
class PlateCandidate:
    plate_candidate: str
    label: str = "Unverified candidate"
    confidence: float = 0.0
    explanation: str = ""


@dataclass
class CandidateAnalysisResult:
    original_ocr_text: str
    clean_text: str
    char_confidences: list[CharConfidence] = field(default_factory=list)
    detected_rto: RTOResult | None = None
    tampering_alerts: list[TamperAlert] = field(default_factory=list)
    candidates: list[PlateCandidate] = field(default_factory=list)
    candidate_status: str = "INSUFFICIENT_CHARACTERS"
    readable_character_count: int = 0
    warning_message: str = (
        "Manual RTO verification required. Do not use for automatic challans, "
        "enforcement decisions, or identity matching. Human review and official "
        "RTO database verification is mandatory."
    )
    disclaimer: str = (
        "Every candidate is strictly an Unverified Candidate. Guesses are never "
        "confirmed registration numbers."
    )


@dataclass
class DetectionResult:
    plate_region: PlateRegion | None
    ocr: OCRResult | None
    tampering: TamperAnalysis
    candidate_analysis: CandidateAnalysisResult | None = None
    annotated_image_path: str | None = None
    plate_crop_path: str | None = None
    processing_ms: int = 0
