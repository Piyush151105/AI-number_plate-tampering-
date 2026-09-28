"""FastAPI backend for plate tampering detection and e-Challan verification."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import cv2
from pydantic import BaseModel, Field

from config.settings import settings
from ml.challan import (
    ChallanCheckRequest,
    ChallanCheckResponse,
    ChallanErrorResponse,
    challan_service as challan_service_provider,
)
from ml.challan_service import challan_service
from ml.pipeline import PlateTamperingPipeline
from ml.rto_predictor import rto_predictor
from ml.rto_service import rto_service
from ml.schemas import PlatePredictionResult

app = FastAPI(
    title=settings.project_name,
    version="1.5.0",
    description="AI-powered vehicle number plate tampering detection, RTO missing character predictor, and e-Challan API",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = PlateTamperingPipeline()


class SignalResponse(BaseModel):
    name: str
    score: float
    description: str
    severity: str


class CharConfidenceResponse(BaseModel):
    char: str
    confidence: float
    position: int
    status: str
    alternatives: list[str] = []


class RTOResponse(BaseModel):
    prefix: str
    code: str
    state_code: str
    state_name: str
    location: str
    district: str
    is_known: bool
    region_zone: str = ""


class TamperAlertResponse(BaseModel):
    indicator: str
    score: float
    severity: str
    description: str
    detected: bool


class PlatePredictionRequest(BaseModel):
    partial_plate: str = Field(..., description="Partial registration string with wildcards e.g. 'MH 46 B* 161*' or 'MH 46 ** 1612'")
    top_k: int = Field(default=10, ge=1, le=50)
    target_category: str | None = Field(default=None, description="Optional vehicle category filter (Four-Wheeler, Two-Wheeler, Commercial)")


class PredictedCandidateResponse(BaseModel):
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
    vehicle_details: dict[str, str] = {}


class RTOAreaConstraintResponse(BaseModel):
    rto_code: str
    prefix: str
    state_code: str
    state_name: str
    location: str
    district: str
    region_zone: str = ""
    allowed_series: list[str] = []
    prohibited_letters: list[str] = ["I", "O", "Q"]
    standard_format: str = "SS RR CC NNNN"


class PlatePredictionResponse(BaseModel):
    input_pattern: str
    normalized_pattern: str
    state_code: str | None = None
    rto_code: str | None = None
    rto_area_info: RTOAreaConstraintResponse | None = None
    candidates: list[PredictedCandidateResponse] = []
    total_candidates_found: int = 0
    validation_notes: list[str] = []
    status: str = "SUCCESS"
    error_message: str | None = None


class ChallanRecordResponse(BaseModel):
    challan_no: str
    date: str
    offense: str
    fine_amount: float
    status: str
    location: str
    traffic_unit: str


class ChallanSummaryResponse(BaseModel):
    plate_number: str
    total_challans: int
    unpaid_challans: int
    paid_challans: int = 0
    total_due: float
    rto_office: str
    state: str
    vehicle_class: str
    source: str = "Official MahaTraffic / Parivahan Verified Records"
    records: list[ChallanRecordResponse]


class DetectionResponse(BaseModel):
    plate_text: str
    ocr_confidence: float
    is_valid_format: bool
    format_message: str
    tamper_score: float
    is_tampered: bool
    verdict: str
    signals: list[SignalResponse]
    annotated_image_url: str | None
    plate_crop_url: str | None
    processing_ms: int
    challan: ChallanSummaryResponse | None = None


def serialize_prediction_result(res: PlatePredictionResult) -> PlatePredictionResponse:
    rto_resp = None
    if res.rto_area_info:
        rto_resp = RTOAreaConstraintResponse(
            rto_code=res.rto_area_info.rto_code,
            prefix=res.rto_area_info.prefix,
            state_code=res.rto_area_info.state_code,
            state_name=res.rto_area_info.state_name,
            location=res.rto_area_info.location,
            district=res.rto_area_info.district,
            region_zone=res.rto_area_info.region_zone,
            allowed_series=res.rto_area_info.allowed_series,
            prohibited_letters=res.rto_area_info.prohibited_letters,
            standard_format=res.rto_area_info.standard_format,
        )
    return PlatePredictionResponse(
        input_pattern=res.input_pattern,
        normalized_pattern=res.normalized_pattern,
        state_code=res.state_code,
        rto_code=res.rto_code,
        rto_area_info=rto_resp,
        candidates=[
            PredictedCandidateResponse(
                complete_plate=c.complete_plate,
                raw_plate=c.raw_plate,
                match_confidence=c.match_confidence,
                match_confidence_pct=c.match_confidence_pct,
                predicted_chars=c.predicted_chars,
                vehicle_class=c.vehicle_class,
                rto_area=c.rto_area,
                validation_status=c.validation_status,
                explanation=c.explanation,
                is_verified_truth=c.is_verified_truth,
                vehicle_details=c.vehicle_details,
            )
            for c in res.candidates
        ],
        total_candidates_found=res.total_candidates_found,
        validation_notes=res.validation_notes,
        status=res.status,
        error_message=res.error_message,
    )


@app.get("/")
def root() -> dict[str, str]:
    return {
        "project": settings.project_name,
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": "1.3.0", "engine": "balanced-forensics-v3"}


@app.post(
    "/api/challan/check",
    response_model=ChallanCheckResponse,
    responses={
        400: {"model": ChallanErrorResponse},
        429: {"model": ChallanErrorResponse},
        502: {"model": ChallanErrorResponse},
        504: {"model": ChallanErrorResponse},
    },
)
async def check_challan(request: ChallanCheckRequest, req: Request):
    """
    Standardized e-Challan check endpoint.
    Queries the decoupled provider (Mock or Authorized) with safe error shielding.
    """
    client_ip = req.client.host if req.client else "127.0.0.1"
    success_resp, err_resp, status_code = await challan_service_provider.check(
        vehicle_number=request.vehicleNumber,
        client_id=client_ip,
    )
    if err_resp is not None:
        return JSONResponse(status_code=status_code, content=err_resp.model_dump())
    return success_resp


@app.get("/api/challan/{plate_number}", response_model=ChallanSummaryResponse)
def get_challan(
    plate_number: str,
    is_tampered: bool = False,
    tamper_verdict: str = "",
) -> ChallanSummaryResponse:
    summary = challan_service.lookup(
        plate_number=plate_number,
        is_tampered=is_tampered,
        tamper_verdict=tamper_verdict,
    )
    return ChallanSummaryResponse(
        plate_number=summary.plate_number,
        total_challans=summary.total_challans,
        unpaid_challans=summary.unpaid_challans,
        paid_challans=summary.paid_challans,
        total_due=summary.total_due,
        rto_office=summary.rto_office,
        state=summary.state,
        vehicle_class=summary.vehicle_class,
        source=summary.source,
        records=[
            ChallanRecordResponse(
                challan_no=r.challan_no,
                date=r.date,
                offense=r.offense,
                fine_amount=r.fine_amount,
                status=r.status,
                location=r.location,
                traffic_unit=r.traffic_unit,
            )
            for r in summary.records
        ],
    )


class PaymentRequest(BaseModel):
    plate_number: str
    challan_nos: list[str] = []


@app.post("/api/challan/pay")
def pay_challan(payload: PaymentRequest) -> dict:
    challan_service.pay_challans(payload.plate_number, payload.challan_nos)
    summary = challan_service.lookup(payload.plate_number)
    return {
        "status": "success",
        "message": f"Challans for {payload.plate_number} settled.",
        "unpaid_challans": summary.unpaid_challans,
        "total_due": summary.total_due,
    }


@app.post("/api/challan/reset")
def reset_challan(payload: PaymentRequest) -> dict:
    challan_service.reset_challans(payload.plate_number)
    summary = challan_service.lookup(payload.plate_number)
    return {
        "status": "success",
        "message": f"Challans for {payload.plate_number} reset to unpaid.",
        "unpaid_challans": summary.unpaid_challans,
        "total_due": summary.total_due,
    }


@app.post("/api/detect", response_model=DetectionResponse)
async def detect_tampering(file: UploadFile = File(...)) -> DetectionResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
    upload_path = settings.upload_dir / f"upload_{Path(file.filename or 'image').stem}{suffix}"
    upload_path.write_bytes(await file.read())

    try:
        result = pipeline.run(upload_path)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    ocr = result.ocr
    tamper = result.tampering
    plate_text = ocr.text if ocr else ""

    # Generate e-Challan summary for the detected plate
    challan_summary = None
    if plate_text:
        summary_obj = challan_service.lookup(
            plate_number=plate_text,
            is_tampered=tamper.is_tampered,
            tamper_verdict=tamper.verdict,
        )
        challan_summary = ChallanSummaryResponse(
            plate_number=summary_obj.plate_number,
            total_challans=summary_obj.total_challans,
            unpaid_challans=summary_obj.unpaid_challans,
            paid_challans=summary_obj.paid_challans,
            total_due=summary_obj.total_due,
            rto_office=summary_obj.rto_office,
            state=summary_obj.state,
            vehicle_class=summary_obj.vehicle_class,
            source=summary_obj.source,
            records=[
                ChallanRecordResponse(
                    challan_no=r.challan_no,
                    date=r.date,
                    offense=r.offense,
                    fine_amount=r.fine_amount,
                    status=r.status,
                    location=r.location,
                    traffic_unit=r.traffic_unit,
                )
                for r in summary_obj.records
            ],
        )

    return DetectionResponse(
        plate_text=plate_text,
        ocr_confidence=round(ocr.confidence, 4) if ocr else 0.0,
        is_valid_format=ocr.is_valid_format if ocr else False,
        format_message=ocr.format_message if ocr else "",
        tamper_score=tamper.tamper_score,
        is_tampered=tamper.is_tampered,
        verdict=tamper.verdict,
        signals=[
            SignalResponse(
                name=signal.name,
                score=round(signal.score, 4),
                description=signal.description,
                severity=signal.severity,
            )
            for signal in tamper.signals
        ],
        annotated_image_url=(
            f"/api/outputs/{Path(result.annotated_image_path).name}"
            if result.annotated_image_path
            else None
        ),
        plate_crop_url=(
            f"/api/outputs/{Path(result.plate_crop_path).name}"
            if result.plate_crop_path
            else None
        ),
        processing_ms=result.processing_ms,
        challan=challan_summary,
    )


@app.post("/api/rto-predictor/predict", response_model=PlatePredictionResponse)
def predict_missing_plate_characters(payload: PlatePredictionRequest) -> PlatePredictionResponse:
    """
    Predict missing characters in a partial Indian vehicle registration number using
    RTO geography mapping, series constraints, and probabilistic ML matching.
    """
    result = rto_predictor.predict(
        partial_plate=payload.partial_plate,
        top_k=payload.top_k,
        target_category=payload.target_category,
    )
    return serialize_prediction_result(result)


@app.get("/api/rto-predictor/rto-info/{rto_code}", response_model=RTOAreaConstraintResponse)
def get_rto_constraint_info(rto_code: str) -> RTOAreaConstraintResponse:
    """Get local RTO area details and vehicle series constraints."""
    constraint = rto_predictor.get_rto_constraint(rto_code)
    if not constraint:
        raise HTTPException(status_code=404, detail=f"RTO constraint for '{rto_code}' not found.")
    return RTOAreaConstraintResponse(
        rto_code=constraint.rto_code,
        prefix=constraint.prefix,
        state_code=constraint.state_code,
        state_name=constraint.state_name,
        location=constraint.location,
        district=constraint.district,
        region_zone=constraint.region_zone,
        allowed_series=constraint.allowed_series,
        prohibited_letters=constraint.prohibited_letters,
        standard_format=constraint.standard_format,
    )


@app.get("/api/rto/{prefix_or_code}", response_model=RTOResponse)
def get_rto_details(prefix_or_code: str) -> RTOResponse:
    """Lookup configurable RTO location details (e.g. MH-46 -> Panvel, MH-08 -> Ratnagiri)."""
    res = rto_service.lookup(prefix_or_code)
    if not res:
        raise HTTPException(status_code=404, detail=f"RTO code/prefix '{prefix_or_code}' not found.")
    return RTOResponse(
        prefix=res.prefix,
        code=res.code,
        state_code=res.state_code,
        state_name=res.state_name,
        location=res.location,
        district=res.district,
        is_known=res.is_known,
        region_zone=res.region_zone,
    )


@app.get("/api/outputs/{filename}")
def get_output_image(filename: str) -> FileResponse:
    file_path = settings.output_dir / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Output image not found.")
    return FileResponse(file_path)
