# AI-Enabled Vehicle Number Plate Tampering Detection

**Final Year Project** — Computer Vision, OCR, and Deep Learning for forensic plate analysis.

## Problem Statement

Vehicle number plate tampering (character replacement, paint alteration, sticker overlays, forged plates) is used to evade traffic enforcement and conceal vehicle identity. Manual inspection is slow and inconsistent. This project automates detection using AI.

## Objectives

1. Localize number plates in vehicle images
2. Extract plate text using OCR
3. Validate Indian plate format
4. Detect tampering via multi-signal forensic analysis + CNN classifier
5. Present results through API and web dashboard

## System Architecture

```
Vehicle Image
     │
     ▼
┌─────────────────┐
│ Plate Detector  │  OpenCV edge + contour analysis
└────────┬────────┘
         ▼
┌─────────────────┐
│ OCR Engine      │  EasyOCR + format validation
└────────┬────────┘
         ▼
┌─────────────────┐
│ Tamper Analyzer │  Heuristics + TamperCNN (PyTorch)
└────────┬────────┘
         ▼
┌─────────────────┐
│ API + Dashboard │  FastAPI + Streamlit
└─────────────────┘
```

## Tech Stack

| Layer | Technology |
|-------|------------|
| Plate detection | OpenCV |
| OCR | EasyOCR |
| Tampering ML | PyTorch CNN + forensic heuristics |
| Backend API | FastAPI |
| Frontend | Streamlit |
| Language | Python 3.10+ |

## Project Structure

```
ai-plate-tampering-detection/
├── backend/main.py           # REST API
├── frontend/app.py           # Streamlit dashboard
├── ml/
│   ├── plate_detector.py     # Plate localization
│   ├── ocr_engine.py         # OCR + validation
│   ├── tampering_detector.py # Forensic scoring
│   ├── pipeline.py           # End-to-end orchestration
│   └── models/tamper_cnn.py  # CNN architecture
├── training/train_tamper_model.py
├── scripts/generate_samples.py
├── docs/                     # Report-ready documentation
├── tests/
├── run_api.py
└── run_dashboard.py
```

## Quick Start

### 1. Create virtual environment

```powershell
cd /d "d:\MAJOR PROJECT\ai-plate-tampering-detection"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Generate sample data and train model

```powershell
python scripts/generate_samples.py
python training/train_tamper_model.py
```

### 3. Run the system

Terminal 1 — API:
```powershell
python run_api.py
```

Terminal 2 — Dashboard:
```powershell
python run_dashboard.py
```

Open http://localhost:8501 and upload a vehicle image.

## API Usage

```bash
curl -X POST "http://127.0.0.1:8000/api/detect" -F "file=@data/samples/tampered/tampered_000.jpg"
```

## e-Challan Verification Feature

The system integrates an automated e-Challan verification module that links directly with the OCR-extracted vehicle registration number.

### Architecture

```
Existing AI Plate Detection
           ↓
OCR / Vehicle Number (e.g. MH46BW1612)
           ↓
User Edit / Confirmation [ MH46BW1612 ]
           ↓
[ Check e-Challan ] Button
           ↓
Backend Endpoint (POST /api/challan/check)
           ↓
Challan Service Orchestrator
           ↓
Challan Provider Adapter
     ├── MockChallanProvider (Default, with realistic test records)
     └── AuthorizedChallanProvider (Official REST API gateway)
```

### API Endpoint

`POST /api/challan/check`

**Request:**
```json
{
  "vehicleNumber": "MH46BW1612"
}
```

**Response (Mock Provider):**
```json
{
  "success": true,
  "vehicleNumber": "MH46BW1612",
  "totalPending": 2,
  "totalDue": 2000.0,
  "challans": [
    {
      "challanNumber": "SATCO25SKFJ68C32",
      "date": "2025-06-08 12:32",
      "offence": "Section 129 / 194D MV Act — Riding Two-Wheeler without Protective Helmet",
      "location": "Satara / NH-48 Highway Division, Maharashtra",
      "amount": 500.0,
      "status": "Pending",
      "dueAmount": 500.0
    },
    {
      "challanNumber": "SATCM24000267081",
      "date": "2024-01-10 18:01",
      "offence": "Section 119 / 177 MV Act — Dangerous Driving / Traffic Signal Non-Compliance",
      "location": "Satara City Traffic Division, Maharashtra",
      "amount": 1500.0,
      "status": "Pending",
      "dueAmount": 1500.0
    }
  ],
  "provider": "mock",
  "disclaimer": "Demo data. Connect an authorized e-Challan provider for live results."
}
```

**Error Response:**
```json
{
  "success": false,
  "error": {
    "code": "INVALID_VEHICLE_NUMBER",
    "message": "Vehicle number 'INVALID' is invalid. Expected standard Indian format (e.g. MH12AB1234)."
  }
}
```

### Environment Configuration

Configure the provider in `.env` or `config/settings.py`:

```ini
# e-Challan Provider: "mock" (default) or "authorized"
CHALLAN_PROVIDER=mock

# Required only when CHALLAN_PROVIDER=authorized
CHALLAN_API_BASE_URL=https://api.your-authorized-gateway.com/v1/challans
CHALLAN_API_KEY=your_secret_api_key_here
CHALLAN_TIMEOUT_SEC=8
CHALLAN_RATE_LIMIT_PER_MIN=20
```

### Security & Compliance Rules

* **Server-side only:** API keys are never exposed to the frontend client.
* **No scraping or bypass:** The system strictly forbids scraping private government portals, bypassing CAPTCHA/OTP, or reverse-engineering mobile app tokens.
* **Authorized integration:** Live challan queries strictly require an authorized commercial or official transport department API gateway. When in mock mode, results are explicitly labeled with the demo disclaimer.

## Tampering Detection Methodology

The system combines **multi-signal forensic indicators**:

| Indicator | What it detects |
|-----------|-----------------|
| Occlusion | Opaque patches, stickers, tape, or paint masks covering characters |
| Broken Characters | Fragmented strokes, scratched paint, or split contours |
| Blur / Defocus | Laplacian variance drop & selective smearing masking glyphs |
| Altered Font / Spacing | Irregular character kerning, non-standard widths, and baseline drift |
| Low OCR Confidence | Drop in character confidence or placeholder tokens (`*`, `?`) |
| CNN Classifier | Deep learning model distinguishing authentic vs altered plates |

Final score = weighted combination. Threshold default: **0.52**.

## RTO-Based Missing Character Predictor Feature

The missing character predictor reconstructs complete Indian vehicle registration numbers from damaged, occluded, or tampered plates containing wildcards (`*` or `?`), leveraging local RTO geographic constraints, vehicle series registries, and probabilistic ML matching.

### Architecture

```
Partial Vehicle Number with Wildcards (e.g. MH 46 B* 161* or MH 46 ** 1612)
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│ Partial Plate Parser & Normalizer                           │  Aligns State, RTO, Series, Number
└──────────────────────────────┬──────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│ RTO Geography & Vehicle Series Constraints Mapping          │  Identifies State & RTO jurisdiction
│ (e.g. MH-46 -> Panvel, Raigad, Maharashtra)                │  Extracts local active series & vehicle classes
└──────────────────────────────┬──────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│ ML Prediction Engine & Syntax Validator                     │  Enforces MoRTH rules (excludes I, O, Q)
│ (ml/rto_predictor.py)                                       │  Generates valid plate candidate hypotheses
└──────────────────────────────┬──────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│ Candidate Ranking & Match Confidence Scoring                │  Calculates Match Confidence Score (%)
│ (Results Table + Local Area Details + Explanations)         │  Ranks top candidates for investigative review
└─────────────────────────────────────────────────────────────┘
```

### Key Capabilities

> [!NOTE]
> - **Wildcard Flexibility:** Accepts inputs such as `MH 46 B* 161*`, `MH 46 ** 1612`, `MH 08 A* 1400`, `DL 01 C* 5678`, etc.
> - **RTO Series Constraints:** Validates candidate series against active series registries for that specific local RTO area (e.g., Panvel `BW`, `CA`, `BA`).
> - **MoRTH Compliance:** Strictly enforces exclusion of confusing letters (`I`, `O`, `Q`).
> - **Match Confidence Score:** Probabilistic score incorporating geographic match, local series adherence, syntactic validity, and wildcard uncertainty.

### API Endpoints

- `POST /api/rto-predictor/predict` — JSON payload (`{"partial_plate": "MH 46 B* 161*", "top_k": 10}`).
- `GET /api/rto-predictor/rto-info/{rto_code}` — Query local RTO area details and series constraints (e.g. `MH-46`, `MH-08`).
- `GET /api/rto/{code}` — Query general RTO prefix dataset.

## Future Enhancements

- YOLOv8 plate detector for real-world datasets
- Integration with CCTV / RTSP streams
- Blockchain-based audit trail for law enforcement
- Mobile app for field officers

## Team & Academic Use

Suitable for final year B.Tech/BCA/MCA project submission. See `docs/` for abstract, methodology, and architecture write-ups.

## License

MIT — for academic and research use.
