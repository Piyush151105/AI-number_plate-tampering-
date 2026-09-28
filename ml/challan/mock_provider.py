"""Mock e-Challan provider supplying authentic test records and deterministic simulation."""

from __future__ import annotations

import hashlib
from ml.challan.base import ChallanProvider
from ml.challan.schemas import ChallanCheckResponse, ChallanItem

# Known test vehicle records
MOCK_DATABASE: dict[str, list[dict]] = {
    "MH46BW1612": [
        {
            "challanNumber": "SATCO25SKFJ68C32",
            "date": "2025-06-08 12:32",
            "offence": "Section 129 / 194D MV Act — Riding Two-Wheeler without Protective Helmet",
            "location": "Satara / NH-48 Highway Division, Maharashtra",
            "amount": 500.0,
            "status": "Pending",
            "dueAmount": 500.0,
        },
        {
            "challanNumber": "SATCM24000267081",
            "date": "2024-01-10 18:01",
            "offence": "Section 119 / 177 MV Act — Dangerous Driving / Traffic Signal Non-Compliance",
            "location": "Satara City Traffic Division, Maharashtra",
            "amount": 1500.0,
            "status": "Pending",
            "dueAmount": 1500.0,
        },
    ],
    "DL2SKA2187": [
        {
            "challanNumber": "DL012511050211",
            "date": "2025-11-05 14:15",
            "offence": "Section 112 / 183 MV Act — Exceeding Prescribed Speed Limit (Over-Speeding)",
            "location": "Outer Ring Road, Near Majnu Ka Tilla, Delhi",
            "amount": 2000.0,
            "status": "Pending",
            "dueAmount": 2000.0,
        }
    ],
    "MH08AX1400": [
        {
            "challanNumber": "RATCM2500019284",
            "date": "2025-05-14 11:20",
            "offence": "Section 177 MV Act — Parking in No Parking Zone / Obstructing Highway Traffic",
            "location": "Ratnagiri City Traffic Division, Maharashtra",
            "amount": 500.0,
            "status": "Pending",
            "dueAmount": 500.0,
        }
    ],
    "KA01AB1234": [],  # Clean vehicle record
}


class MockChallanProvider(ChallanProvider):
    """Mock provider with realistic records and explicit demonstration tags."""

    @property
    def provider_name(self) -> str:
        return "mock"

    async def check_challans(self, vehicle_number: str) -> ChallanCheckResponse:
        v_num = vehicle_number.upper().strip()

        if v_num in MOCK_DATABASE:
            raw_items = MOCK_DATABASE[v_num]
        else:
            # Deterministic simulation for arbitrary plate numbers
            plate_hash = int(hashlib.md5(v_num.encode()).hexdigest(), 16)
            has_challan = (plate_hash % 2 != 0)
            if has_challan:
                ref = str(plate_hash)[:8]
                raw_items = [
                    {
                        "challanNumber": f"{v_num[:2]}TRF2026{ref}",
                        "date": "2026-02-22 15:10",
                        "offence": "Section 119 / 177 MV Act — Traffic Signal Non-Compliance",
                        "location": f"{v_num[:2]} Traffic Division Corridor",
                        "amount": 1000.0,
                        "status": "Pending",
                        "dueAmount": 1000.0,
                    }
                ]
            else:
                raw_items = []

        challans = [ChallanItem(**item) for item in raw_items]
        pending_items = [c for c in challans if c.status.lower() in ("pending", "unpaid")]
        total_due = sum(c.dueAmount for c in pending_items)

        return ChallanCheckResponse(
            success=True,
            vehicleNumber=v_num,
            totalPending=len(pending_items),
            totalDue=total_due,
            challans=challans,
            provider=self.provider_name,
            disclaimer="These challan results are sample data and are not real vehicle records.",
        )
