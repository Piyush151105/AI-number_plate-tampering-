"""e-Challan Service for Indian Vehicles.

Provides live third-party e-Challan lookups with resilient fallback
to deterministic demo data, request caching, and MoRTH plate syntax validation.
"""

from __future__ import annotations

import hashlib
import os
import re
import time
from typing import Any

from dotenv import load_dotenv
import requests
import streamlit as st

# Regular expressions for Indian vehicle registration marks
# Standard format: State(2) + RTO(1-2) + Series(0-3) + Number(4)
INDIAN_STANDARD_PLATE_REGEX = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{4}$")
# Bharat Series (BH): Year(2) + BH + Number(4) + Series(1-2) (e.g., 22BH1234AA)
BHARAT_SERIES_REGEX = re.compile(r"^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$")

# Authentic test vehicle records for demonstration fallback
DEMO_DATABASE: dict[str, list[dict[str, Any]]] = {
    "MH46BW1612": [
        {
            "challan_no": "SATCO25SKFJ68C32",
            "date": "2025-06-08 12:32",
            "offence": "Section 129 / 194D MV Act — Riding Two-Wheeler without Protective Helmet",
            "location": "Satara / NH-48 Highway Division, Maharashtra",
            "amount": 500.0,
            "due_amount": 500.0,
            "status": "Pending",
        },
        {
            "challan_no": "SATCM24000267081",
            "date": "2024-01-10 18:01",
            "offence": "Section 119 / 177 MV Act — Dangerous Driving / Traffic Signal Non-Compliance",
            "location": "Satara City Traffic Division, Maharashtra",
            "amount": 1500.0,
            "due_amount": 1500.0,
            "status": "Pending",
        },
    ],
    "DL2SKA2187": [
        {
            "challan_no": "DL012511050211",
            "date": "2025-11-05 14:15",
            "offence": "Section 112 / 183 MV Act — Exceeding Prescribed Speed Limit (Over-Speeding)",
            "location": "Outer Ring Road, Near Majnu Ka Tilla, Delhi",
            "amount": 2000.0,
            "due_amount": 2000.0,
            "status": "Pending",
        }
    ],
    "MH08AX1400": [
        {
            "challan_no": "RATCM2500019284",
            "date": "2025-05-14 11:20",
            "offence": "Section 177 MV Act — Parking in No Parking Zone / Obstructing Traffic",
            "location": "Ratnagiri City Traffic Division, Maharashtra",
            "amount": 500.0,
            "due_amount": 500.0,
            "status": "Pending",
        }
    ],
    "KA01AB1234": [],  # Clean vehicle record with zero pending violations
}


def validate_vehicle_number(plate: str) -> str:
    """Validate and normalize Indian vehicle registration mark.

    Normalizes by stripping whitespaces and hyphens and converting to uppercase.
    Validates against MoRTH Indian standard plate pattern and Bharat Series (BH).

    Args:
        plate: Raw registration string (e.g. 'mh-46 bw 1612' or '22BH1234AA')

    Returns:
        Cleaned, uppercase registration string.

    Raises:
        ValueError: If string does not conform to valid Indian registration pattern.
    """
    if not plate or not isinstance(plate, str):
        raise ValueError("Vehicle registration number cannot be empty.")

    clean = re.sub(r"[\s\-]+", "", plate.upper().strip())

    if not clean:
        raise ValueError("Vehicle registration number cannot be empty.")

    if not (INDIAN_STANDARD_PLATE_REGEX.match(clean) or BHARAT_SERIES_REGEX.match(clean)):
        raise ValueError(
            f"Invalid Indian vehicle registration number format: '{plate}'. "
            "Expected format like 'MH12AB1234' or Bharat Series like '22BH1234AA'."
        )

    return clean


def fetch_challans_live(vehicle_no: str) -> dict[str, Any]:
    """Fetch live traffic citations from third-party e-Challan API.

    Keeps the request/response mapping isolated to allow easily switching providers.
    Uses a 10-second timeout, retries once on transient errors, and strictly avoids
    logging or exposing sensitive API keys.

    Args:
        vehicle_no: Normalized vehicle registration number.

    Returns:
        dict: Normalized intermediate dict containing 'total_pending', 'total_due',
              and 'challans'.

    Raises:
        ValueError: If required API credentials are missing from environment.
        RuntimeError: If the remote API call fails.
    """
    load_dotenv()
    api_url = os.getenv("CHALLAN_API_URL", "").strip()
    api_key = os.getenv("CHALLAN_API_KEY", "").strip()

    if not api_url or not api_key:
        raise ValueError("CHALLAN_API_URL and CHALLAN_API_KEY must be configured in environment.")

    # ==============================================================================
    # TODO: CONFIGURE YOUR THIRD-PARTY CHALLAN PROVIDER DETAILS BELOW
    #
    # 1. ENDPOINT URL & PARAMETERS:
    #    Adjust parameter names according to your provider documentation
    #    (e.g., 'vehicle_number', 'plate_number', 'rc_number', 'registration_no').
    #
    # 2. AUTHENTICATION HEADERS:
    #    Adjust header name if your provider expects a different scheme:
    #    - Bearer token: {"Authorization": f"Bearer {api_key}"}
    #    - API Key header: {"x-api-key": api_key} or {"apikey": api_key}
    #
    # 3. RESPONSE FIELD MAPPING:
    #    Adjust the JSON dictionary keys below to match your vendor's response schema.
    # ==============================================================================

    # TODO: Replace or adjust headers as required by your provider's docs
    headers = {
        "Authorization": f"Bearer {api_key}",
        "x-api-key": api_key,
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    # TODO: Replace parameter name ('vehicle_number') if your provider uses a different key
    params = {
        "vehicle_number": vehicle_no,
    }

    max_attempts = 2
    raw_data: Any = None
    last_err: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(
                api_url,
                headers=headers,
                params=params,
                timeout=10,
            )

            # Retry once on transient server errors (502 Bad Gateway, 503 Service Unavailable, 504 Gateway Timeout)
            if response.status_code in (502, 503, 504) and attempt < max_attempts:
                time.sleep(1)
                continue

            response.raise_for_status()
            raw_data = response.json()
            break
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            last_err = exc
            if attempt < max_attempts:
                time.sleep(1)
                continue
            raise RuntimeError(f"Network connection error ({type(exc).__name__})") from exc
        except requests.exceptions.HTTPError as exc:
            # Avoid logging query params or headers that might contain tokens
            raise RuntimeError(f"HTTP Error {response.status_code}") from exc
        except Exception as exc:
            raise RuntimeError(f"API request failed: {exc}") from exc
    else:
        raise RuntimeError(f"Failed to fetch live challans after {max_attempts} attempts: {last_err}")

    # ==============================================================================
    # TODO: MAP PROVIDER JSON RESPONSE TO NORMALIZED STRUCTURE
    # Update field names based on your vendor's JSON output
    # ==============================================================================
    raw_items: list[dict[str, Any]] = []
    if isinstance(raw_data, dict):
        raw_items = (
            raw_data.get("challans")
            or raw_data.get("data")
            or raw_data.get("records")
            or raw_data.get("result")
            or []
        )
    elif isinstance(raw_data, list):
        raw_items = raw_data

    challans_list: list[dict[str, Any]] = []
    for item in raw_items:
        amt = float(item.get("amount") or item.get("fine_amount") or item.get("total_amount") or 0.0)
        due = float(item.get("due_amount") or item.get("pending_amount") or amt)
        status_val = str(item.get("status") or ("Pending" if due > 0 else "Paid")).title()

        challans_list.append(
            {
                "challan_no": str(item.get("challan_no") or item.get("challan_number") or item.get("challanNo") or "N/A"),
                "date": str(item.get("date") or item.get("challan_date") or item.get("violation_date") or "N/A"),
                "offence": str(item.get("offence") or item.get("violation") or item.get("reason") or "Traffic Violation"),
                "location": str(item.get("location") or item.get("place") or "Traffic Division"),
                "amount": amt,
                "due_amount": due,
                "status": status_val,
            }
        )

    pending_items = [c for c in challans_list if c["status"].lower() in ("pending", "unpaid")]
    total_due = sum(c["due_amount"] for c in pending_items)

    return {
        "total_pending": len(pending_items),
        "total_due": total_due,
        "challans": challans_list,
    }


def fetch_challans_demo(vehicle_no: str) -> dict[str, Any]:
    """Generate realistic demonstration e-Challan records.

    Uses known authentic records for test plates and deterministic hashing
    simulation for arbitrary valid plate numbers.

    Args:
        vehicle_no: Cleaned vehicle registration number.

    Returns:
        dict: Normalized intermediate dict with 'total_pending', 'total_due',
              and 'challans'.
    """
    v_num = vehicle_no.upper()

    if v_num in DEMO_DATABASE:
        raw_items = DEMO_DATABASE[v_num]
    else:
        # Deterministic simulation for arbitrary valid plates
        plate_hash = int(hashlib.md5(v_num.encode()).hexdigest(), 16)
        has_challan = (plate_hash % 2 != 0)
        if has_challan:
            ref = str(plate_hash)[:8].upper()
            state_prefix = v_num[:2]
            raw_items = [
                {
                    "challan_no": f"{state_prefix}TRF2026{ref}",
                    "date": "2026-02-22 15:10",
                    "offence": "Section 119 / 177 MV Act — Dangerous Driving / Traffic Signal Violation",
                    "location": f"{state_prefix} Traffic Enforcement Corridor",
                    "amount": 1000.0,
                    "due_amount": 1000.0,
                    "status": "Pending",
                }
            ]
        else:
            raw_items = []

    pending_items = [c for c in raw_items if c["status"].lower() in ("pending", "unpaid")]
    total_due = sum(c["due_amount"] for c in pending_items)

    return {
        "total_pending": len(pending_items),
        "total_due": total_due,
        "challans": [dict(c) for c in raw_items],
    }


def _execute_challan_lookup(vehicle_no: str) -> dict[str, Any]:
    """Internal implementation of get_challans without caching layer."""
    cleaned_plate = validate_vehicle_number(vehicle_no)

    load_dotenv()
    use_live = os.getenv("USE_LIVE_API", "").strip().lower() in ("true", "1", "yes")
    api_key = os.getenv("CHALLAN_API_KEY", "").strip()

    warning_msg: str | None = None

    if use_live and api_key:
        try:
            live_res = fetch_challans_live(cleaned_plate)
            return {
                "source": "live",
                "vehicle": cleaned_plate,
                "total_pending": live_res.get("total_pending", 0),
                "total_due": float(live_res.get("total_due", 0.0)),
                "challans": live_res.get("challans", []),
                "warning": None,
            }
        except Exception as exc:
            warning_msg = f"Live e-Challan lookup failed ({exc}). Fell back to demo mode."

    # Demo mode fallback or default
    demo_res = fetch_challans_demo(cleaned_plate)
    return {
        "source": "demo",
        "vehicle": cleaned_plate,
        "total_pending": demo_res.get("total_pending", 0),
        "total_due": float(demo_res.get("total_due", 0.0)),
        "challans": demo_res.get("challans", []),
        "warning": warning_msg,
    }


@st.cache_data(ttl=300)
def get_challans(vehicle_no: str) -> dict[str, Any]:
    """Retrieve e-Challan data with 5-minute caching for Streamlit.

    Uses live provider mode if USE_LIVE_API=true and CHALLAN_API_KEY is set.
    Gracefully falls back to demo mode with a warning if the live call fails.

    Args:
        vehicle_no: Raw or formatted Indian registration number.

    Returns:
        dict: Normalized structure:
            {
                "source": "live" | "demo",
                "vehicle": str,
                "total_pending": int,
                "total_due": float,
                "challans": list[dict],
                "warning": str | None
            }
    """
    return _execute_challan_lookup(vehicle_no)
