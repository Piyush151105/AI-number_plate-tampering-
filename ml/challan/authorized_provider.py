"""Authorized live e-Challan provider adapter for official/authorized API gateways."""

from __future__ import annotations

import requests
from ml.challan.base import ChallanProvider
from ml.challan.schemas import ChallanCheckResponse, ChallanItem


class AuthorizedChallanProvider(ChallanProvider):
    """
    Adapter communicating with an authorized e-Challan REST API gateway.
    Ensures safe timeouts, credential injection, and response normalization.
    """

    def __init__(self, base_url: str, api_key: str, timeout_sec: int = 8) -> None:
        self.base_url = base_url.strip()
        self.api_key = api_key.strip()
        self.timeout_sec = timeout_sec

    @property
    def provider_name(self) -> str:
        return "authorized"

    async def check_challans(self, vehicle_number: str) -> ChallanCheckResponse:
        v_num = vehicle_number.upper().strip()

        if not self.base_url or not self.api_key:
            raise PermissionError("UNAUTHORIZED_PROVIDER: Missing authorized provider credentials.")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "x-api-key": self.api_key,
            "Accept": "application/json",
            "User-Agent": "AIPlateTamperingSystem/1.4",
        }

        try:
            # Server-side HTTP call with strict timeout
            response = requests.get(
                self.base_url,
                params={"vehicleNumber": v_num, "plate": v_num},
                headers=headers,
                timeout=self.timeout_sec,
            )
        except requests.Timeout as exc:
            raise TimeoutError("REQUEST_TIMEOUT: Authorized e-Challan gateway timed out.") from exc
        except requests.RequestException as exc:
            raise ConnectionError("PROVIDER_UNAVAILABLE: Could not connect to authorized gateway.") from exc

        if response.status_code == 401 or response.status_code == 403:
            raise PermissionError("UNAUTHORIZED_PROVIDER: Invalid credentials for authorized provider.")
        if response.status_code != 200:
            raise ConnectionError(f"PROVIDER_UNAVAILABLE: Gateway responded with status {response.status_code}.")

        try:
            payload = response.json()
        except Exception as exc:
            raise ValueError("PROVIDER_UNAVAILABLE: Invalid JSON payload returned by provider.") from exc

        challans = self._adapt_provider_payload(payload)
        pending_items = [c for c in challans if c.status.lower() in ("pending", "unpaid")]
        total_due = sum(c.dueAmount for c in pending_items)

        return ChallanCheckResponse(
            success=True,
            vehicleNumber=v_num,
            totalPending=len(pending_items),
            totalDue=total_due,
            challans=challans,
            provider=self.provider_name,
            disclaimer=None,
        )

    def _adapt_provider_payload(self, data: dict) -> list[ChallanItem]:
        """Adapter mapping third-party / authorized gateway JSON to ChallanItem models."""
        raw_items = data.get("challans") or data.get("data") or data.get("records") or []
        adapted: list[ChallanItem] = []

        for item in raw_items:
            challan_no = str(item.get("challanNumber") or item.get("challan_no") or item.get("number") or "UNKNOWN")
            date_str = str(item.get("date") or item.get("challanDate") or item.get("timestamp") or "N/A")
            offence_str = str(item.get("offence") or item.get("violation") or item.get("reason") or "Traffic Offense")
            location_str = str(item.get("location") or item.get("place") or "Traffic Division")
            
            try:
                amt = float(item.get("amount") or item.get("fineAmount") or 0.0)
            except (ValueError, TypeError):
                amt = 0.0

            try:
                due_amt = float(item.get("dueAmount") or item.get("due_amount") or amt)
            except (ValueError, TypeError):
                due_amt = amt

            status_str = str(item.get("status") or ("Pending" if due_amt > 0 else "Paid"))

            adapted.append(
                ChallanItem(
                    challanNumber=challan_no,
                    date=date_str,
                    offence=offence_str,
                    location=location_str,
                    amount=amt,
                    status=status_str,
                    dueAmount=due_amt,
                )
            )

        return adapted
