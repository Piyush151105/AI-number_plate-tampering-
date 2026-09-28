"""ChallanService orchestrator managing provider resolution, validation, rate limiting, and safe errors."""

from __future__ import annotations

import re
import time
from collections import defaultdict

from config.settings import settings
from ml.challan.authorized_provider import AuthorizedChallanProvider
from ml.challan.base import ChallanProvider
from ml.challan.mock_provider import MockChallanProvider
from ml.challan.schemas import (
    ChallanCheckResponse,
    ChallanErrorDetail,
    ChallanErrorResponse,
)


class ChallanService:
    """Core service coordinating e-Challan checks across decoupled providers."""

    def __init__(self) -> None:
        self.plate_pattern = re.compile(settings.plate_regex)
        self._request_log: dict[str, list[float]] = defaultdict(list)

    @staticmethod
    def normalize_vehicle_number(raw_text: str) -> str:
        """
        Normalize vehicle plate string:
        e.g., 'mh 12 ab 1234', 'MH-46-BW-1612', 'MNHO8AX1LOO' -> 'MH12AB1234', 'MH46BW1612', 'MH08AX1400'.
        """
        from ml.ocr_engine import clean_indian_plate
        return clean_indian_plate(raw_text)

    def is_valid_vehicle_number(self, normalized_plate: str) -> bool:
        """Validate if the normalized plate string matches expected Indian format."""
        return bool(self.plate_pattern.match(normalized_plate))

    def _resolve_provider(self) -> ChallanProvider:
        """Resolve the configured provider based on settings."""
        provider_name = (settings.challan_provider or "mock").lower().strip()

        if provider_name == "authorized":
            return AuthorizedChallanProvider(
                base_url=settings.challan_api_base_url,
                api_key=settings.challan_api_key,
                timeout_sec=settings.challan_timeout_sec,
            )

        return MockChallanProvider()

    def _check_rate_limit(self, client_id: str) -> bool:
        """In-memory sliding window rate limiter."""
        now = time.time()
        window = 60.0
        max_req = settings.challan_rate_limit_per_min

        timestamps = [ts for ts in self._request_log[client_id] if now - ts < window]
        self._request_log[client_id] = timestamps

        if len(timestamps) >= max_req:
            return False

        self._request_log[client_id].append(now)
        return True

    async def check(
        self,
        vehicle_number: str,
        client_id: str = "default_client",
    ) -> tuple[ChallanCheckResponse | None, ChallanErrorResponse | None, int]:
        """
        Execute e-Challan check.
        Returns (success_response, error_response, http_status_code).
        """
        normalized = self.normalize_vehicle_number(vehicle_number)

        # 1. Validation
        if not normalized or not self.is_valid_vehicle_number(normalized):
            err = ChallanErrorResponse(
                success=False,
                error=ChallanErrorDetail(
                    code="INVALID_VEHICLE_NUMBER",
                    message="Please enter a valid vehicle registration number.",
                ),
            )
            return None, err, 400

        # 2. Rate limiting
        if not self._check_rate_limit(client_id):
            err = ChallanErrorResponse(
                success=False,
                error=ChallanErrorDetail(
                    code="RATE_LIMIT_EXCEEDED",
                    message="Too many requests. Please wait a moment before trying again.",
                ),
            )
            return None, err, 429

        # 3. Provider invocation with safe error shielding
        provider = self._resolve_provider()

        try:
            res = await provider.check_challans(normalized)
            return res, None, 200
        except TimeoutError:
            err = ChallanErrorResponse(
                success=False,
                error=ChallanErrorDetail(
                    code="REQUEST_TIMEOUT",
                    message="Live e-Challan service timed out.",
                ),
            )
            return None, err, 504
        except PermissionError:
            err = ChallanErrorResponse(
                success=False,
                error=ChallanErrorDetail(
                    code="UNAUTHORIZED_PROVIDER",
                    message="Live e-Challan provider is not configured.",
                ),
            )
            return None, err, 502
        except Exception:
            err = ChallanErrorResponse(
                success=False,
                error=ChallanErrorDetail(
                    code="PROVIDER_UNAVAILABLE",
                    message="Live e-Challan service is temporarily unavailable.",
                ),
            )
            return None, err, 502


challan_service = ChallanService()
