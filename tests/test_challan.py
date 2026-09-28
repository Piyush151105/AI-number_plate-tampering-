"""Unit and integration tests for e-Challan checking feature without external testclient dependency."""

from __future__ import annotations

from unittest.mock import MagicMock
import pytest

from backend.main import check_challan
from ml.challan.base import ChallanProvider
from ml.challan.mock_provider import MockChallanProvider
from ml.challan.schemas import ChallanCheckRequest, ChallanCheckResponse, ChallanItem
from ml.challan.service import ChallanService


# ------------------------------------------------------------------------------
# 1. Vehicle Number Normalization & Validation Tests
# ------------------------------------------------------------------------------
def test_vehicle_number_normalization() -> None:
    service = ChallanService()
    assert service.normalize_vehicle_number("mh 12 ab 1234") == "MH12AB1234"
    assert service.normalize_vehicle_number("MH-46-BW-1612") == "MH46BW1612"
    assert service.normalize_vehicle_number("dl.01.sk.2187") == "DL01SK2187"
    assert service.normalize_vehicle_number("  ka 01 ab 1234  ") == "KA01AB1234"


def test_vehicle_number_validation() -> None:
    service = ChallanService()
    # Valid Indian plate patterns
    assert service.is_valid_vehicle_number("MH12AB1234") is True
    assert service.is_valid_vehicle_number("MH46BW1612") is True
    assert service.is_valid_vehicle_number("DL2SKA2187") is True
    assert service.is_valid_vehicle_number("KA01A1234") is True

    # Invalid patterns
    assert service.is_valid_vehicle_number("INVALID") is False
    assert service.is_valid_vehicle_number("1234567890") is False
    assert service.is_valid_vehicle_number("") is False
    assert service.is_valid_vehicle_number("M12AB1234") is False


# ------------------------------------------------------------------------------
# 2. Mock Provider Tests
# ------------------------------------------------------------------------------
@pytest.mark.anyio
async def test_mock_provider_known_multiple_challans() -> None:
    provider = MockChallanProvider()
    res = await provider.check_challans("MH46BW1612")

    assert res.success is True
    assert res.vehicleNumber == "MH46BW1612"
    assert res.totalPending == 2
    assert res.totalDue == 2000.0
    assert len(res.challans) == 2
    assert res.provider == "mock"
    assert "sample data" in (res.disclaimer or "").lower()
    assert res.challans[0].challanNumber == "SATCO25SKFJ68C32"
    assert res.challans[1].challanNumber == "SATCM24000267081"


@pytest.mark.anyio
async def test_mock_provider_clean_vehicle() -> None:
    provider = MockChallanProvider()
    res = await provider.check_challans("KA01AB1234")

    assert res.success is True
    assert res.vehicleNumber == "KA01AB1234"
    assert res.totalPending == 0
    assert res.totalDue == 0.0
    assert len(res.challans) == 0


# ------------------------------------------------------------------------------
# 3. API Endpoint Tests (POST /api/challan/check)
# ------------------------------------------------------------------------------
@pytest.mark.anyio
async def test_api_check_valid_vehicle() -> None:
    req = MagicMock()
    req.client.host = "127.0.0.1"

    payload = ChallanCheckRequest(vehicleNumber="MH46BW1612")
    res = await check_challan(payload, req)

    assert isinstance(res, ChallanCheckResponse)
    assert res.success is True
    assert res.vehicleNumber == "MH46BW1612"
    assert res.totalPending == 2
    assert res.totalDue == 2000.0
    assert res.provider == "mock"
    assert len(res.challans) == 2


@pytest.mark.anyio
async def test_api_check_normalizes_spacing_in_request() -> None:
    req = MagicMock()
    req.client.host = "127.0.0.1"

    payload = ChallanCheckRequest(vehicleNumber="mh 46 bw 1612")
    res = await check_challan(payload, req)

    assert isinstance(res, ChallanCheckResponse)
    assert res.vehicleNumber == "MH46BW1612"


@pytest.mark.anyio
async def test_api_check_invalid_vehicle() -> None:
    req = MagicMock()
    req.client.host = "127.0.0.1"

    payload = ChallanCheckRequest(vehicleNumber="NOT_A_PLATE")
    res = await check_challan(payload, req)

    # Returns JSONResponse with 400 status code
    assert res.status_code == 400
    import json
    data = json.loads(res.body.decode())
    assert data["success"] is False
    assert data["error"]["code"] == "INVALID_VEHICLE_NUMBER"
    assert data["error"]["message"] == "Please enter a valid vehicle registration number."


# ------------------------------------------------------------------------------
# 4. Error Shielding & Security Tests
# ------------------------------------------------------------------------------
@pytest.mark.anyio
async def test_service_timeout_shielding(monkeypatch) -> None:
    service = ChallanService()

    class TimeoutProvider(ChallanProvider):
        @property
        def provider_name(self) -> str:
            return "timeout_sim"

        async def check_challans(self, vehicle_number: str) -> ChallanCheckResponse:
            raise TimeoutError("Gateway connection timed out")

    monkeypatch.setattr(service, "_resolve_provider", lambda: TimeoutProvider())
    success_resp, err_resp, code = await service.check("MH12AB1234")

    assert success_resp is None
    assert err_resp is not None
    assert code == 504
    assert err_resp.error.code == "REQUEST_TIMEOUT"
    assert err_resp.error.message == "Live e-Challan service timed out."


@pytest.mark.anyio
async def test_service_unauthorized_shielding(monkeypatch) -> None:
    service = ChallanService()

    class UnauthorizedProvider(ChallanProvider):
        @property
        def provider_name(self) -> str:
            return "unauth_sim"

        async def check_challans(self, vehicle_number: str) -> ChallanCheckResponse:
            raise PermissionError("Invalid API key")

    monkeypatch.setattr(service, "_resolve_provider", lambda: UnauthorizedProvider())
    success_resp, err_resp, code = await service.check("MH12AB1234")

    assert success_resp is None
    assert err_resp is not None
    assert code == 502
    assert err_resp.error.code == "UNAUTHORIZED_PROVIDER"
    assert err_resp.error.message == "Live e-Challan provider is not configured."
    # Ensure no secret keys leaked in error message
    assert "Invalid API key" not in err_resp.error.message
