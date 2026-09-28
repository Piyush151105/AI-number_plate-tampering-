"""Unit tests for challan_service.py: validation, demo mode, live mode, and fallback logic."""

import os
from unittest.mock import MagicMock, patch
import pytest
import requests

from challan_service import (
    _execute_challan_lookup,
    fetch_challans_demo,
    fetch_challans_live,
    get_challans,
    validate_vehicle_number,
)


class TestVehicleNumberValidation:
    """Test validate_vehicle_number with standard Indian and Bharat (BH) formats."""

    @pytest.mark.parametrize(
        "raw_plate, expected",
        [
            ("MH12AB1234", "MH12AB1234"),
            ("mh-12 ab 1234", "MH12AB1234"),
            ("DL-2SKA-2187", "DL2SKA2187"),
            ("KA011234", "KA011234"),
            ("MH46BW1612", "MH46BW1612"),
            ("22BH1234AA", "22BH1234AA"),
            ("21bh9999a", "21BH9999A"),
            ("  GJ01cd5678  ", "GJ01CD5678"),
        ],
    )
    def test_valid_plate_formats(self, raw_plate: str, expected: str):
        assert validate_vehicle_number(raw_plate) == expected

    @pytest.mark.parametrize(
        "invalid_plate",
        [
            "INVALID",
            "12345",
            "",
            "   ",
            "ABC",
            "MH123456789",
            "MH12ABCDE1234",
            "123BH1234A",
            "US-PLATE-12",
        ],
    )
    def test_invalid_plate_formats_raise_value_error(self, invalid_plate: str):
        with pytest.raises(ValueError, match="Invalid Indian vehicle registration number format|cannot be empty"):
            validate_vehicle_number(invalid_plate)


class TestDemoChallanService:
    """Test fetch_challans_demo logic and normalized data structure."""

    def test_known_plate_with_pending_challans(self):
        res = fetch_challans_demo("MH46BW1612")
        assert res["total_pending"] == 2
        assert res["total_due"] == 2000.0
        assert len(res["challans"]) == 2

        # Check required schema fields
        first = res["challans"][0]
        for field in ("challan_no", "date", "offence", "location", "amount", "due_amount", "status"):
            assert field in first

    def test_clean_vehicle_record(self):
        res = fetch_challans_demo("KA01AB1234")
        assert res["total_pending"] == 0
        assert res["total_due"] == 0.0
        assert res["challans"] == []

    def test_arbitrary_plate_deterministic_behavior(self):
        res1 = fetch_challans_demo("DL01AB9999")
        res2 = fetch_challans_demo("DL01AB9999")
        assert res1 == res2
        assert "total_pending" in res1
        assert "total_due" in res1
        assert "challans" in res1


class TestLiveChallanServiceAndFallback:
    """Test live third-party querying, credential handling, and graceful demo fallback."""

    @patch("challan_service.requests.get")
    def test_live_lookup_success(self, mock_get: MagicMock, monkeypatch: pytest.MonkeyPatch):
        secret_key = "super_secret_test_key_xyz"
        monkeypatch.setenv("USE_LIVE_API", "true")
        monkeypatch.setenv("CHALLAN_API_URL", "https://api.testprovider.com/v1/challans")
        monkeypatch.setenv("CHALLAN_API_KEY", secret_key)

        # Mock successful provider response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "challans": [
                {
                    "challan_no": "TEST_LIVE_001",
                    "date": "2026-03-01 10:15",
                    "offence": "Exceeding speed limit",
                    "location": "NH-48 Corridor",
                    "amount": 1000.0,
                    "due_amount": 1000.0,
                    "status": "Pending",
                }
            ]
        }
        mock_get.return_value = mock_response

        # Execute lookup
        res = _execute_challan_lookup("MH12AB1234")

        assert res["source"] == "live"
        assert res["vehicle"] == "MH12AB1234"
        assert res["total_pending"] == 1
        assert res["total_due"] == 1000.0
        assert res["warning"] is None
        assert len(res["challans"]) == 1
        assert res["challans"][0]["challan_no"] == "TEST_LIVE_001"

        # Verify request parameters
        mock_get.assert_called()
        call_kwargs = mock_get.call_args[1]
        assert call_kwargs["timeout"] == 10
        assert call_kwargs["params"] == {"vehicle_number": "MH12AB1234"}
        assert call_kwargs["headers"]["Authorization"] == f"Bearer {secret_key}"

    @patch("challan_service.requests.get")
    def test_live_lookup_timeout_falls_back_to_demo(self, mock_get: MagicMock, monkeypatch: pytest.MonkeyPatch):
        secret_key = "secret_key_12345"
        monkeypatch.setenv("USE_LIVE_API", "true")
        monkeypatch.setenv("CHALLAN_API_URL", "https://api.testprovider.com/v1/challans")
        monkeypatch.setenv("CHALLAN_API_KEY", secret_key)

        # Mock timeout exception on all attempts
        mock_get.side_effect = requests.exceptions.Timeout("Connection timed out after 10s")

        res = _execute_challan_lookup("MH46BW1612")

        # Must fall back gracefully to demo mode
        assert res["source"] == "demo"
        assert res["vehicle"] == "MH46BW1612"
        assert res["warning"] is not None
        assert "Fell back to demo mode" in res["warning"]
        # Must return valid demo citations
        assert res["total_pending"] == 2
        assert res["total_due"] == 2000.0

        # Sensitive API key must NEVER appear in the warning message
        assert secret_key not in res["warning"]

    @patch("challan_service.requests.get")
    def test_live_lookup_http_500_falls_back_to_demo(self, mock_get: MagicMock, monkeypatch: pytest.MonkeyPatch):
        secret_key = "secret_key_500"
        monkeypatch.setenv("USE_LIVE_API", "true")
        monkeypatch.setenv("CHALLAN_API_URL", "https://api.testprovider.com/v1/challans")
        monkeypatch.setenv("CHALLAN_API_KEY", secret_key)

        mock_err_resp = MagicMock()
        mock_err_resp.status_code = 500
        mock_err_resp.raise_for_status.side_effect = requests.exceptions.HTTPError("500 Internal Server Error")
        mock_get.return_value = mock_err_resp

        res = _execute_challan_lookup("MH46BW1612")

        assert res["source"] == "demo"
        assert res["warning"] is not None
        assert "500" in res["warning"]
        assert secret_key not in res["warning"]

    @patch("challan_service.requests.get")
    def test_live_lookup_http_401_unauthorized_falls_back_to_demo(self, mock_get: MagicMock, monkeypatch: pytest.MonkeyPatch):
        secret_key = "bad_key"
        monkeypatch.setenv("USE_LIVE_API", "true")
        monkeypatch.setenv("CHALLAN_API_URL", "https://api.testprovider.com/v1/challans")
        monkeypatch.setenv("CHALLAN_API_KEY", secret_key)

        mock_err_resp = MagicMock()
        mock_err_resp.status_code = 401
        mock_err_resp.raise_for_status.side_effect = requests.exceptions.HTTPError("401 Unauthorized")
        mock_get.return_value = mock_err_resp

        res = _execute_challan_lookup("MH46BW1612")

        assert res["source"] == "demo"
        assert res["warning"] is not None
        assert "401" in res["warning"]
        assert secret_key not in res["warning"]

    def test_demo_mode_when_use_live_is_false(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("USE_LIVE_API", "false")
        monkeypatch.setenv("CHALLAN_API_KEY", "some_key")

        res = _execute_challan_lookup("MH46BW1612")
        assert res["source"] == "demo"
        assert res["warning"] is None
        assert res["total_pending"] == 2


class TestStreamlitCaching:
    """Test that get_challans works and caches result."""

    def test_get_challans_caching(self):
        get_challans.clear()
        res1 = get_challans("MH46BW1612")
        res2 = get_challans("MH46BW1612")
        assert res1 == res2
        assert res1["source"] == "demo"
        assert res1["total_pending"] == 2
