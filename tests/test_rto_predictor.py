"""Unit and integration tests for RTO-Based Missing Character Predictor module."""

from __future__ import annotations

import pytest

from backend.main import (
    PlatePredictionRequest,
    get_rto_constraint_info,
    predict_missing_plate_characters,
)
from ml.rto_predictor import (
    PROHIBITED_LETTERS,
    RTOMissingCharacterPredictor,
    rto_predictor,
)


# ------------------------------------------------------------------------------
# 1. RTO Geography Mapping Tests
# ------------------------------------------------------------------------------
def test_rto_geography_mapping_mh46() -> None:
    """Verify MH-46 resolves to Panvel, Raigad, Maharashtra with its constraints."""
    info = rto_predictor.get_rto_constraint("MH-46")
    assert info is not None
    assert info.rto_code == "MH-46"
    assert info.prefix == "MH46"
    assert info.state_code == "MH"
    assert "Panvel" in info.location
    assert info.district == "Raigad"
    assert info.state_name == "Maharashtra"
    assert "BW" in info.allowed_series
    assert all(p not in info.allowed_series for p in PROHIBITED_LETTERS)


def test_rto_geography_mapping_mh08() -> None:
    """Verify MH-08 resolves to Ratnagiri, Maharashtra with its constraints."""
    info = rto_predictor.get_rto_constraint("MH-08")
    assert info is not None
    assert info.rto_code == "MH-08"
    assert info.prefix == "MH08"
    assert "Ratnagiri" in info.location
    assert info.district == "Ratnagiri"
    assert "AX" in info.allowed_series


def test_rto_geography_mapping_various_formats() -> None:
    """Verify lookup works for 'MH46', 'mh 46', 'MH-46'."""
    for q in ["MH-46", "MH46", "mh 46", "mh-46"]:
        info = rto_predictor.get_rto_constraint(q)
        assert info is not None
        assert info.rto_code == "MH-46"


# ------------------------------------------------------------------------------
# 2. Partial Plate Parser Tests
# ------------------------------------------------------------------------------
def test_parse_partial_plate_spaced() -> None:
    """Test parsing space-separated partial plate: 'MH 46 B* 161*'."""
    state, rto, series, num = rto_predictor.parse_partial_plate("MH 46 B* 161*")
    assert state == "MH"
    assert rto == "46"
    assert series == "B*"
    assert num == "161*"


def test_parse_partial_plate_double_wildcard() -> None:
    """Test parsing double wildcard series: 'MH 46 ** 1612'."""
    state, rto, series, num = rto_predictor.parse_partial_plate("MH 46 ** 1612")
    assert state == "MH"
    assert rto == "46"
    assert series == "**"
    assert num == "1612"


def test_parse_partial_plate_contiguous() -> None:
    """Test parsing unspaced string: 'MH46B*161*'."""
    state, rto, series, num = rto_predictor.parse_partial_plate("MH46B*161*")
    assert state == "MH"
    assert rto == "46"
    assert series == "B*"
    assert num == "161*"


def test_parse_partial_plate_question_mark() -> None:
    """Test '?' wildcard converted to '*'."""
    state, rto, series, num = rto_predictor.parse_partial_plate("MH 46 B? 161?")
    assert series == "B*"
    assert num == "161*"


# ------------------------------------------------------------------------------
# 3. ML Prediction Engine Tests
# ------------------------------------------------------------------------------
def test_predict_mh46_b_wildcard() -> None:
    """
    Test prediction on 'MH 46 B* 161*'.
    Should predict MH 46 BW 1612 as a top candidate with Two-Wheeler ground truth.
    """
    result = rto_predictor.predict("MH 46 B* 161*", top_k=10)
    assert result.status == "SUCCESS"
    assert result.rto_code == "MH-46"
    assert result.rto_area_info is not None
    assert "Panvel" in result.rto_area_info.location
    assert len(result.candidates) > 0
    assert len(result.candidates) <= 10

    # Verify candidates are sorted descending by confidence
    scores = [c.match_confidence for c in result.candidates]
    assert scores == sorted(scores, reverse=True)

    # Check that top candidate is MH 46 BW 1612
    plates = [c.complete_plate for c in result.candidates]
    assert "MH 46 BW 1612" in plates
    top_cand = result.candidates[0]
    assert top_cand.complete_plate == "MH 46 BW 1612"
    assert top_cand.match_confidence > 0.80
    assert "Panvel" in top_cand.rto_area
    assert "Two-Wheeler" in top_cand.vehicle_class
    assert top_cand.is_verified_truth is True
    assert "Hero Splendor" in top_cand.vehicle_details.get("maker_model", "")


def test_predict_mh46_double_wildcard_series() -> None:
    """
    Test prediction on 'MH 46 ** 1612'.
    Missing full series letters. Should generate valid Panvel series.
    """
    result = rto_predictor.predict("MH 46 ** 1612", top_k=10)
    assert result.status == "SUCCESS"
    assert len(result.candidates) > 0
    for cand in result.candidates:
        assert cand.complete_plate.startswith("MH 46 ")
        assert cand.complete_plate.endswith(" 1612")
        # Ensure no prohibited characters
        assert not any(p in cand.raw_plate for p in PROHIBITED_LETTERS)


def test_predict_mh08_ratnagiri() -> None:
    """
    Test prediction on 'MH 08 A* 1400'.
    MH 08 AX 1400 must be predicted with exact Four-Wheeler (Motor Car / LMV) ground truth.
    """
    result = rto_predictor.predict("MH 08 A* 1400", top_k=10)
    assert result.status == "SUCCESS"
    assert result.rto_code == "MH-08"
    assert "Ratnagiri" in result.rto_area_info.location
    plates = [c.complete_plate for c in result.candidates]
    assert "MH 08 AX 1400" in plates

    # Top candidate should be the verified ground-truth car
    top_cand = result.candidates[0]
    assert top_cand.complete_plate == "MH 08 AX 1400"
    assert "Four-Wheeler" in top_cand.vehicle_class or "Motor Car" in top_cand.vehicle_class
    assert top_cand.is_verified_truth is True
    assert "Maruti Suzuki Swift" in top_cand.vehicle_details.get("maker_model", "")
    assert top_cand.vehicle_details.get("fuel_type") == "PETROL"


def test_vehicle_category_series_heuristics() -> None:
    """Test vehicle category classification heuristics for series in MH, DL, KA."""
    from ml.rto_predictor import predict_vehicle_category

    rto_mh08 = rto_predictor.get_rto_constraint("MH-08")
    rto_dl01 = rto_predictor.get_rto_constraint("DL-01")

    # Ratnagiri MH-08 series AY should classify as Four-Wheeler
    cat_ay, is_t, _, _ = predict_vehicle_category("MH08AY9999", rto_mh08, "AY", "9999")
    assert "Four-Wheeler" in cat_ay

    # Ratnagiri MH-08 series BA should classify as Two-Wheeler
    cat_ba, _, _, _ = predict_vehicle_category("MH08BA9999", rto_mh08, "BA", "9999")
    assert "Two-Wheeler" in cat_ba

    # Delhi DL-01 series C (Car) vs S (Two-Wheeler)
    cat_dl_c, _, _, _ = predict_vehicle_category("DL01CA9999", rto_dl01, "CA", "9999")
    assert "Four-Wheeler" in cat_dl_c

    cat_dl_s, _, _, _ = predict_vehicle_category("DL01SA9999", rto_dl01, "SA", "9999")
    assert "Two-Wheeler" in cat_dl_s


def test_predict_target_category_filtering() -> None:
    """Test filtering candidates by target vehicle category."""
    # When filtering for Four-Wheeler on MH 46
    res_4w = rto_predictor.predict("MH 46 ** 1612", top_k=5, target_category="Four-Wheeler")
    assert res_4w.status == "SUCCESS"
    assert len(res_4w.candidates) > 0
    for cand in res_4w.candidates:
        assert "Four-Wheeler" in cand.vehicle_class or "Car" in cand.vehicle_class

    # When filtering for Two-Wheeler on MH 46
    res_2w = rto_predictor.predict("MH 46 ** 1612", top_k=5, target_category="Two-Wheeler")
    assert res_2w.status == "SUCCESS"
    assert len(res_2w.candidates) > 0
    for cand in res_2w.candidates:
        assert "Two-Wheeler" in cand.vehicle_class


def test_predict_delhi_series() -> None:
    """Test prediction on 'DL 01 C* 5678'."""
    result = rto_predictor.predict("DL 01 C* 5678", top_k=5)
    assert result.status == "SUCCESS"
    assert result.rto_code == "DL-01"
    assert "Delhi" in result.rto_area_info.state_name
    for cand in result.candidates:
        assert cand.complete_plate.startswith("DL 01 C")


def test_prohibited_characters_never_generated() -> None:
    """Ensure characters 'I', 'O', and 'Q' are strictly excluded per MoRTH standard."""
    result = rto_predictor.predict("MH 46 ** ****", top_k=20)
    for cand in result.candidates:
        series_part = cand.complete_plate.split()[2]
        for prohibited in PROHIBITED_LETTERS:
            assert prohibited not in series_part, f"Prohibited letter '{prohibited}' found in series '{series_part}'"


def test_invalid_empty_input_handling() -> None:
    """Ensure empty or invalid input produces clean error status."""
    res = rto_predictor.predict("")
    assert res.status == "INVALID_INPUT"
    assert len(res.candidates) == 0


# ------------------------------------------------------------------------------
# 4. Backend API Endpoints Integration Tests
# ------------------------------------------------------------------------------
def test_api_predict_endpoint() -> None:
    """Test predict_missing_plate_characters function via API request model."""
    req = PlatePredictionRequest(partial_plate="MH 46 B* 161*", top_k=5)
    resp = predict_missing_plate_characters(req)
    assert resp.status == "SUCCESS"
    assert resp.rto_code == "MH-46"
    assert resp.rto_area_info is not None
    assert "Panvel" in resp.rto_area_info.location
    assert len(resp.candidates) == 5
    assert resp.candidates[0].complete_plate == "MH 46 BW 1612"
    assert resp.candidates[0].is_verified_truth is True
    assert "Two-Wheeler" in resp.candidates[0].vehicle_class
    assert resp.candidates[0].match_confidence_pct != ""


def test_api_rto_constraint_info_endpoint() -> None:
    """Test get_rto_constraint_info endpoint."""
    info46 = get_rto_constraint_info("MH-46")
    assert info46.rto_code == "MH-46"
    assert "Panvel" in info46.location
    assert len(info46.allowed_series) > 0

    info08 = get_rto_constraint_info("MH08")
    assert info08.rto_code == "MH-08"
    assert "Ratnagiri" in info08.location
