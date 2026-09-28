"""Streamlit dashboard for AI plate tampering detection and e-Challan verification."""

from __future__ import annotations

import io
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import requests
import streamlit as st
from PIL import Image

import challan_service

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="AI Plate Tampering & e-Challan System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Session state initialization
if "detection_data" not in st.session_state:
    st.session_state.detection_data = None
if "challan_check_result" not in st.session_state:
    st.session_state.challan_check_result = None
if "manual_challan_result" not in st.session_state:
    st.session_state.manual_challan_result = None
if "current_file_name" not in st.session_state:
    st.session_state.current_file_name = None
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "prediction_input_text" not in st.session_state:
    st.session_state.prediction_input_text = "MH 46 B* 161*"

st.title("AI-Enabled Vehicle Number Plate Tampering & e-Challan Verification")
st.caption("Computer Vision · Deep Learning Forensics · e-Challan Integration")

with st.sidebar:
    st.header("System Status")
    api_url = st.text_input("API URL", value=API_URL)
    try:
        health = requests.get(f"{api_url}/health", timeout=3).json()
        st.caption(f"API Engine: **{health.get('engine', 'unknown')}** (v{health.get('version', '?')})")
        st.success("Backend Connected", icon="✅")
    except requests.RequestException:
        st.error("Backend not reachable. Start `python run_api.py` first.", icon="⚠️")

    st.markdown(
        """
        ### Pipeline Architecture
        1. **Plate Localization**: Edge & contour filtering + text-guided candidate fallback
        2. **Dual-Row OCR**: EasyOCR spatial sorting with Indian state code normalization
        3. **Forensic Analysis**: Multi-signal heuristics & TamperCNN
        4. **e-Challan Integration**: Standardized `POST /api/challan/check` via decoupled provider adapter
        """
    )
    st.divider()
    st.caption("Current Provider Mode: `CHALLAN_PROVIDER=mock`")


def render_challan_display(data: dict, current_plate_num: str = "") -> None:
    """Render standardized e-Challan results matching project visual style."""
    official_portal_url = "https://mahatrafficechallan.gov.in/payechallan/"
    parivahan_accused_url = "https://echallan.parivahan.gov.in/index/accused-challan"
    parivahan_url = "https://echallan.parivahan.gov.in/"

    if not data:
        st.info("No e-Challan query performed yet.")
        return

    # Check for legacy error object or friendly error string
    if not data.get("success", True) and "error" in data:
        err = data.get("error", {})
        code = err.get("code", "ERROR")
        msg = err.get("message", "An unexpected error occurred.")
        st.error(f"⚠️ {msg}")
        return

    # Normalized fields (support both new normalized keys and legacy backend keys)
    source = str(data.get("source") or data.get("provider", "demo")).lower()
    v_num = data.get("vehicle") or data.get("vehicleNumber", current_plate_num or "UNKNOWN")
    total_pending = int(data.get("total_pending", data.get("totalPending", 0)))
    total_due = float(data.get("total_due", data.get("totalDue", 0.0)))
    challans = data.get("challans", [])
    warning = data.get("warning")

    # Header status card
    has_violations = total_pending > 0
    border_color = "#e53e3e" if has_violations else "#38a169"
    badge_label = f"🔴 {total_pending} PENDING CHALLAN(S)" if has_violations else "🟢 0 PENDING CHALLANS (CLEAN RECORD)"

    st.markdown(
        f"""
        <div style="border: 2px solid {border_color}; border-radius: 10px; padding: 18px; background-color: rgba(255,255,255,0.03); margin-top: 10px; margin-bottom: 12px;">
            <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255,255,255,0.12); padding-bottom: 10px;">
                <h4 style="margin: 0; color: #3498db;">🏛️ e-Challan Verification Status</h4>
                <span style="background: {border_color}; color: white; padding: 4px 14px; border-radius: 12px; font-size: 13px; font-weight: bold;">
                    {badge_label}
                </span>
            </div>
            <div style="margin-top: 12px; font-size: 15px; line-height: 1.8;">
                <b>Vehicle Registration:</b> <code style="font-size: 16px; font-weight: bold; color: #f1c40f;">{v_num}</code><br>
                <b>Pending Citations:</b> {total_pending} &nbsp;|&nbsp; <b>Total Outstanding Due:</b> <span style="color: {border_color}; font-weight: bold; font-size: 16px;">₹{total_due:,.2f} INR</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Informative warning if live lookup had to fall back to demo mode
    if warning:
        st.warning(f"⚠️ {warning}")

    # Mode Indicator: Show Demo Mode banner ONLY when source == "demo"
    if source == "demo":
        st.markdown(
            """
            <div style="background: rgba(52, 152, 219, 0.08); border: 1px solid #3182ce; border-radius: 8px; padding: 12px 16px; margin-bottom: 14px;">
                <div style="font-weight: bold; color: #63b3ed; font-size: 14px; display: flex; align-items: center; gap: 6px;">
                    <span>ℹ️</span> <span>Demo Mode</span>
                </div>
                <div style="font-size: 13px; color: #e2e8f0; margin-top: 3px;">
                    Sample data only. These challan results are sample demonstration data and are not real vehicle records.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif source == "live":
        provider_name = os.getenv("CHALLAN_PROVIDER_NAME", "Third-Party Provider").strip() or "Third-Party Provider"
        st.markdown(
            f"""
            <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid #10b981; border-radius: 8px; padding: 12px 16px; margin-bottom: 14px;">
                <div style="font-weight: bold; color: #34d399; font-size: 14px; display: flex; align-items: center; gap: 6px;">
                    <span>🟢</span> <span>Live Registry Query</span>
                </div>
                <div style="font-size: 13px; color: #e2e8f0; margin-top: 3px;">
                    Live data via {provider_name}, verify on the official portal.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Citations Display
    if challans:
        st.write("#### 📄 Citation Details")

        # Desktop Table View
        table_rows = []
        for c in challans:
            c_no = c.get("challan_no") or c.get("challanNumber") or "N/A"
            c_date = c.get("date") or "N/A"
            c_offence = c.get("offence") or "Traffic Violation"
            c_loc = c.get("location") or "N/A"
            c_amt = float(c.get("amount", 0.0))
            c_due = float(c.get("due_amount", c.get("dueAmount", c_amt)))
            c_status = str(c.get("status", "Pending")).title()

            table_rows.append(
                {
                    "Challan Number": c_no,
                    "Date": c_date,
                    "Violation / Offence": c_offence,
                    "Location": c_loc,
                    "Amount (INR)": f"₹{c_amt:,.2f}",
                    "Due Amount": f"₹{c_due:,.2f}",
                    "Status": c_status,
                }
            )
        st.dataframe(table_rows, use_container_width=True, hide_index=True)

        # Responsive Cards
        with st.expander("📱 View Responsive Citation Cards", expanded=False):
            for idx, c in enumerate(challans, 1):
                c_no = c.get("challan_no") or c.get("challanNumber") or "N/A"
                c_date = c.get("date") or "N/A"
                c_offence = c.get("offence") or "Traffic Violation"
                c_loc = c.get("location") or "N/A"
                c_due = float(c.get("due_amount", c.get("dueAmount", 0.0)))
                c_status = str(c.get("status", "Pending")).title()
                is_pending = c_status.lower() in ("pending", "unpaid")

                st.markdown(
                    f"""
                    <div style="background: #ffffff; color: #1a202c; border-radius: 8px; margin-bottom: 10px; padding: 12px 16px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); border-left: 6px solid {'#e53e3e' if is_pending else '#38a169'};">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="font-weight: bold; font-family: monospace; font-size: 15px;">#{idx} {c_no}</span>
                            <span style="font-weight: bold; color: {'#e53e3e' if is_pending else '#38a169'};">₹{c_due:,.2f}</span>
                        </div>
                        <div style="font-size: 13px; color: #4a5568; margin-top: 4px;"><b>Date:</b> {c_date} · <b>Location:</b> {c_loc}</div>
                        <div style="font-size: 12px; color: #718096; margin-top: 2px;">{c_offence}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    else:
        st.success("🟢 No pending challans found for this vehicle. Clean driving record!")

    # Official Verification Section
    st.markdown(
        f"""
        <div style="background: rgba(255, 255, 255, 0.04); border: 1px solid rgba(255, 255, 255, 0.12); border-radius: 8px; padding: 14px 16px; margin-top: 15px; margin-bottom: 10px;">
            <div style="font-weight: bold; font-size: 14px; color: #e2e8f0;">🏛️ Official Government Verification</div>
            <div style="font-size: 13px; color: #cbd5e0; margin-top: 4px;">
                Enter vehicle number <b>{v_num}</b> on the official portal to verify live government records.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_gov1, col_gov2 = st.columns([1, 1])
    with col_gov1:
        st.link_button("🏛️ Verify on State e-Challan Portal", official_portal_url, use_container_width=True, type="primary")
    with col_gov2:
        st.link_button("🌐 Parivahan Accused Challan Portal", parivahan_accused_url, use_container_width=True)

    # Privacy Note
    st.markdown(
        """
        <div style="margin-top: 14px; padding: 8px 12px; background: rgba(255, 255, 255, 0.02); border-left: 3px solid #64748b; border-radius: 4px; font-size: 12px; color: #94a3b8;">
            🔒 <b>Privacy Notice:</b> Vehicle registration and challan records are personal information. Data is queried in real time strictly for the queried number plate and is not stored or shared.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_rto_prediction_results(pred_data: dict) -> None:
    """Render RTO-Based Missing Character Predictor results."""
    if not pred_data:
        return

    status = pred_data.get("status", "SUCCESS")
    if status != "SUCCESS":
        st.error(f"Prediction Error: {pred_data.get('error_message', 'Unknown error occurred.')}")
        return

    rto_info = pred_data.get("rto_area_info")
    candidates = pred_data.get("candidates", [])
    input_pat = pred_data.get("input_pattern", "")
    norm_pat = pred_data.get("normalized_pattern", "")

    # 1. RTO GEOGRAPHY & SERIES CONSTRAINTS CARD
    st.markdown("### 🏢 Identified Local RTO Geography & Constraints")
    if rto_info:
        rto_code = rto_info.get("rto_code", "N/A")
        rto_loc = rto_info.get("location", "N/A")
        rto_dist = rto_info.get("district", "N/A")
        rto_state = rto_info.get("state_name", "N/A")
        rto_zone = rto_info.get("region_zone", "")
        std_fmt = rto_info.get("standard_format", "SS RR CC NNNN")
        allowed = rto_info.get("allowed_series", [])
        prohibited = rto_info.get("prohibited_letters", ["I", "O", "Q"])

        col_rto1, col_rto2, col_rto3 = st.columns([1.2, 1.2, 1.6])
        with col_rto1:
            st.markdown(
                f"""
                <div style="background: rgba(59, 130, 246, 0.08); border: 1px solid #3b82f6; border-radius: 8px; padding: 14px 18px;">
                    <div style="font-size: 11px; color: #60a5fa; text-transform: uppercase; font-weight: bold;">RTO Jurisdiction</div>
                    <div style="font-size: 24px; font-weight: bold; color: #93c5fd; margin-top: 4px;">{rto_code}</div>
                    <div style="font-size: 14px; color: #bfdbfe; margin-top: 2px;"><b>{rto_loc}</b></div>
                    <div style="font-size: 12px; color: #94a3b8; margin-top: 2px;">District: {rto_dist} · {rto_state}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_rto2:
            st.markdown(
                f"""
                <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid #10b981; border-radius: 8px; padding: 14px 18px;">
                    <div style="font-size: 11px; color: #34d399; text-transform: uppercase; font-weight: bold;">Syntax & MoRTH Rules</div>
                    <div style="font-size: 15px; font-weight: bold; color: #6ee7b7; margin-top: 4px;">Standard: <code>{std_fmt}</code></div>
                    <div style="font-size: 12px; color: #a7f3d0; margin-top: 6px;">🚫 Prohibited Letters: <b>{', '.join(prohibited)}</b></div>
                    <div style="font-size: 12px; color: #94a3b8; margin-top: 2px;">Zone: {rto_zone or 'Standard State Area'}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_rto3:
            active_preview = ", ".join(allowed[:10]) + ("..." if len(allowed) > 10 else "")
            st.markdown(
                f"""
                <div style="background: rgba(255, 255, 255, 0.04); border: 1px solid rgba(255, 255, 255, 0.12); border-radius: 8px; padding: 14px 18px;">
                    <div style="font-size: 11px; color: #94a3b8; text-transform: uppercase; font-weight: bold;">Vehicle Series Constraints</div>
                    <div style="font-size: 12px; color: #cbd5e1; margin-top: 4px;">Active RTO Series Registry: <b>{len(allowed)}</b> series</div>
                    <div style="font-size: 12px; color: #38bdf8; margin-top: 4px; font-family: monospace;">{active_preview}</div>
                    <div style="font-size: 11px; color: #94a3b8; margin-top: 4px;">Classes: Two-Wheeler (B/S), Four-Wheeler (C/D), Commercial (T)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.write("")
    # 2. RESULTS TABLE
    st.markdown("### 📊 Top Predicted Complete Number Plates")
    st.caption(f"Evaluated input pattern <code>{input_pat}</code> (normalized: <code>{norm_pat}</code>). Displaying top {len(candidates)} candidates ranked by Match Confidence Score.")

    if not candidates:
        st.warning("No valid plate candidates could be generated for the given wildcard pattern under RTO constraints.")
        return

    table_rows = []
    for idx, cand in enumerate(candidates, 1):
        plate = cand.get("complete_plate", "")
        conf = cand.get("match_confidence", 0.0)
        conf_pct = cand.get("match_confidence_pct", f"{conf:.1%}")
        v_class = cand.get("vehicle_class", "Private Vehicle")
        rto_loc_str = cand.get("rto_area", "")
        v_status = cand.get("validation_status", "Validated")
        is_truth = cand.get("is_verified_truth", False)
        pred_chars = cand.get("predicted_chars", {})
        predicted_slots = ", ".join([f"Pos {k}: '{v}'" for k, v in pred_chars.items()]) if pred_chars else "Exact Match"

        # Formatted vehicle category with icon
        if "four-wheeler" in v_class.lower() or "car" in v_class.lower():
            cat_display = f"🚗 {v_class}"
        elif "two-wheeler" in v_class.lower() or "motorcycle" in v_class.lower() or "scooter" in v_class.lower():
            cat_display = f"🛵 {v_class}"
        elif "commercial" in v_class.lower() or "transport" in v_class.lower():
            cat_display = f"🚕 {v_class}"
        else:
            cat_display = f"🚘 {v_class}"

        truth_status = "✅ Verified RTO Truth" if is_truth else "🔮 Predicted from Series"

        table_rows.append({
            "Rank": f"#{idx}",
            "Complete Number Plate": plate,
            "Match Confidence Score": conf_pct,
            "Local RTO Area": rto_loc_str,
            "Vehicle Category": cat_display,
            "Ground Truth Status": truth_status,
            "Predicted Characters": predicted_slots,
            "RTO Format Validation": v_status,
        })

    import pandas as pd
    df = pd.DataFrame(table_rows)
    st.dataframe(
        df,
        column_config={
            "Rank": st.column_config.TextColumn("Rank", width="small"),
            "Complete Number Plate": st.column_config.TextColumn("Complete Predicted Plate", width="medium"),
            "Match Confidence Score": st.column_config.TextColumn("Match Confidence Score", width="small"),
            "Local RTO Area": st.column_config.TextColumn("Identified Local RTO Area", width="medium"),
            "Vehicle Category": st.column_config.TextColumn("Vehicle Category", width="medium"),
            "Ground Truth Status": st.column_config.TextColumn("Ground Truth Status", width="medium"),
            "Predicted Characters": st.column_config.TextColumn("Predicted Characters", width="medium"),
            "RTO Format Validation": st.column_config.TextColumn("Validation against RTO Format", width="medium"),
        },
        use_container_width=True,
        hide_index=True,
    )

    # 3. DETAILED CANDIDATE CARDS
    with st.expander("🔍 Detailed Candidate Breakdown & Justification", expanded=True):
        for idx, cand in enumerate(candidates, 1):
            plate = cand.get("complete_plate", "")
            conf_pct = cand.get("match_confidence_pct", "")
            v_class = cand.get("vehicle_class", "Private Vehicle")
            rto_loc_str = cand.get("rto_area", "")
            expl = cand.get("explanation", "")
            v_status = cand.get("validation_status", "")
            is_truth = cand.get("is_verified_truth", False)
            v_details = cand.get("vehicle_details", {})
            pred_chars = cand.get("predicted_chars", {})
            chars_badge = " · ".join([f"Slot {k} ➔ <b>'{v}'</b>" for k, v in pred_chars.items()]) if pred_chars else "Exact match"

            # Formatted vehicle category with icon
            if "four-wheeler" in v_class.lower() or "car" in v_class.lower():
                cat_tag = f"🚗 {v_class}"
            elif "two-wheeler" in v_class.lower() or "motorcycle" in v_class.lower() or "scooter" in v_class.lower():
                cat_tag = f"🛵 {v_class}"
            elif "commercial" in v_class.lower() or "transport" in v_class.lower():
                cat_tag = f"🚕 {v_class}"
            else:
                cat_tag = f"🚘 {v_class}"

            truth_badge = (
                '<span style="background: rgba(16, 185, 129, 0.22); color: #34d399; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; border: 1px solid #10b981;">✅ Verified RTO Ground Truth</span>'
                if is_truth
                else '<span style="background: rgba(59, 130, 246, 0.15); color: #93c5fd; padding: 2px 8px; border-radius: 4px; font-size: 11px;">🔮 Predicted from RTO Series</span>'
            )

            details_html = ""
            if is_truth and v_details:
                maker = v_details.get("maker_model", "")
                fuel = v_details.get("fuel_type", "")
                details_html = f"""
                <div style="margin-top: 6px; padding: 6px 12px; background: rgba(16, 185, 129, 0.08); border-radius: 6px; border-left: 3px solid #10b981; font-size: 12px; color: #a7f3d0;">
                    <b>Official Vehicle Spec:</b> {maker} &nbsp;|&nbsp; <b>Fuel:</b> {fuel} &nbsp;|&nbsp; <b>Category:</b> {v_class}
                </div>
                """

            st.markdown(
                f"""
                <div style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.12); border-radius: 8px; padding: 12px 16px; margin-bottom: 8px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div style="display: flex; align-items: center; gap: 10px; flex-wrap: wrap;">
                            <span style="color: #94a3b8; font-weight: bold;">#{idx}</span>
                            <code style="font-size: 20px; font-weight: bold; color: #38bdf8; letter-spacing: 1.5px;">{plate}</code>
                            <span style="background: rgba(59, 130, 246, 0.2); color: #93c5fd; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600;">{cat_tag}</span>
                            {truth_badge}
                            <span style="background: rgba(16, 185, 129, 0.15); color: #34d399; padding: 2px 8px; border-radius: 4px; font-size: 11px;">{v_status}</span>
                        </div>
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <span style="font-size: 12px; color: #94a3b8;">Match Confidence Score:</span>
                            <span style="font-size: 16px; font-weight: bold; color: #34d399;">{conf_pct}</span>
                        </div>
                    </div>
                    {details_html}
                    <div style="margin-top: 6px; font-size: 12px; color: #cbd5e1;">
                        <b>Local Jurisdiction:</b> {rto_loc_str} &nbsp;|&nbsp; <b>Predicted Characters:</b> {chars_badge}
                    </div>
                    <div style="margin-top: 4px; font-size: 12px; color: #94a3b8;">
                        <b>Explanation:</b> {expl}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# Tri-mode tabs
tab_upload, tab_manual, tab_predictor = st.tabs([
    "📸 Vehicle Photo Analysis & OCR",
    "⌨️ Check by Vehicle Number",
    "🔍 RTO-Based Missing Character Predictor",
])

# ==============================================================================
# TAB 1: VEHICLE PHOTO ANALYSIS & OCR
# ==============================================================================
with tab_upload:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("1. Upload Vehicle Photo")
        uploaded = st.file_uploader(
            "Upload car, scooter, or motorcycle photo",
            type=["jpg", "jpeg", "png", "webp"],
            key="upload_file_widget",
        )
        if uploaded:
            st.image(uploaded, caption="Input Vehicle Photo", use_container_width=True)
            if st.session_state.current_file_name != uploaded.name:
                st.session_state.detection_data = None
                st.session_state.challan_check_result = None
                st.session_state.current_file_name = uploaded.name

    with col2:
        st.subheader("2. Detection & e-Challan Verification")
        if uploaded:
            if st.button("🚀 Run AI Plate Detection", type="primary", use_container_width=True):
                with st.spinner("Localizing plate, extracting characters & analyzing tampering..."):
                    files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
                    try:
                        response = requests.post(f"{api_url}/api/detect", files=files, timeout=120)
                        response.raise_for_status()
                        st.session_state.detection_data = response.json()
                        st.session_state.challan_check_result = None
                    except requests.RequestException as error:
                        st.error(f"API request failed: {error}")
                        st.info("Make sure the backend API is running on port 8000.")

        data = st.session_state.detection_data
        if data:
            verdict_color = "red" if data["is_tampered"] else "green"
            st.markdown(f"### :{verdict_color}[{data['verdict']}]")

            m_col1, m_col2 = st.columns(2)
            with m_col1:
                st.metric("Tamper Score", f"{data['tamper_score']:.2%}")
            with m_col2:
                st.metric("OCR Confidence", f"{data['ocr_confidence']:.2%}")

            detected_text = data.get("plate_text") or ""

            # ------------------------------------------------------------------
            # DETECTED VEHICLE NUMBER + EDIT/CONFIRM + CHECK E-CHALLAN BUTTON
            # ------------------------------------------------------------------
            st.write("---")
            st.write("#### Detected Vehicle Number")
            plate_input_col, check_btn_col = st.columns([2, 1])

            with plate_input_col:
                confirmed_plate = st.text_input(
                    "Confirm or Edit Vehicle Registration Number",
                    value=detected_text,
                    help="Verify or edit the registration number before running e-Challan check",
                    key="ocr_confirmed_plate_input",
                )

            with check_btn_col:
                st.write("")
                st.write("")
                check_challan_clicked = st.button("🔍 Check e-Challan", type="primary", use_container_width=True)

            if check_challan_clicked:
                clean_num = confirmed_plate.strip()
                if not clean_num:
                    st.error("Please enter a valid vehicle registration number.")
                else:
                    with st.spinner("Checking e-Challan records..."):
                        try:
                            st.session_state.challan_check_result = challan_service.get_challans(clean_num)
                        except ValueError as val_err:
                            st.error(f"⚠️ {val_err}")
                        except Exception as exc:
                            st.error(f"⚠️ Challan lookup error: {exc}")

            # Display e-Challan Results
            if st.session_state.challan_check_result:
                st.write("---")
                render_challan_display(st.session_state.challan_check_result, current_plate_num=confirmed_plate)

            st.write("---")
            st.write(f"**Format Status:** {data['format_message']}")
            st.write(f"**Processing Time:** {data['processing_ms']} ms")

            # Localized Plate Crop
            if data.get("annotated_image_url"):
                img_response = requests.get(f"{api_url}{data['annotated_image_url']}", timeout=30)
                annotated = Image.open(io.BytesIO(img_response.content))
                st.image(annotated, caption="Localized Number Plate Region", use_container_width=True)

            # Forensic Signals
            st.subheader("Forensic Tampering Signals")
            for signal in data["signals"]:
                st.progress(min(max(signal["score"], 0.0), 1.0), text=f"{signal['name']} ({signal['severity']})")
                st.caption(signal["description"])


# ==============================================================================
# TAB 2: MANUAL VEHICLE NUMBER SEARCH
# ==============================================================================
with tab_manual:
    st.subheader("Direct e-Challan Check by Vehicle Number")
    st.write("Verify traffic citations directly by entering any Indian vehicle registration number.")

    m_col1, m_col2 = st.columns([2, 1])
    with m_col1:
        manual_plate_input = st.text_input(
            "Enter Vehicle Number (e.g. MH46BW1612, DL2SKA2187, KA01AB1234)",
            value="MH46BW1612",
            key="tab2_manual_plate_box",
        )
    with m_col2:
        st.write("")
        st.write("")
        manual_check_btn = st.button("🔍 Check e-Challan", type="primary", use_container_width=True, key="manual_check_btn")

    if manual_check_btn or st.session_state.manual_challan_result:
        if manual_check_btn:
            clean_manual = manual_plate_input.strip()
            if not clean_manual:
                st.error("Please enter a valid vehicle registration number.")
            else:
                with st.spinner("Checking e-Challan records..."):
                    try:
                        st.session_state.manual_challan_result = challan_service.get_challans(clean_manual)
                    except ValueError as val_err:
                        st.error(f"⚠️ {val_err}")
                    except Exception as err:
                        st.error(f"⚠️ Service request failed: {err}")

        if st.session_state.manual_challan_result:
            render_challan_display(st.session_state.manual_challan_result, current_plate_num=manual_plate_input)


# ==============================================================================
# TAB 3: RTO-BASED MISSING CHARACTER PREDICTOR
# ==============================================================================
with tab_predictor:
    st.subheader("🔍 RTO-Based Missing Character Predictor")
    st.write(
        "Predict missing characters in damaged, occluded, or tampered Indian vehicle license plates. "
        "The model identifies the State & RTO geography, evaluates local vehicle series constraints, "
        "enforces MoRTH syntax rules, and generates syntactically and geographically valid complete plate candidates."
    )

    # Preset test examples
    st.write("##### 🧪 Quick Test Presets")
    p_col1, p_col2, p_col3, p_col4, p_col5 = st.columns(5)
    with p_col1:
        if st.button("📍 MH 46 B* 161*", use_container_width=True, help="Panvel RTO: missing series letter and last digit"):
            st.session_state.prediction_input_text = "MH 46 B* 161*"
            st.session_state.prediction_result = None
    with p_col2:
        if st.button("📍 MH 46 ** 1612", use_container_width=True, help="Panvel RTO: missing two-letter series"):
            st.session_state.prediction_input_text = "MH 46 ** 1612"
            st.session_state.prediction_result = None
    with p_col3:
        if st.button("📍 MH 08 A* 1400", use_container_width=True, help="Ratnagiri RTO: missing series letter"):
            st.session_state.prediction_input_text = "MH 08 A* 1400"
            st.session_state.prediction_result = None
    with p_col4:
        if st.button("📍 DL 01 C* 5678", use_container_width=True, help="Delhi Mall Road RTO: 4-wheeler series"):
            st.session_state.prediction_input_text = "DL 01 C* 5678"
            st.session_state.prediction_result = None
    with p_col5:
        if st.button("📍 KA 03 M* 9999", use_container_width=True, help="Bangalore East RTO: 2-wheeler series"):
            st.session_state.prediction_input_text = "KA 03 M* 9999"
            st.session_state.prediction_result = None

    input_col, run_col = st.columns([3, 1])
    with input_col:
        pred_text_input = st.text_input(
            "Partial Indian Vehicle Registration Number (use '*' or '?' for missing slots)",
            value=st.session_state.prediction_input_text,
            key="prediction_input_text_widget",
            help="Enter partial plate with wildcards, e.g. 'MH 46 B* 161*' or 'MH 46 ** 1612'",
        )
    with run_col:
        st.write("")
        st.write("")
        run_pred_btn = st.button("🔮 Predict Missing Characters", type="primary", use_container_width=True, key="run_predictor_btn")

    filter_col1, filter_col2 = st.columns([1.8, 1.2])
    with filter_col1:
        top_k_slider = st.slider("Max Candidates to Return", min_value=3, max_value=20, value=10, step=1)
    with filter_col2:
        category_filter = st.selectbox(
            "Vehicle Category Filter",
            options=[
                "All Categories (Auto-Detect Truth)",
                "🚗 Four-Wheeler (Car / LMV)",
                "🛵 Two-Wheeler (Motorcycle / Scooter)",
                "🚕 Commercial / Transport",
            ],
            index=0,
            help="Filter predictions by vehicle type or let the system auto-detect ground truth.",
        )

    target_cat_param = None
    if "Four-Wheeler" in category_filter:
        target_cat_param = "Four-Wheeler"
    elif "Two-Wheeler" in category_filter:
        target_cat_param = "Two-Wheeler"
    elif "Commercial" in category_filter:
        target_cat_param = "Commercial"

    if run_pred_btn or st.session_state.prediction_result is not None:
        if run_pred_btn:
            clean_input = pred_text_input.strip()
            if not clean_input:
                st.error("Please provide a partial plate string to predict.")
            else:
                with st.spinner("Extracting RTO geography, evaluating series constraints & predicting candidates..."):
                    try:
                        p_resp = requests.post(
                            f"{api_url}/api/rto-predictor/predict",
                            json={
                                "partial_plate": clean_input,
                                "top_k": top_k_slider,
                                "target_category": target_cat_param,
                            },
                            timeout=15,
                        )
                        p_resp.raise_for_status()
                        st.session_state.prediction_result = p_resp.json()
                    except Exception:
                        import importlib
                        import ml.schemas
                        import ml.rto_predictor
                        importlib.reload(ml.schemas)
                        importlib.reload(ml.rto_predictor)
                        local_res = ml.rto_predictor.rto_predictor.predict(
                            clean_input,
                            top_k=top_k_slider,
                            target_category=target_cat_param,
                        )
                        st.session_state.prediction_result = {
                            "input_pattern": local_res.input_pattern,
                            "normalized_pattern": local_res.normalized_pattern,
                            "state_code": local_res.state_code,
                            "rto_code": local_res.rto_code,
                            "rto_area_info": {
                                "rto_code": local_res.rto_area_info.rto_code,
                                "prefix": local_res.rto_area_info.prefix,
                                "state_code": local_res.rto_area_info.state_code,
                                "state_name": local_res.rto_area_info.state_name,
                                "location": local_res.rto_area_info.location,
                                "district": local_res.rto_area_info.district,
                                "region_zone": local_res.rto_area_info.region_zone,
                                "allowed_series": local_res.rto_area_info.allowed_series,
                                "prohibited_letters": local_res.rto_area_info.prohibited_letters,
                                "standard_format": local_res.rto_area_info.standard_format,
                            } if local_res.rto_area_info else None,
                            "candidates": [
                                {
                                    "complete_plate": c.complete_plate,
                                    "raw_plate": c.raw_plate,
                                    "match_confidence": c.match_confidence,
                                    "match_confidence_pct": c.match_confidence_pct,
                                    "predicted_chars": c.predicted_chars,
                                    "vehicle_class": c.vehicle_class,
                                    "rto_area": c.rto_area,
                                    "validation_status": c.validation_status,
                                    "explanation": c.explanation,
                                    "is_verified_truth": getattr(c, "is_verified_truth", False),
                                    "vehicle_details": getattr(c, "vehicle_details", {}),
                                }
                                for c in local_res.candidates
                            ],
                            "total_candidates_found": local_res.total_candidates_found,
                            "validation_notes": local_res.validation_notes,
                            "status": local_res.status,
                            "error_message": local_res.error_message,
                        }

        if st.session_state.prediction_result:
            render_rto_prediction_results(st.session_state.prediction_result)
