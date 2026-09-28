"""Configurable RTO (Regional Transport Office) prefix lookup service.

Loads RTO mappings dynamically from JSON or CSV files without hardcoded assumptions.
"""

from __future__ import annotations

import csv
import json
import logging
import re
from pathlib import Path

from config.settings import settings
from ml.schemas import RTOResult

logger = logging.getLogger(__name__)


def normalize_prefix(raw_prefix: str) -> str:
    """Normalize prefix into uppercase alphanumeric format (e.g. 'MH-46' -> 'MH46')."""
    return re.sub(r"[^A-Z0-9]", "", (raw_prefix or "").upper())


def format_rto_code(state_code: str, rto_num: str) -> str:
    """Format into standard hyphenated code (e.g. 'MH-46')."""
    return f"{state_code.upper()}-{rto_num.upper()}"


class RTOService:
    """Service to load, cache, and query RTO prefix mappings."""

    def __init__(self, mapping_path: Path | str | None = None) -> None:
        self.mapping_path = Path(mapping_path) if mapping_path else settings.rto_mapping_path
        self._records: dict[str, RTOResult] = {}
        self._state_index: dict[str, list[str]] = {}
        self.load_mapping(self.mapping_path)

    def load_mapping(self, path: Path | str) -> None:
        """Load RTO records from a JSON or CSV file."""
        file_path = Path(path)
        if not file_path.exists():
            logger.warning(f"RTO mapping file not found at: {file_path}. Using empty dataset.")
            self._records = {}
            self._state_index = {}
            return

        self.mapping_path = file_path
        self._records = {}
        self._state_index = {}

        if file_path.suffix.lower() == ".json":
            self._load_json(file_path)
        elif file_path.suffix.lower() == ".csv":
            self._load_csv(file_path)
        else:
            raise ValueError(f"Unsupported file format for RTO mapping: {file_path.suffix}. Use .json or .csv")

        logger.info(f"Loaded {len(self._records)} RTO prefix mappings from {file_path}")

    def _load_json(self, file_path: Path) -> None:
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        items = data.get("records", data) if isinstance(data, dict) else data
        for item in items:
            self._add_record(
                code=item.get("code", ""),
                prefix=item.get("prefix", ""),
                state_code=item.get("state_code", ""),
                state_name=item.get("state_name", ""),
                location=item.get("location", ""),
                district=item.get("district", ""),
                region_zone=item.get("region_zone", ""),
            )

    def _load_csv(self, file_path: Path) -> None:
        with file_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self._add_record(
                    code=row.get("code", ""),
                    prefix=row.get("prefix", ""),
                    state_code=row.get("state_code", ""),
                    state_name=row.get("state_name", ""),
                    location=row.get("location", ""),
                    district=row.get("district", ""),
                    region_zone=row.get("region_zone", ""),
                )

    def _add_record(
        self,
        code: str,
        prefix: str,
        state_code: str,
        state_name: str,
        location: str,
        district: str,
        region_zone: str = "",
    ) -> None:
        norm_prefix = normalize_prefix(prefix or code)
        if not norm_prefix:
            return

        norm_code = code.strip() if code else norm_prefix
        if "-" not in norm_code and len(norm_prefix) >= 3:
            norm_code = f"{norm_prefix[:2]}-{norm_prefix[2:]}"

        record = RTOResult(
            prefix=norm_prefix,
            code=norm_code,
            state_code=state_code.strip().upper(),
            state_name=state_name.strip(),
            location=location.strip(),
            district=district.strip(),
            is_known=True,
            region_zone=region_zone.strip(),
        )

        # Store primary key and aliases
        self._records[norm_prefix] = record
        if norm_code:
            self._records[norm_code.upper()] = record

        # Add to state index
        st = record.state_code
        if st:
            if st not in self._state_index:
                self._state_index[st] = []
            if norm_prefix not in self._state_index[st]:
                self._state_index[st].append(norm_prefix)

    def lookup(self, prefix_or_code: str) -> RTOResult | None:
        """Find RTO result by code or prefix (e.g. 'MH-46', 'MH46', 'mh 46')."""
        norm = normalize_prefix(prefix_or_code)
        if norm in self._records:
            return self._records[norm]

        # Try matching direct string
        clean_upper = (prefix_or_code or "").strip().upper()
        if clean_upper in self._records:
            return self._records[clean_upper]

        return None

    def extract_rto_prefix(self, plate_text: str) -> tuple[str | None, RTOResult | None]:
        """
        Extract the visible RTO prefix from the beginning of a plate string.
        Matches 4-character prefix (e.g. MH46, MH08, DL01) or 3-character prefix (e.g. DL1).
        """
        cleaned = normalize_prefix(plate_text)
        if len(cleaned) < 3:
            return None, None

        # Check 4-character prefix (State 2 letters + 2 digits, e.g. MH46)
        if len(cleaned) >= 4:
            p4 = cleaned[:4]
            res4 = self.lookup(p4)
            if res4:
                return p4, res4

        # Check 3-character prefix (State 2 letters + 1 digit, e.g. DL1)
        p3 = cleaned[:3]
        res3 = self.lookup(p3)
        if res3:
            return p3, res3

        # If prefix is not in database, but follows [A-Z]{2}[0-9]{1,2}, return an unlisted RTO
        match = re.match(r"^([A-Z]{2}[0-9]{1,2})", cleaned)
        if match:
            raw_p = match.group(1)
            state = raw_p[:2]
            rto_num = raw_p[2:]
            return raw_p, RTOResult(
                prefix=raw_p,
                code=f"{state}-{rto_num}",
                state_code=state,
                state_name=f"State {state}",
                location=f"Unregistered / Unknown RTO ({raw_p})",
                district="Unknown",
                is_known=False,
            )

        return None, None

    def get_known_prefixes_for_state(self, state_code: str) -> list[str]:
        """Return all known RTO prefixes for a given state code."""
        return self._state_index.get((state_code or "").upper(), [])

    @property
    def total_records(self) -> int:
        return len(self._state_index)


rto_service = RTOService()
