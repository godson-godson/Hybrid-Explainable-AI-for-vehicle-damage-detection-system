"""
Document Understanding (OCR) Module — Phase 5 Revision.
Implements a deterministic two-stage pipeline:
  Stage 1: Local PaddleOCR for text detection & recognition (CPU-based).
  Stage 2: Groq Text Model (llama-3.3-70b-versatile) for semantic schema mapping.
  Post-Validation: Regex normalization (VIN, vehicle registration, loose policy/DL heuristic).
  Fallback: Focused horizontal-band Groq Vision fallback for low-yield OCR (<40 chars).

Adheres strictly to Section 3.2.8 of the project report and anti-fabrication guidelines.
"""

import io
import json
import logging
import os
import re
import subprocess
import tempfile
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from ..models.schemas import (
    DocumentExtractionResult,
    DocumentFields,  # Legacy stub compatibility
    ExtractedFieldItem,
)
from .groq_client import GroqClientError, get_groq_client
from .mock_responses import get_mock_document_response, get_mock_ocr_corpus

logger = logging.getLogger("document_ocr")

# Minimum characters in extracted corpus before triggering low-yield vision fallback
LOW_YIELD_CHAR_THRESHOLD = 40

# Paths for dedicated PaddleOCR worker
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_BASE_DIR, "../../../"))
PADDLE_PYTHON = os.path.join(PROJECT_ROOT, ".venv_paddle/bin/python")
PADDLE_WORKER_SCRIPT = os.path.join(_BASE_DIR, "paddle_ocr_worker.py")


# ---------------------------------------------------------------------------
# Stage 2 Schema Prompts for Groq Text Model (llama-3.3-70b-versatile)
# ---------------------------------------------------------------------------

PROMPT_COMMON_DIRECTIVE = """
You are an expert document entity parser.
You are given raw OCR text extracted from an Indian {doc_type}.
Extract target values into exact JSON format matching the schema below.
If a field is not found in the text, return null for it.
Do not guess, infer, or fabricate a plausible-looking value for any field that isn't clearly present in the OCR text.

Output ONLY valid, raw JSON. Do not include markdown formatting, code blocks (```json), or any conversational text before or after the JSON object.
For each extracted field, return an object containing:
  "value": Extracted string value (or null if absent/not present),
  "source_snippet": Exact text snippet from the raw OCR corpus where this was found (or null),
  "confidence": Float confidence between 0.0 and 1.0 reflecting clarity.
"""

SCHEMA_FIELDS_RC = """
Target Schema for Registration Certificate (RC Book):
{
  "registration_number": {"value": string or null, "source_snippet": string or null, "confidence": float},
  "owner_name": {"value": string or null, "source_snippet": string or null, "confidence": float},
  "vehicle_make_model": {"value": string or null, "source_snippet": string or null, "confidence": float},
  "chassis_number": {"value": string or null, "source_snippet": string or null, "confidence": float},
  "engine_number": {"value": string or null, "source_snippet": string or null, "confidence": float},
  "registration_date": {"value": string or null, "source_snippet": string or null, "confidence": float},
  "fuel_type": {"value": string or null, "source_snippet": string or null, "confidence": float},
  "vehicle_class": {"value": string or null, "source_snippet": string or null, "confidence": float}
}
"""

SCHEMA_FIELDS_POLICY = """
Target Schema for Motor Insurance Policy:
{
  "policy_number": {"value": string or null, "source_snippet": string or null, "confidence": float},
  "insurer_name": {"value": string or null, "source_snippet": string or null, "confidence": float},
  "policy_start_date": {"value": string or null, "source_snippet": string or null, "confidence": float},
  "policy_expiry_date": {"value": string or null, "source_snippet": string or null, "confidence": float},
  "insured_name": {"value": string or null, "source_snippet": string or null, "confidence": float},
  "vehicle_reg_number": {"value": string or null, "source_snippet": string or null, "confidence": float},
  "insured_declared_value": {"value": string or null, "source_snippet": string or null, "confidence": float},
  "coverage_type": {"value": string or null, "source_snippet": string or null, "confidence": float}
}
"""

SCHEMA_FIELDS_LICENCE = """
Target Schema for Driving Licence:
{
  "licence_number": {"value": string or null, "source_snippet": string or null, "confidence": float},
  "holder_name": {"value": string or null, "source_snippet": string or null, "confidence": float},
  "validity_date": {"value": string or null, "source_snippet": string or null, "confidence": float},
  "vehicle_classes_authorized": {"value": string or null, "source_snippet": string or null, "confidence": float},
  "issue_date": {"value": string or null, "source_snippet": string or null, "confidence": float}
}
"""


def _get_system_prompt_for_type(document_type: str) -> str:
    """Compose semantic mapping system prompt for the specified document type."""
    dt = document_type.lower()
    if "rc" in dt or "reg" in dt:
        doc_label = "Registration Certificate (RC Book)"
        schema = SCHEMA_FIELDS_RC
    elif "policy" in dt or "insur" in dt:
        doc_label = "Motor Insurance Policy"
        schema = SCHEMA_FIELDS_POLICY
    elif "licence" in dt or "license" in dt or "dl" in dt:
        doc_label = "Driving Licence"
        schema = SCHEMA_FIELDS_LICENCE
    else:
        doc_label = "Vehicle Document"
        schema = SCHEMA_FIELDS_RC

    directive = PROMPT_COMMON_DIRECTIVE.replace("{doc_type}", doc_label)
    return f"{directive}\n{schema}"


# ---------------------------------------------------------------------------
# JSON Bracket-Stripping Parser
# ---------------------------------------------------------------------------

def _strip_and_parse_json(raw_text: str) -> Optional[Dict[str, Any]]:
    """
    Regex / bracket-stripping to extract valid JSON between first '{' and last '}'
    and remove conversational padding or markdown code fences.
    """
    if not raw_text or not raw_text.strip():
        return None

    cleaned = raw_text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")

    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        json_candidate = cleaned[first_brace:last_brace + 1]
    else:
        json_candidate = cleaned

    try:
        return json.loads(json_candidate)
    except json.JSONDecodeError as e:
        logger.warning(f"[!] JSON parsing failed: {e}. Snippet: {cleaned[:120]}")
        return None


# ---------------------------------------------------------------------------
# Step 1: Stage 1 Deterministic Text Extraction (PaddleOCR)
# ---------------------------------------------------------------------------

def _run_paddle_ocr(
    image_path: str,
    document_type: str = "rc_book",
    allow_mock: bool = False,
) -> Dict[str, Any]:
    """
    Run PaddleOCR in the dedicated .venv_paddle environment.
    Only falls back to deterministic mock OCR corpus if explicitly allowed (e.g. unit tests).
    Real surveyor uploads never fall back to mock data.
    """
    t0 = time.perf_counter()
    if not os.path.exists(PADDLE_PYTHON):
        logger.error(f"[!] PaddleOCR environment not found at {PADDLE_PYTHON}")
        if allow_mock:
            logger.info("[*] Using deterministic mock PaddleOCR text extraction (mock allowed).")
            mock_data = get_mock_ocr_corpus(document_type)
            return {
                "status": "SUCCESS",
                "lines": mock_data["lines"],
                "scores": mock_data["scores"],
                "corpus": mock_data["corpus"],
                "char_count": mock_data["char_count"],
                "latency_ms": 15.0,
            }
        return {
            "status": "FAILED",
            "error": f"PaddleOCR Python executable not found at {PADDLE_PYTHON}",
            "lines": [],
            "scores": [],
            "corpus": "",
            "char_count": 0,
            "latency_ms": 0.0,
        }

    try:
        proc = subprocess.run(
            [PADDLE_PYTHON, PADDLE_WORKER_SCRIPT, image_path],
            capture_output=True,
            text=True,
            timeout=150,
        )
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        if proc.returncode == 0:
            stdout_str = proc.stdout.strip()
            # Extract JSON object from stdout
            first_brace = stdout_str.find("{")
            last_brace = stdout_str.rfind("}")
            if first_brace != -1 and last_brace != -1:
                json_str = stdout_str[first_brace:last_brace + 1]
                data = json.loads(json_str)
                if "latency_ms" not in data or data["latency_ms"] == 0:
                    data["latency_ms"] = latency_ms
                return data

        logger.error(
            f"[!] PaddleOCR worker exit code {proc.returncode}. Stderr: {proc.stderr[:300]}"
        )
    except Exception as exc:
        logger.error(f"[!] PaddleOCR execution error: {exc}", exc_info=True)

    if allow_mock:
        logger.info("[*] Using mock OCR corpus fallback (mock allowed).")
        mock_data = get_mock_ocr_corpus(document_type)
        return {
            "status": "SUCCESS",
            "lines": mock_data["lines"],
            "scores": mock_data["scores"],
            "corpus": mock_data["corpus"],
            "char_count": mock_data["char_count"],
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        }

    return {
        "status": "FAILED",
        "error": "PaddleOCR execution failed or returned invalid output",
        "lines": [],
        "scores": [],
        "corpus": "",
        "char_count": 0,
        "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
    }


# ---------------------------------------------------------------------------
# Step 2: Post-Validation & Normalization (Regex-based)
# ---------------------------------------------------------------------------

PATTERNS = {
    "registration_number": r"[A-Z]{2}[-\s]?[0-9]{1,2}[-\s]?[A-Z]{1,3}[-\s]?[0-9]{4}",
    "chassis_number": r"[A-HJ-NPR-Z0-9]{17}",  # 17-char VIN
    "policy_or_dl_number": r"POL-[0-9]{5,8}|[A-Z0-9/]{10,25}",
}

DATE_PATTERNS = [
    r"^\d{2}[/-]\d{2}[/-]\d{4}$",
    r"^\d{4}[/-]\d{2}[/-]\d{2}$",
    r"^\d{2}\s+[A-Za-z]{3}\s+\d{4}$",
]


def _validate_date_string(date_str: str) -> bool:
    """Check if date string matches standard calendar format."""
    cleaned = date_str.strip()
    return any(re.match(p, cleaned) for p in DATE_PATTERNS)


def post_validate_and_normalize_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Step 2: Regex-based field normalization pass:
    - Trims and uppercases string fields.
    - Normalizes registration_number format.
    - Cleans chassis_number (17-char VIN); does NOT touch engine_number.
    - Applies loose heuristic check to policy_or_dl_number without discarding value.
    - If a regex fails to match, keeps original LLM value (never nulls out valid values).
    """
    normalized: Dict[str, Any] = {}
    for k, v in data.items():
        if isinstance(v, dict):
            val = v.get("value")
            snippet = v.get("source_snippet")
            conf = float(v.get("confidence", 0.85))
            if isinstance(val, str):
                val_clean = val.strip().upper()
                if not val_clean or val_clean.lower() in ("null", "none"):
                    val = None
                else:
                    val = val_clean
            normalized[k] = {"value": val, "source_snippet": snippet, "confidence": conf}
        elif isinstance(v, str):
            clean_v = v.strip().upper()
            if not clean_v or clean_v.lower() in ("null", "none"):
                clean_v = None
            normalized[k] = {"value": clean_v, "source_snippet": None, "confidence": 0.85}
        else:
            normalized[k] = {"value": v, "source_snippet": None, "confidence": 0.85}

    # 1. Registration Number regex extraction & normalization
    for reg_key in ("registration_number", "vehicle_reg_number"):
        reg_entry = normalized.get(reg_key)
        if reg_entry and reg_entry.get("value"):
            reg_val = str(reg_entry["value"]).strip()
            match = re.search(PATTERNS["registration_number"], reg_val)
            if match:
                normalized[reg_key]["value"] = match.group(0).strip()
            # If no regex match, keep original LLM value

    # 2. Chassis number 17-char VIN cleaning (do NOT apply to engine_number!)
    chassis_entry = normalized.get("chassis_number")
    if chassis_entry and chassis_entry.get("value"):
        cleaned_vin = re.sub(r"[^A-Z0-9]", "", str(chassis_entry["value"]))
        if len(cleaned_vin) == 17:
            chassis_entry["value"] = cleaned_vin
        # If not 17 chars, keep original LLM value

    # 3. Policy / DL number sanity heuristic (loose heuristic, keep value regardless)
    for pol_key in ("policy_number", "licence_number"):
        pol_entry = normalized.get(pol_key)
        if pol_entry and pol_entry.get("value"):
            pol_val = str(pol_entry["value"]).strip()
            # Loose heuristic check: keep value regardless
            _ = re.search(PATTERNS["policy_or_dl_number"], pol_val)

    return normalized


def _validate_extracted_fields(
    document_type: str,
    raw_fields: Dict[str, Any],
) -> Tuple[Dict[str, ExtractedFieldItem], bool, List[str]]:
    """
    Perform heuristic presence and sanity checks independent of the LLM.
    Flags needs_manual_review if critical required fields are absent or malformed.
    """
    # First run the Step 2 post-validation normalization
    normalized_data = post_validate_and_normalize_fields(raw_fields)

    validated_items: Dict[str, ExtractedFieldItem] = {}
    warnings: List[str] = []
    needs_review = False

    dt = document_type.lower()
    is_rc = "rc" in dt or "reg" in dt
    is_policy = "policy" in dt or "insur" in dt
    is_dl = "licence" in dt or "license" in dt or "dl" in dt

    for field_name, field_dict in normalized_data.items():
        val = field_dict.get("value")
        snippet = field_dict.get("source_snippet")
        conf = float(field_dict.get("confidence", 0.85))

        item = ExtractedFieldItem(
            value=val,
            source_snippet=str(snippet).strip() if snippet else None,
            confidence=conf,
            is_valid=True,
            validation_error=None,
        )
        validated_items[field_name] = item

    # Required field verification per document type
    if is_rc:
        reg_item = validated_items.get("registration_number")
        if not reg_item or not reg_item.value:
            warnings.append("Registration Certificate missing registration number.")
            needs_review = True
            if reg_item:
                reg_item.is_valid = False
                reg_item.validation_error = "Required registration number missing"
        else:
            clean_reg = reg_item.value.replace(" ", "").replace("-", "")
            if not re.match(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{4}$", clean_reg):
                warnings.append(f"Registration number '{reg_item.value}' irregular format.")
                needs_review = True
                reg_item.is_valid = False
                reg_item.validation_error = "Irregular registration number format"

        chassis_item = validated_items.get("chassis_number")
        if chassis_item and chassis_item.value:
            clean_chassis = chassis_item.value.replace(" ", "")
            if len(clean_chassis) < 10 or len(clean_chassis) > 20:
                warnings.append(f"Chassis number length ({len(clean_chassis)}) appears irregular.")
                chassis_item.is_valid = False
                chassis_item.validation_error = "Plausible chassis length warning"

    elif is_policy:
        pol_item = validated_items.get("policy_number")
        if not pol_item or not pol_item.value:
            warnings.append("Insurance Policy missing policy certificate number.")
            needs_review = True
            if pol_item:
                pol_item.is_valid = False
                pol_item.validation_error = "Required policy number missing"

        exp_item = validated_items.get("policy_expiry_date")
        if exp_item and exp_item.value:
            if not _validate_date_string(exp_item.value):
                warnings.append(f"Policy expiry date '{exp_item.value}' has unrecognized date format.")
                exp_item.is_valid = False
                exp_item.validation_error = "Malformed date format"
                needs_review = True

    elif is_dl:
        dl_item = validated_items.get("licence_number")
        if not dl_item or not dl_item.value:
            warnings.append("Driving Licence missing licence identification number.")
            needs_review = True
            if dl_item:
                dl_item.is_valid = False
                dl_item.validation_error = "Required licence number missing"

        holder_item = validated_items.get("holder_name")
        if not holder_item or not holder_item.value:
            warnings.append("Driving licence holder name was not detected.")
            needs_review = True

    # Check for low optical confidence scores (<0.60)
    for name, item in validated_items.items():
        if item.value and item.confidence < 0.60:
            warnings.append(f"Field '{name}' has low optical confidence ({item.confidence:.2f}).")
            needs_review = True

    return validated_items, needs_review, warnings


# ---------------------------------------------------------------------------
# Deterministic Local Schema Extractor (Offline / Groq-Unset Mode)
# ---------------------------------------------------------------------------

EMPTY_FIELD_NAMES = {
    "rc_book": [
        "registration_number", "owner_name", "vehicle_make_model", "chassis_number",
        "engine_number", "registration_date", "fuel_type", "vehicle_class"
    ],
    "insurance_policy": [
        "policy_number", "insurer_name", "policy_start_date", "policy_expiry_date",
        "insured_name", "vehicle_reg_number", "insured_declared_value", "coverage_type"
    ],
    "driving_licence": [
        "licence_number", "holder_name", "validity_date", "vehicle_classes_authorized", "issue_date"
    ],
}


def _build_empty_document_fields(document_type: str) -> Dict[str, ExtractedFieldItem]:
    """Return an empty ExtractedFieldItem mapping when document extraction fails."""
    dt = document_type.lower()
    if "rc" in dt or "reg" in dt:
        field_keys = EMPTY_FIELD_NAMES["rc_book"]
    elif "policy" in dt or "insur" in dt:
        field_keys = EMPTY_FIELD_NAMES["insurance_policy"]
    elif "licence" in dt or "license" in dt or "dl" in dt:
        field_keys = EMPTY_FIELD_NAMES["driving_licence"]
    else:
        field_keys = EMPTY_FIELD_NAMES["rc_book"]

    return {
        key: ExtractedFieldItem(
            value=None,
            source_snippet=None,
            confidence=0.0,
            is_valid=False,
            validation_error="Extraction failed — manual review required",
        )
        for key in field_keys
    }


def _parse_document_corpus_locally(
    corpus: str,
    lines: List[str],
    document_type: str,
) -> Dict[str, Any]:
    """
    Robust deterministic rule-based entity extractor for Indian KYC documents.
    Extracts schema-compliant fields from the raw OCR corpus when Groq API is offline,
    unset, or rate-limited, ensuring real uploaded document text is never replaced
    with hardcoded placeholder data.
    """
    dt = document_type.lower()
    clean_lines = [l.strip() for l in lines if l.strip()]
    full_text = corpus or "\n".join(clean_lines)
    result: Dict[str, Any] = {}

    if "rc" in dt or "reg" in dt:
        # 1. Registration Number
        reg_match = re.search(r"\b([A-Z]{2}[-\s]?[0-9]{1,2}[-\s]?[A-Z]{1,3}[-\s]?[0-9]{4})\b", full_text)
        reg_val = reg_match.group(1).strip() if reg_match else None
        result["registration_number"] = {
            "value": reg_val,
            "source_snippet": reg_match.group(0) if reg_match else None,
            "confidence": 0.95 if reg_val else 0.0,
        }

        # 2. Owner Name
        owner_val = None
        owner_snip = None
        for i, line in enumerate(clean_lines):
            m = re.search(r"Owner(?:\s*Name)?\s*[:\-]?\s*([A-Za-z\s\.]+)", line, re.IGNORECASE)
            if m and len(m.group(1).strip()) > 2:
                owner_val = m.group(1).strip()
                owner_snip = line
                break
            elif re.match(r"^Owner(?:\s*Name)?\s*[:]?$", line, re.IGNORECASE) and i + 1 < len(clean_lines):
                next_l = clean_lines[i + 1].lstrip(":- ").strip()
                if next_l and not any(k in next_l.lower() for k in ["chassis", "engine", "regn", "date"]):
                    owner_val = next_l
                    owner_snip = f"{line} {next_l}"
                    break
        result["owner_name"] = {
            "value": owner_val,
            "source_snippet": owner_snip,
            "confidence": 0.92 if owner_val else 0.0,
        }

        # 3. Chassis Number (VIN)
        chassis_val = None
        chassis_snip = None
        m_chas = re.search(r"Chassis(?:\s*No)?\s*[:\-]?\s*([A-HJ-NPR-Z0-9]{10,20})", full_text, re.IGNORECASE)
        if m_chas:
            chassis_val = m_chas.group(1).strip()
            chassis_snip = m_chas.group(0)
        else:
            for i, line in enumerate(clean_lines):
                if re.search(r"Chassis", line, re.IGNORECASE) and i + 1 < len(clean_lines):
                    m_next = re.search(r"([A-HJ-NPR-Z0-9]{10,20})", clean_lines[i + 1])
                    if m_next:
                        chassis_val = m_next.group(1).strip()
                        chassis_snip = clean_lines[i + 1]
                        break
        result["chassis_number"] = {
            "value": chassis_val,
            "source_snippet": chassis_snip,
            "confidence": 0.95 if chassis_val else 0.0,
        }

        # 4. Engine Number
        engine_val = None
        engine_snip = None
        m_eng = re.search(r"Engine(?:\s*No)?\s*[:\-]?\s*([A-Z0-9]{6,20})", full_text, re.IGNORECASE)
        if m_eng:
            engine_val = m_eng.group(1).strip()
            engine_snip = m_eng.group(0)
        else:
            for i, line in enumerate(clean_lines):
                if re.search(r"^Engine(?:\s*No)?\s*[:]?$", line, re.IGNORECASE) and i + 1 < len(clean_lines):
                    engine_val = clean_lines[i + 1].lstrip(":- ").strip()
                    engine_snip = f"{line} {engine_val}"
                    break
        result["engine_number"] = {
            "value": engine_val,
            "source_snippet": engine_snip,
            "confidence": 0.92 if engine_val else 0.0,
        }

        # 5. Vehicle Make / Model
        maker_val = ""
        model_val = ""
        maker_m = re.search(r"Maker\s*[:\-]?\s*([^\n]+)", full_text, re.IGNORECASE)
        if maker_m:
            maker_val = maker_m.group(1).strip()
        model_m = re.search(r"Model\s*[:\-]?\s*([^\n]+)", full_text, re.IGNORECASE)
        if model_m:
            model_val = model_m.group(1).strip()

        make_model = f"{maker_val} {model_val}".strip()
        if not make_model:
            mm_m = re.search(r"Make\s*/\s*Model\s*[:\-]?\s*([^\n]+)", full_text, re.IGNORECASE)
            if mm_m:
                make_model = mm_m.group(1).strip()
        result["vehicle_make_model"] = {
            "value": make_model or None,
            "source_snippet": f"Maker: {maker_val}, Model: {model_val}" if make_model else None,
            "confidence": 0.90 if make_model else 0.0,
        }

        # 6. Registration Date
        date_val = None
        date_snip = None
        m_date = re.search(r"(?:Date\s*of\s*Regn|Regn\s*Date)\s*[:\-]?\s*(\d{2}[/-]\d{2}[/-]\d{4})", full_text, re.IGNORECASE)
        if m_date:
            date_val = m_date.group(1).strip()
            date_snip = m_date.group(0)
        else:
            m_gen_date = re.search(r"\b(\d{2}[/-]\d{2}[/-]\d{4})\b", full_text)
            if m_gen_date:
                date_val = m_gen_date.group(1).strip()
                date_snip = m_gen_date.group(0)
        result["registration_date"] = {
            "value": date_val,
            "source_snippet": date_snip,
            "confidence": 0.88 if date_val else 0.0,
        }

        # 7. Fuel Type
        fuel_val = None
        fuel_snip = None
        m_fuel = re.search(r"Fuel(?:\s*Type)?\s*[:\-]?\s*([A-Za-z]+)", full_text, re.IGNORECASE)
        if m_fuel:
            fuel_val = m_fuel.group(1).strip().upper()
            fuel_snip = m_fuel.group(0)
        result["fuel_type"] = {
            "value": fuel_val,
            "source_snippet": fuel_snip,
            "confidence": 0.95 if fuel_val else 0.0,
        }

        # 8. Vehicle Class
        vclass_val = None
        vclass_snip = None
        m_class = re.search(r"(?:Class\s*of\s*Vehicle|Vehicle\s*Class|Class)\s*[:\-]?\s*([A-Za-z0-9\s\(\)]+)", full_text, re.IGNORECASE)
        if m_class:
            vclass_val = m_class.group(1).strip()
            vclass_snip = m_class.group(0)
        elif "LMV" in full_text.upper():
            vclass_val = "LMV (Light Motor Vehicle)"
            vclass_snip = "LMV"
        result["vehicle_class"] = {
            "value": vclass_val,
            "source_snippet": vclass_snip,
            "confidence": 0.90 if vclass_val else 0.0,
        }

    elif "policy" in dt or "insur" in dt:
        # 1. Policy Number
        pol_val = None
        pol_snip = None
        m_pol = re.search(r"Policy\s*(?:Number|No\.?)\s*[:\-]?\s*(POL-[0-9]{4,8}|[A-Z0-9/]{6,25})", full_text, re.IGNORECASE)
        if m_pol:
            pol_val = m_pol.group(1).strip()
            pol_snip = m_pol.group(0)
        result["policy_number"] = {
            "value": pol_val,
            "source_snippet": pol_snip,
            "confidence": 0.95 if pol_val else 0.0,
        }

        # 2. Insurer Name
        ins_val = None
        ins_snip = None
        for line in clean_lines[:6]:
            if any(term in line.upper() for term in ["ASSURANCE", "INSURANCE", "GENERAL INSURANCE"]):
                ins_val = line
                ins_snip = line
                break
        result["insurer_name"] = {
            "value": ins_val,
            "source_snippet": ins_snip,
            "confidence": 0.95 if ins_val else 0.0,
        }

        # 3. Insured Name
        ins_name_val = None
        ins_name_snip = None
        for i, line in enumerate(clean_lines):
            m = re.search(r"Insured(?:'s)?\s*Name\s*[:\-]?\s*([A-Za-z\s\.]+)", line, re.IGNORECASE)
            if m and len(m.group(1).strip()) > 2:
                ins_name_val = m.group(1).strip()
                ins_name_snip = line
                break
            elif re.match(r"^Insured(?:'s)?\s*Name\s*[:]?$", line, re.IGNORECASE) and i + 1 < len(clean_lines):
                next_l = clean_lines[i + 1].lstrip(":- ").strip()
                if next_l and not any(k in next_l.lower() for k in ["customer", "pan", "address", "policy"]):
                    ins_name_val = next_l
                    ins_name_snip = f"{line} {next_l}"
                    break
        result["insured_name"] = {
            "value": ins_name_val,
            "source_snippet": ins_name_snip,
            "confidence": 0.94 if ins_name_val else 0.0,
        }

        # 4. Vehicle Registration Number
        vreg_val = None
        vreg_snip = None
        m_vreg = re.search(r"(?:Registration\s*(?:No\.?|Number)|Regn\s*No\.?|Vehicle\s*Regn)\s*[:\-]?\s*([A-Z]{2}[-\s]?[0-9]{1,2}[-\s]?[A-Z]{1,3}[-\s]?[0-9]{4})", full_text, re.IGNORECASE)
        if m_vreg:
            vreg_val = m_vreg.group(1).strip()
            vreg_snip = m_vreg.group(0)
        else:
            m_any_reg = re.search(r"\b([A-Z]{2}[-\s]?[0-9]{1,2}[-\s]?[A-Z]{1,3}[-\s]?[0-9]{4})\b", full_text)
            if m_any_reg:
                vreg_val = m_any_reg.group(1).strip()
                vreg_snip = m_any_reg.group(0)
        result["vehicle_reg_number"] = {
            "value": vreg_val,
            "source_snippet": vreg_snip,
            "confidence": 0.95 if vreg_val else 0.0,
        }

        # 5. Policy Dates
        start_date = None
        expiry_date = None
        m_range = re.search(r"Valid\s*from\s*(\d{2}[/-]\d{2}[/-]\d{4})\s*to\s*(\d{2}[/-]\d{2}[/-]\d{4})", full_text, re.IGNORECASE)
        if m_range:
            start_date = m_range.group(1).strip()
            expiry_date = m_range.group(2).strip()
        else:
            m_eff = re.search(r"(?:Effective\s*From|Start\s*Date)\s*[:\-]?\s*(\d{2}[/-]\d{2}[/-]\d{4})", full_text, re.IGNORECASE)
            if m_eff:
                start_date = m_eff.group(1).strip()
            m_exp = re.search(r"(?:Expiry\s*Date|Valid\s*To)\s*[:\-]?\s*(\d{2}[/-]\d{2}[/-]\d{4})", full_text, re.IGNORECASE)
            if m_exp:
                expiry_date = m_exp.group(1).strip()
        result["policy_start_date"] = {
            "value": start_date,
            "source_snippet": m_range.group(0) if m_range else start_date,
            "confidence": 0.90 if start_date else 0.0,
        }
        result["policy_expiry_date"] = {
            "value": expiry_date,
            "source_snippet": m_range.group(0) if m_range else expiry_date,
            "confidence": 0.90 if expiry_date else 0.0,
        }

        # 6. IDV
        idv_val = None
        idv_snip = None
        m_idv = re.search(r"(?:Total\s*Value|IDV|Insured\s*Declared\s*Value)[^\n:]*[:\-]?\s*[₹Rs\.]*\s*([0-9,]+)", full_text, re.IGNORECASE)
        if m_idv:
            idv_clean = re.sub(r"[^0-9]", "", m_idv.group(1))
            if idv_clean and idv_clean != "0":
                idv_val = idv_clean
                idv_snip = m_idv.group(0)
        result["insured_declared_value"] = {
            "value": idv_val,
            "source_snippet": idv_snip,
            "confidence": 0.90 if idv_val else 0.0,
        }

        # 7. Coverage Type
        cov_val = None
        for line in clean_lines:
            if "Policy" in line and any(w in line for w in ["Liability", "Comprehensive", "Package", "Package Policy", "Commercial"]):
                cov_val = line.strip()
                break
        result["coverage_type"] = {
            "value": cov_val,
            "source_snippet": cov_val,
            "confidence": 0.90 if cov_val else 0.0,
        }

    elif "licence" in dt or "license" in dt or "dl" in dt:
        # 1. Licence Number
        lic_val = None
        lic_snip = None
        m_lic = re.search(r"(?:Licence\s*No\.?|DL\s*No\.?|No\s*:)[\s:]*([A-Z0-9\s/]{8,22})", full_text, re.IGNORECASE)
        if m_lic:
            lic_val = m_lic.group(1).split("\n")[0].strip()
            lic_snip = m_lic.group(0).split("\n")[0].strip()
        result["licence_number"] = {
            "value": lic_val,
            "source_snippet": lic_snip,
            "confidence": 0.95 if lic_val else 0.0,
        }

        # 2. Holder Name
        holder_val = None
        holder_snip = None
        for i, line in enumerate(clean_lines):
            m = re.search(r"^Name\s*[:\-]?\s*([A-Za-z\s\.]+)", line, re.IGNORECASE)
            if m and len(m.group(1).strip()) > 2:
                holder_val = m.group(1).strip()
                holder_snip = line
                break
            elif re.match(r"^Name\s*[:]?$", line, re.IGNORECASE) and i + 1 < len(clean_lines):
                next_l = clean_lines[i + 1].lstrip(":- ").strip()
                if next_l and not any(k in next_l.lower() for k in ["s/w/d", "dob", "address", "valid"]):
                    holder_val = next_l
                    holder_snip = f"{line} {next_l}"
                    break
        result["holder_name"] = {
            "value": holder_val,
            "source_snippet": holder_snip,
            "confidence": 0.94 if holder_val else 0.0,
        }

        # 3. Validity Date
        val_date = None
        val_snip = None
        m_val = re.search(r"Valid\s+To[^\n:]*[:\s]*(\d{2}[/-]\d{2}[/-]\d{4})", full_text, re.IGNORECASE)
        if m_val:
            val_date = m_val.group(1).strip()
            val_snip = m_val.group(0)
        result["validity_date"] = {
            "value": val_date,
            "source_snippet": val_snip,
            "confidence": 0.90 if val_date else 0.0,
        }

        # 4. Issue Date
        iss_date = None
        iss_snip = None
        m_iss = re.search(r"(?:Issue\s*Date|Date\s*:)[\s:]*(\d{2}[/-]\d{2}[/-]\d{4})", full_text, re.IGNORECASE)
        if m_iss:
            iss_date = m_iss.group(1).strip()
            iss_snip = m_iss.group(0)
        result["issue_date"] = {
            "value": iss_date,
            "source_snippet": iss_snip,
            "confidence": 0.88 if iss_date else 0.0,
        }

        # 5. Vehicle Classes Authorized
        vclasses = None
        vclass_snip = None
        m_vc = re.search(r"Vehicle\s*Class(?:es)?[\s:]*([A-Z0-9,\s]+)", full_text, re.IGNORECASE)
        if m_vc:
            vclasses = m_vc.group(1).split("\n")[0].strip()
            vclass_snip = m_vc.group(0).split("\n")[0].strip()
        elif any(c in full_text for c in ["LMV", "MCWG"]):
            matches = [c for c in ["MCWG", "LMV", "TRANS", "3W-CAB"] if c in full_text]
            if matches:
                vclasses = ", ".join(matches)
                vclass_snip = vclasses
        result["vehicle_classes_authorized"] = {
            "value": vclasses,
            "source_snippet": vclass_snip,
            "confidence": 0.90 if vclasses else 0.0,
        }

    return result


# ---------------------------------------------------------------------------
# Step 3: Low-Yield Horizontal-Band Vision Fallback
# ---------------------------------------------------------------------------

def _run_low_yield_vision_fallback(
    image_bytes: bytes,
    document_type: str,
    mime_type: str = "image/jpeg",
    allow_mock: bool = False,
) -> Tuple[Dict[str, Any], float]:
    """
    Step 3: If PaddleOCR extraction yields very low character density (<40 chars),
    crop the document into 3 focused horizontal bands:
      - Band 1 (Top 0% - 28%): Header, document type, policy or reg number
      - Band 2 (Mid 25% - 50%): Insured / holder name and personal details
      - Band 3 (Lower 45% - 75%): Vehicle specifications, chassis, engine, dates
    Sends each band as a separate focused Vision LLM call to maximize resolution.
    """
    t0 = time.perf_counter()
    groq = get_groq_client()

    if groq.is_mock_mode():
        if allow_mock:
            mock_res = get_mock_document_response(document_type)
            return mock_res, 85.0
        logger.warning("[!] Vision fallback unavailable: Groq is in mock mode (GROQ_API_KEY unset).")
        return {}, 0.0

    try:
        pil_img = Image.open(io.BytesIO(image_bytes))
        width, height = pil_img.size

        # Define 3 overlapping horizontal bands
        bands = [
            ("top_band", (0, 0, width, int(height * 0.28))),
            ("mid_band", (0, int(height * 0.25), width, int(height * 0.50))),
            ("lower_band", (0, int(height * 0.45), width, min(height, int(height * 0.75)))),
        ]

        combined_fields: Dict[str, Any] = {}
        total_vision_latency = 0.0
        prompt = _get_system_prompt_for_type(document_type)

        for band_name, box in bands:
            cropped = pil_img.crop(box)
            buf = io.BytesIO()
            cropped.save(buf, format="JPEG", quality=90)
            band_bytes = buf.getvalue()

            raw_resp, lat = groq.vision_completion(
                messages=[{"role": "user", "content": prompt}],
                image_bytes=band_bytes,
                mime_type="image/jpeg",
                document_type=document_type,
                temperature=0.0,
            )
            total_vision_latency += lat
            parsed = _strip_and_parse_json(raw_resp)
            if parsed:
                # Merge non-null extracted fields
                for k, v in parsed.items():
                    if isinstance(v, dict) and v.get("value") is not None:
                        combined_fields[k] = v
                    elif v is not None and k not in combined_fields:
                        combined_fields[k] = v

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return combined_fields, latency_ms

    except Exception as exc:
        logger.error(f"[!] Low-yield vision fallback error: {exc}", exc_info=True)
        if allow_mock:
            mock_res = get_mock_document_response(document_type)
            return mock_res, round((time.perf_counter() - t0) * 1000.0, 2)
        return {}, round((time.perf_counter() - t0) * 1000.0, 2)


# ---------------------------------------------------------------------------
# Main Hybrid Two-Stage Extraction Flow
# ---------------------------------------------------------------------------

def extract_document_hybrid(
    image_bytes: Optional[bytes] = None,
    image_path: Optional[str] = None,
    document_type: str = "rc_book",
    filename: str = "document.jpg",
    mime_type: str = "image/jpeg",
    image_url: Optional[str] = None,
    allow_mock: bool = False,
) -> DocumentExtractionResult:
    """
    Two-stage Hybrid Document Extraction:
      Stage 1: Deterministic text detection & recognition via local PaddleOCR.
      Stage 2: Semantic mapping using Groq Text Model (llama-3.3-70b-versatile)
               or local deterministic entity parser if Groq is unset/offline.
      Post-Validation: Regex cleanup (VIN, registration number, loose policy sanity).
      Fallback: Horizontal-band Groq Vision fallback if character yield < 40 chars.
    """
    t_start = time.perf_counter()
    doc_uid = f"doc_{uuid.uuid4().hex[:8]}"
    groq = get_groq_client()

    # Ensure image exists on disk for PaddleOCR
    temp_file_created = False
    target_path = image_path

    if not target_path or not os.path.exists(target_path):
        if image_bytes:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
                tf.write(image_bytes)
                target_path = tf.name
                temp_file_created = True
        else:
            target_path = "/tmp/dummy_doc.jpg"

    extraction_method = "paddle_hybrid"
    extracted_corpus = ""

    try:
        # -------------------------------------------------------------------
        # Stage 1: Local PaddleOCR Deterministic Line Extraction
        # -------------------------------------------------------------------
        ocr_result = _run_paddle_ocr(target_path, document_type=document_type, allow_mock=allow_mock)
        raw_text_lines = ocr_result.get("lines", [])
        extracted_corpus = ocr_result.get("corpus", "\n".join(raw_text_lines)).strip()
        ocr_latency_ms = ocr_result.get("latency_ms", 0.0)

        raw_parsed_data: Optional[Dict[str, Any]] = None
        total_latency_ms = ocr_latency_ms

        # -------------------------------------------------------------------
        # Step 3: Low-Yield OCR Fallback Check (<40 chars)
        # -------------------------------------------------------------------
        if len(extracted_corpus) < LOW_YIELD_CHAR_THRESHOLD and image_bytes:
            logger.warning(
                f"[!] Low-yield OCR detected ({len(extracted_corpus)} chars < {LOW_YIELD_CHAR_THRESHOLD}). "
                "Triggering horizontal-band Groq Vision fallback."
            )
            fallback_data, fb_latency = _run_low_yield_vision_fallback(
                image_bytes=image_bytes,
                document_type=document_type,
                mime_type=mime_type,
                allow_mock=allow_mock,
            )
            raw_parsed_data = fallback_data if fallback_data else None
            total_latency_ms += fb_latency
            if raw_parsed_data:
                extraction_method = "vision_fallback"

        else:
            # ---------------------------------------------------------------
            # Stage 2: Semantic Mapping with Groq Text Model or Local Parser
            # ---------------------------------------------------------------
            if not groq.is_mock_mode():
                system_prompt = _get_system_prompt_for_type(document_type)
                user_content = f"Raw OCR Corpus:\n{extracted_corpus}"
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ]
                try:
                    raw_completion, llm_latency = groq.chat_completion(
                        messages=messages,
                        temperature=0.0,
                        response_format={"type": "json_object"},
                    )
                    total_latency_ms += llm_latency
                    raw_parsed_data = _strip_and_parse_json(raw_completion)
                except Exception as e:
                    logger.error(f"[!] Groq chat completion failed: {e}. Falling back to local parser.", exc_info=True)
                    raw_parsed_data = None

            if not raw_parsed_data:
                if allow_mock:
                    logger.info("[*] Using deterministic mock for Groq text mapping (mock allowed).")
                    raw_parsed_data = get_mock_document_response(document_type)
                    total_latency_ms += 45.0
                else:
                    # Run deterministic local rule-based extractor on real OCR text
                    logger.info("[*] Using deterministic local regex parser on extracted OCR text.")
                    raw_parsed_data = _parse_document_corpus_locally(
                        corpus=extracted_corpus,
                        lines=raw_text_lines,
                        document_type=document_type,
                    )
                    extraction_method = "paddle_local_regex"

        # Check if we have any valid parsed fields
        has_extracted_content = False
        if raw_parsed_data and isinstance(raw_parsed_data, dict):
            for k, v in raw_parsed_data.items():
                if isinstance(v, dict) and v.get("value") is not None:
                    has_extracted_content = True
                    break
                elif isinstance(v, str) and v.strip() and v.strip().lower() not in ("null", "none"):
                    has_extracted_content = True
                    break

        if not has_extracted_content:
            logger.warning("[!] Primary and fallback extraction yielded no fields — flagging extraction failed.")
            overall_latency_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
            empty_fields = _build_empty_document_fields(document_type)
            return DocumentExtractionResult(
                document_id=doc_uid,
                document_type=document_type,
                filename=filename,
                file_url=image_url,
                uploaded_at=datetime.now(timezone.utc).isoformat(),
                fields=empty_fields,
                needs_manual_review=True,
                validation_warnings=["Extraction failed — manual review required"],
                latency_ms=overall_latency_ms,
                extraction_method="extraction_failed",
                raw_ocr_corpus=extracted_corpus,
                status="EXTRACTION_FAILED",
            )

        # -------------------------------------------------------------------
        # Step 2: Regex-based Post-Validation & Normalization
        # -------------------------------------------------------------------
        validated_fields, needs_review, warnings = _validate_extracted_fields(
            document_type,
            raw_parsed_data,
        )

        overall_latency_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
        status_flag = "NEEDS_REVIEW" if (needs_review or len(warnings) > 0) else "SUCCESS"

        return DocumentExtractionResult(
            document_id=doc_uid,
            document_type=document_type,
            filename=filename,
            file_url=image_url,
            uploaded_at=datetime.now(timezone.utc).isoformat(),
            fields=validated_fields,
            needs_manual_review=needs_review,
            validation_warnings=warnings,
            latency_ms=overall_latency_ms,
            extraction_method=extraction_method,
            raw_ocr_corpus=extracted_corpus,
            status=status_flag,
        )

    finally:
        # Clean up temporary file if created
        if temp_file_created and target_path and os.path.exists(target_path):
            try:
                os.remove(target_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Backwards-Compatible Function Alias for Existing Callers
# ---------------------------------------------------------------------------

def extract_document_fields(
    image_bytes: bytes,
    document_type: str = "rc_book",
    filename: str = "document.jpg",
    mime_type: str = "image/jpeg",
    image_url: Optional[str] = None,
    allow_mock: bool = False,
) -> DocumentExtractionResult:
    """
    Backwards-compatible wrapper routing to extract_document_hybrid().
    Ensures seamless compatibility with routes_inspection.py and test suites.
    """
    return extract_document_hybrid(
        image_bytes=image_bytes,
        document_type=document_type,
        filename=filename,
        mime_type=mime_type,
        image_url=image_url,
        allow_mock=allow_mock,
    )
