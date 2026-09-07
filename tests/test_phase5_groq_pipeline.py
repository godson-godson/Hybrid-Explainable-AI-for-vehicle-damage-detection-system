"""
Phase 5 Comprehensive Verification Suite:
Tests Groq Client, Document Understanding (OCR), Explainable Report Generation,
and API Routes under strict non-fabrication and validation constraints.
"""

import io
import json
import os
import sys
import time
from PIL import Image, ImageDraw

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.models.schemas import (
    ClaimDocuments,
    ClaimInspectionResponse,
    DamageDeformation,
    DamageDetection,
    DamageSummary,
    BoundingBox,
    InspectionRecommendation,
)
from backend.app.pipeline.document_ocr import (
    _strip_and_parse_json,
    _validate_extracted_fields,
    post_validate_and_normalize_fields,
    extract_document_fields,
    extract_document_hybrid,
    _run_paddle_ocr,
)
from backend.app.pipeline.groq_client import GroqClient, get_groq_client
from backend.app.pipeline.report_generator import (
    _prepare_compact_pipeline_payload,
    generate_survey_report,
)
from backend.app.storage.repository import get_repository


def create_synthetic_document_image(doc_title: str, text_lines: list) -> bytes:
    """Create a synthetic high-contrast document image for OCR testing."""
    img = Image.new("RGB", (600, 400), color=(250, 250, 250))
    draw = ImageDraw.Draw(img)
    draw.text((30, 25), doc_title, fill=(10, 10, 10))
    y = 70
    for line in text_lines:
        draw.text((30, y), line, fill=(40, 40, 40))
        y += 35
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Test 1: Shared Groq Client & Fallback Mocking
# ---------------------------------------------------------------------------

def test_groq_client_initialization_and_mock():
    """Verify GroqClient initializes and operates deterministically in mock mode."""
    client = GroqClient(api_key="")
    assert client.is_mock_mode() is True

    # Test text completion in mock mode
    text, latency_ms = client.text_chat_completion(
        messages=[{"role": "user", "content": "Generate survey report"}],
        temperature=0.25,
    )
    assert len(text) > 50
    assert latency_ms >= 0.0

    # Test vision completion in mock mode
    fake_bytes = b"FAKE_JPEG_STREAM"
    raw_ocr, vis_latency = client.vision_completion(
        messages=[{"role": "user", "content": "Extract RC"}],
        image_bytes=fake_bytes,
        document_type="rc_book",
    )
    assert "registration_number" in raw_ocr
    assert vis_latency >= 0.0

    # Test model catalog availability
    models = client.list_available_models()
    assert len(models) >= 3


# ---------------------------------------------------------------------------
# Test 2: JSON Stripping & Conversation Filler Removal (Critical Refinement 1)
# ---------------------------------------------------------------------------

def test_json_stripping_and_parsing():
    """Verify regex/bracket stripping handles conversational filler and code fences."""
    # Case 1: Markdown code fence
    fence_input = '```json\n{"registration_number": {"value": "DL03CC4921"}}\n```'
    parsed1 = _strip_and_parse_json(fence_input)
    assert parsed1 is not None
    assert parsed1["registration_number"]["value"] == "DL03CC4921"

    # Case 2: Conversational prefix and suffix
    conversational = (
        'Here is the extracted document data you requested from the image:\n\n'
        '{\n  "registration_number": {"value": "MH12AB1234", "confidence": 0.95}\n}\n\n'
        'I hope this helps your insurance survey.'
    )
    parsed2 = _strip_and_parse_json(conversational)
    assert parsed2 is not None
    assert parsed2["registration_number"]["value"] == "MH12AB1234"

    # Case 3: Completely invalid / unparseable output
    invalid = "I cannot read any text on this blurry image."
    parsed3 = _strip_and_parse_json(invalid)
    assert parsed3 is None


# ---------------------------------------------------------------------------
# Test 3: Independent Post-Extraction Validation
# ---------------------------------------------------------------------------

def test_independent_document_validation():
    """Verify validation logic independent of the LLM catches format anomalies."""
    # 1. Valid RC fields
    valid_rc = {
        "registration_number": {"value": "DL 03 CC 4921", "confidence": 0.95},
        "owner_name": {"value": "Rajesh Kumar", "confidence": 0.92},
        "chassis_number": {"value": "MALBB51BLAM123456", "confidence": 0.94},
    }
    items, needs_review, warnings = _validate_extracted_fields("rc_book", valid_rc)
    assert items["registration_number"].is_valid is True
    assert needs_review is False
    assert len(warnings) == 0

    # 2. Missing required field (null) -> MUST flag needs_review
    missing_rc = {
        "registration_number": {"value": None, "confidence": 0.0},
        "owner_name": {"value": "Rajesh Kumar", "confidence": 0.92},
    }
    items_m, needs_review_m, warnings_m = _validate_extracted_fields("rc_book", missing_rc)
    assert needs_review_m is True
    assert any("missing registration number" in w.lower() for w in warnings_m)

    # 3. Malformed date in insurance policy
    bad_policy = {
        "policy_number": {"value": "POL-12345678", "confidence": 0.90},
        "policy_expiry_date": {"value": "INVALID_DATE_FORMAT", "confidence": 0.85},
    }
    items_p, needs_review_p, warnings_p = _validate_extracted_fields("insurance_policy", bad_policy)
    assert needs_review_p is True
    assert items_p["policy_expiry_date"].is_valid is False
    assert any("date format" in w.lower() for w in warnings_p)


# ---------------------------------------------------------------------------
# Test 3B: Regex Post-Validation & Normalization (Step 2 Revision)
# ---------------------------------------------------------------------------

def test_regex_post_validation_and_normalization():
    """Verify regex post-validation normalizes formats without discarding values."""
    raw_input = {
        "registration_number": {"value": "dl-03-cc-4921", "confidence": 0.9},
        "chassis_number": {"value": " mal-bb51blam123456 ", "confidence": 0.95},
        "engine_number": {"value": "g4laj-123456", "confidence": 0.9},  # Non-17 char engine number!
        "policy_number": {"value": "custom-policy-format-9999", "confidence": 0.8},
    }

    norm = post_validate_and_normalize_fields(raw_input)

    # 1. Registration number normalized and standardized
    assert norm["registration_number"]["value"] == "DL-03-CC-4921"

    # 2. Chassis number cleaned to 17-char VIN
    assert norm["chassis_number"]["value"] == "MALBB51BLAM123456"
    assert len(norm["chassis_number"]["value"]) == 17

    # 3. Engine number must NOT have 17-char length check applied
    assert norm["engine_number"]["value"] == "G4LAJ-123456"

    # 4. Unusual policy number is preserved (loose heuristic, not discarded)
    assert norm["policy_number"]["value"] == "CUSTOM-POLICY-FORMAT-9999"


# ---------------------------------------------------------------------------
# Test 3C: Two-Stage Hybrid Extraction Flow (Step 1 & Step 4)
# ---------------------------------------------------------------------------

def test_two_stage_paddle_hybrid_extraction():
    """Verify two-stage PaddleOCR + Groq Text hybrid extraction flow."""
    doc_bytes = create_synthetic_document_image(
        "FORM 23 - CERTIFICATE OF REGISTRATION",
        [
            "REGISTRATION NUMBER: DL 03 CC 4921",
            "OWNER: RAJESH KUMAR SHARMA",
            "CHASSIS: MALBB51BLAM123456",
            "ENGINE: G4LAJ123456",
            "DATE: 14/08/2021",
            "FUEL: PETROL",
        ],
    )

    t0 = time.perf_counter()
    result = extract_document_hybrid(
        image_bytes=doc_bytes,
        document_type="rc_book",
        filename="test_rc.jpg",
    )
    latency_ms = (time.perf_counter() - t0) * 1000.0

    assert result.document_type == "rc_book"
    assert result.extraction_method in ("paddle_hybrid", "paddle_local_regex")
    assert result.raw_ocr_corpus is not None
    assert len(result.raw_ocr_corpus) > 30
    assert "registration_number" in result.fields
    assert result.fields["registration_number"].value is not None
    print(f"\n[✓] Two-Stage Hybrid OCR Latency: {latency_ms:.2f}ms (Method: {result.extraction_method})")


# ---------------------------------------------------------------------------
# Test 3D: Low-Yield OCR Fallback (<40 chars) (Step 3)
# ---------------------------------------------------------------------------

def test_low_yield_ocr_fallback_triggers_on_sparse_text():
    """Verify fallback to horizontal-band vision model triggers when OCR text is sparse."""
    # Blank/sparse image with under 40 characters
    blank_img = create_synthetic_document_image("A", [])

    # Verify that allow_mock=False never injects fake demo data on sparse text
    from backend.app.pipeline.document_ocr import _run_low_yield_vision_fallback
    fallback_fields_real, _ = _run_low_yield_vision_fallback(
        image_bytes=blank_img,
        document_type="rc_book",
        allow_mock=False,
    )
    assert len(fallback_fields_real) == 0

    # Verify mock fallback is available when explicitly requested (e.g. offline unit testing)
    fallback_fields_mock, lat = _run_low_yield_vision_fallback(
        image_bytes=blank_img,
        document_type="rc_book",
        allow_mock=True,
    )
    assert len(fallback_fields_mock) > 0
    assert "registration_number" in fallback_fields_mock
    print(f"[✓] Low-Yield Vision Fallback Latency: {lat:.2f}ms")


# ---------------------------------------------------------------------------
# Test 3E: Verification with Real Distinct Documents & Genuine Failure State
# ---------------------------------------------------------------------------

def test_distinct_documents_and_genuine_failure_handling():
    """
    Verify with two different real document images with distinct field values,
    and confirm bad/unclear image returns EXTRACTION_FAILED with manual review warning.
    """
    # 1. Distinct Document A (Kerala RC)
    res_a = extract_document_hybrid(
        image_path="backend/static/uploads/CLM-2026-1217/documents/rc_book_1182bc.png",
        document_type="rc_book",
        allow_mock=False,
    )
    assert res_a.status == "SUCCESS"
    assert res_a.fields["registration_number"].value == "KL-08-CD-4567"
    assert "RAHUL" in res_a.fields["owner_name"].value.upper()

    # 2. Distinct Document B (Karnataka RC)
    res_b = extract_document_hybrid(
        image_path="backend/static/samples/test_rc_distinct_karnataka.png",
        document_type="rc_book",
        allow_mock=False,
    )
    assert res_b.status == "SUCCESS"
    assert res_b.fields["registration_number"].value == "KA-05-MJ-8822"
    assert "ANANYA" in res_b.fields["owner_name"].value.upper()

    # Confirm values differ completely
    assert res_a.fields["registration_number"].value != res_b.fields["registration_number"].value
    assert res_a.fields["owner_name"].value != res_b.fields["owner_name"].value

    # 3. Blank / Corrupt Document (Failure handling)
    res_fail = extract_document_hybrid(
        image_path="backend/static/samples/test_blank_corrupt.png",
        document_type="rc_book",
        allow_mock=False,
    )
    assert res_fail.status == "EXTRACTION_FAILED"
    assert res_fail.needs_manual_review is True
    assert "Extraction failed — manual review required" in res_fail.validation_warnings
    assert res_fail.fields["registration_number"].value is None


# ---------------------------------------------------------------------------
# Test 4: Compact Payload Summarization (Critical Refinement 2)
# ---------------------------------------------------------------------------

def test_compact_payload_summarization_no_raw_masks():
    """Verify summarized payload contains numerical metrics and zero mask coordinate arrays."""
    sample_detection = DamageDetection(
        class_id=0,
        damage_type="dent",
        confidence=0.88,
        bbox=BoundingBox(x1=10.0, y1=20.0, x2=150.0, y2=200.0, width=140.0, height=180.0),
        view_angle="front",
        detected_panel="front_bumper",
        deformation=DamageDeformation(
            relative_deformation_score=0.125,
            severity_tier="moderate",
            deformation_type="recessed",
        ),
    )

    claim = ClaimInspectionResponse(
        claim_id="TEST-COMPACT-001",
        vehicle_reg_number="DL 03 CC 4921",
        status="COMPLETED",
        images=[],
        damage_summary=DamageSummary(
            total_damages_count=1,
            damage_counts_by_type={"dent": 1},
            severity_assessment="Moderate",
        ),
        structural_risk_matrix=[
            InspectionRecommendation(
                component_name="front longitudinal frame rails",
                impact_zone="front",
                risk_score=0.45,
                safety_risk_level="MEDIUM",
                recommended_action="Visual Borescope",
                estimated_labor_hours=0.5,
                load_path=["front bumper", "frame rails"],
                rationale="Traversing load path from bumper impact.",
            )
        ],
    )
    # Inject detection
    from backend.app.models.schemas import ImageInspectionResult
    claim.images.append(
        ImageInspectionResult(
            image_id="img_1",
            filename="front.jpg",
            original_image_url="/static/test.jpg",
            annotated_image_url="/static/test_annot.jpg",
            damage_count=1,
            detections=[sample_detection],
        )
    )

    payload = _prepare_compact_pipeline_payload(claim)

    # Convert to json string to inspect size
    json_str = json.dumps(payload)
    assert len(json_str) < 5000  # Highly compact, prevents context window overflow!
    assert "polygon_points" not in json_str
    assert "sam2_mask" not in json_str
    assert "depth_map" not in json_str
    assert payload["classified_damage_instances"][0]["relative_deformation"] == 0.125


# ---------------------------------------------------------------------------
# Test 5: Explainable Report Generation (Chapter 4.6.5 Sections)
# ---------------------------------------------------------------------------

def test_explainable_survey_report_generation():
    """Verify generated report adheres to Chapter 4.6.5 structure and non-fabrication."""
    claim = ClaimInspectionResponse(
        claim_id="CLM-VERIFY-001",
        vehicle_reg_number="DL 03 CC 4921",
        status="COMPLETED",
        images=[],
        damage_summary=DamageSummary(
            total_damages_count=3,
            damage_counts_by_type={"dent": 2, "crack": 1},
            severity_assessment="Moderate",
        ),
        structural_risk_matrix=[
            InspectionRecommendation(
                component_name="front longitudinal frame rails",
                impact_zone="front",
                risk_score=0.496,
                safety_risk_level="MEDIUM",
                recommended_action="Visual Borescope & Mount Tolerance Inspection",
                estimated_labor_hours=0.5,
                load_path=["front bumper cover", "bumper reinforcement bar", "frame rails"],
                rationale="Impact on bumper transmitted along reinforcement bar to frame rails.",
            )
        ],
    )

    t0 = time.perf_counter()
    report = generate_survey_report(claim)
    latency_ms = (time.perf_counter() - t0) * 1000

    assert report.report_id.startswith("REP-CLM-VERIFY-001")
    assert report.claim_id == "CLM-VERIFY-001"
    assert len(report.executive_summary) > 20
    assert len(report.internal_inspection_plan) > 0
    assert len(report.markdown_dossier) > 100
    assert "front longitudinal frame rails" in report.markdown_dossier.lower()
    print(f"\n[✓] Survey Report Generation Latency: {latency_ms:.2f}ms (Model: {report.model_used})")


# ---------------------------------------------------------------------------
# Test 6: API Endpoints Integration Flow
# ---------------------------------------------------------------------------

def test_phase5_api_endpoints_flow():
    """Test document upload, manual override, and report generation via FastAPI."""
    client = TestClient(app)
    test_claim_id = f"CLM-TEST-P5-{int(time.time())}"

    # 1. First inspect vehicle images to create base claim
    synthetic_img = create_synthetic_document_image("Car Damage Photo", ["Dent on front bumper"])
    files = [("files", ("front.jpg", synthetic_img, "image/jpeg"))]
    resp_inspect = client.post(
        f"/api/claims/{test_claim_id}/inspect",
        files=files,
        data={"vehicle_reg_number": "DL 03 CC 4921", "surveyor_notes": "Phase 5 integration test"},
    )
    assert resp_inspect.status_code == 200

    # 2. Upload Document (RC Book)
    synthetic_rc = create_synthetic_document_image(
        "REGISTRATION CERTIFICATE",
        [
            "REGN NO: DL 03 CC 4921",
            "OWNER: RAJESH KUMAR SHARMA",
            "CHASSIS: MALBB51BLAM123456",
        ]
    )
    doc_file = {"file": ("rc_book.jpg", synthetic_rc, "image/jpeg")}
    resp_doc = client.post(
        f"/api/claims/{test_claim_id}/documents",
        files=doc_file,
        data={"document_type": "rc_book"},
    )
    assert resp_doc.status_code == 200
    doc_json = resp_doc.json()
    assert doc_json["document_type"] == "rc_book"
    assert "registration_number" in doc_json["fields"]
    assert "extraction_method" in doc_json

    # 3. Retrieve Documents
    resp_get_docs = client.get(f"/api/claims/{test_claim_id}/documents")
    assert resp_get_docs.status_code == 200
    docs_payload = resp_get_docs.json()
    assert docs_payload["rc_book"] is not None

    # 4. Manual Field Override (Human-in-the-loop)
    resp_put = client.put(
        f"/api/claims/{test_claim_id}/documents/rc_book",
        json={"registration_number": "DL 03 CC 4921", "owner_name": "Rajesh K. Sharma (Confirmed)"},
    )
    assert resp_put.status_code == 200
    updated_doc = resp_put.json()
    assert updated_doc["fields"]["owner_name"]["value"] == "Rajesh K. Sharma (Confirmed)"
    assert updated_doc["fields"]["owner_name"]["surveyor_overridden"] is True

    # 5. Generate Survey Report
    resp_rep = client.post(f"/api/claims/{test_claim_id}/generate-report")
    assert resp_rep.status_code == 200
    rep_json = resp_rep.json()
    assert rep_json["claim_id"] == test_claim_id
    assert "markdown_dossier" in rep_json
    assert rep_json["status"] == "DRAFT"

    # 6. Retrieve Report
    resp_get_rep = client.get(f"/api/claims/{test_claim_id}/report")
    assert resp_get_rep.status_code == 200

    # 7. Finalize Report
    resp_fin = client.post(
        f"/api/claims/{test_claim_id}/finalize-report",
        json={"signoff_notes": "Physical borescope inspection completed. Authorized."},
    )
    assert resp_fin.status_code == 200
    fin_json = resp_fin.json()
    assert fin_json["status"] == "FINALIZED"
    assert "Physical borescope" in fin_json["surveyor_signoff_notes"]
