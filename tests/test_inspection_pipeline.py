"""
Unit and Integration Tests for Phase 1 Vehicle Damage Assessment Pipeline.
Tests DamageDetector, Repository, API Routes, and Pipeline Stubs.
"""

import io
import os
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.app.main import app
from backend.app.models.schemas import (
    BoundingBox,
    ClaimInspectionResponse,
    DamageDetection,
    DamageSummary,
    DocumentFields,
    InspectionRecommendation,
)
from backend.app.pipeline.document_ocr import extract_document_fields
from backend.app.pipeline.damage_segmentation import segment_damaged_parts
from backend.app.pipeline.depth_estimation import estimate_damage_depth
from backend.app.pipeline.knowledge_graph import recommend_inspections
from backend.app.pipeline.report_generator import generate_claim_report
from backend.app.storage.repository import LocalJsonRepository
from backend.app.vision.damage_detector import DamageDetector, get_detector

client = TestClient(app)


def _create_dummy_image_bytes() -> bytes:
    """Create a dummy in-memory JPEG image for testing."""
    img = Image.new("RGB", (300, 200), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# 1. Pipeline Stubs Tests
# ---------------------------------------------------------------------------

def test_document_ocr_functional():
    # Document OCR is implemented in Phase 5 (PaddleOCR / Groq)
    dummy_pdf = b"%PDF-1.4 dummy content"
    res = extract_document_fields(dummy_pdf)
    assert res is not None
    assert hasattr(res, "fields")


def test_damage_segmentation_operational():
    # Empty detections return empty list
    masks_empty = segment_damaged_parts(_create_dummy_image_bytes(), [])
    assert masks_empty == []

    # Real inference on dummy image with mock bounding box
    img_bytes = _create_dummy_image_bytes()
    det = DamageDetection(
        class_id=0,
        damage_type="dent",
        confidence=0.88,
        bbox=BoundingBox(x1=50, y1=50, x2=150, y2=150, width=100, height=100)
    )
    masks = segment_damaged_parts(img_bytes, [det])
    assert len(masks) == 1
    assert masks[0].damage_type == "dent"
    assert masks[0].area_pixels > 0
    assert masks[0].area_percentage > 0.0
    assert len(masks[0].polygon_points) >= 3
    assert det.segmentation is not None
    assert det.segmentation.area_pixels == masks[0].area_pixels


from backend.app.pipeline.depth_estimation import (
    estimate_damage_depth,
    get_depth_estimator,
    compute_deformation_severity_tier,
)
import numpy as np


def test_depth_anything_v2_operational():
    # Real depth estimation on in-memory dummy image
    img_bytes = _create_dummy_image_bytes()
    result = estimate_damage_depth(img_bytes)
    assert result is not None
    assert result.inference_skipped is False
    assert 0.0 <= result.min_scene_depth <= result.max_scene_depth
    assert 0.0 <= result.mean_scene_depth <= result.max_scene_depth


def test_depth_deformation_dent_vs_scratch_contrast():
    estimator = get_depth_estimator()
    # Create synthetic 200x200 normalized depth map (flat baseline at depth 0.5)
    depth_map = np.full((200, 200), 0.5, dtype=np.float32)

    # 1. Simulated Dent: target mask area pushed inward (higher distance value = 0.65)
    dent_mask = np.zeros((200, 200), dtype=np.uint8)
    dent_mask[80:120, 80:120] = 1
    depth_map[80:120, 80:120] = 0.65  # deeper into scene (+0.15 relative)

    dent_bbox = BoundingBox(x1=80, y1=80, x2=120, y2=120, width=40, height=40)
    dent_deform = estimator.compute_instance_deformation(
        depth_map=depth_map,
        target_mask=dent_mask,
        all_other_masks=[],
        bbox=dent_bbox,
        area_percentage=4.0,
        damage_type="dent",
    )

    assert dent_deform.reference_available is True
    assert dent_deform.relative_deformation_score is not None
    assert dent_deform.relative_deformation_score > 0.05
    assert dent_deform.deformation_type == "recessed"
    assert dent_deform.severity_tier in ("moderate", "severe")

    # 2. Simulated Scratch: planar surface without indentation (depth matches baseline 0.5)
    scratch_depth_map = np.full((200, 200), 0.5, dtype=np.float32)
    scratch_mask = np.zeros((200, 200), dtype=np.uint8)
    scratch_mask[90:93, 70:130] = 1  # thin line scratch

    scratch_bbox = BoundingBox(x1=70, y1=90, x2=130, y2=93, width=60, height=3)
    scratch_deform = estimator.compute_instance_deformation(
        depth_map=scratch_depth_map,
        target_mask=scratch_mask,
        all_other_masks=[],
        bbox=scratch_bbox,
        area_percentage=0.5,
        damage_type="scratch",
    )

    assert scratch_deform.reference_available is True
    assert scratch_deform.relative_deformation_score is not None
    assert abs(scratch_deform.relative_deformation_score) < 0.01
    assert scratch_deform.deformation_type == "planar"
    assert scratch_deform.severity_tier == "minor"


def test_depth_deformation_reference_unavailable_at_border():
    estimator = get_depth_estimator()
    depth_map = np.full((100, 100), 0.5, dtype=np.float32)

    # Damage mask touching image edge [0:3, 0:3] inside the 5px border exclusion zone
    border_mask = np.zeros((100, 100), dtype=np.uint8)
    border_mask[0:3, 0:3] = 1
    border_bbox = BoundingBox(x1=0, y1=0, x2=3, y2=3, width=3, height=3)

    deform = estimator.compute_instance_deformation(
        depth_map=depth_map,
        target_mask=border_mask,
        all_other_masks=[],
        bbox=border_bbox,
        area_percentage=0.1,
        min_reference_pixels=30,
        border_margin_px=5,
    )

    # Reference ring should be inside exclusion or too small (< 30 valid pixels)
    assert deform.reference_available is False
    assert deform.relative_deformation_score is None
    assert deform.deformation_status == "reference_unavailable"


def test_depth_deformation_excludes_adjacent_damage_masks():
    estimator = get_depth_estimator()
    depth_map = np.full((200, 200), 0.5, dtype=np.float32)

    # Mask A and adjacent Mask B
    mask_a = np.zeros((200, 200), dtype=np.uint8)
    mask_a[90:110, 80:100] = 1
    bbox_a = BoundingBox(x1=80, y1=90, x2=100, y2=110, width=20, height=20)

    mask_b = np.zeros((200, 200), dtype=np.uint8)
    mask_b[90:110, 105:125] = 1  # 5px away, would fall inside dilation ring
    # Give mask_b extreme depth that would contaminate mask_a reference if not excluded
    depth_map[90:110, 105:125] = 0.99

    deform_with_exclusion = estimator.compute_instance_deformation(
        depth_map=depth_map,
        target_mask=mask_a,
        all_other_masks=[mask_b],
        bbox=bbox_a,
        area_percentage=1.0,
    )

    # Baseline depth outside mask_a and mask_b is 0.5. Since mask_a depth is also 0.5,
    # the relative deformation score must remain ~0.0 when mask_b is properly excluded.
    assert deform_with_exclusion.reference_available is True
    assert abs(deform_with_exclusion.relative_deformation_score) < 0.01


from backend.app.pipeline.knowledge_graph import (
    localize_damage_panel,
    localize_damage_panel_details,
    get_knowledge_graph,
    recommend_inspections,
    aggregate_claim_structural_risks,
)


def test_panel_localization_grid_and_aspect_ratio():
    roi = BoundingBox(x1=0, y1=0, x2=1000, y2=1000, width=1000, height=1000)

    # 1. Front view, wide horizontal box spanning 50% width at bottom -> bumper cover
    wide_box = BoundingBox(x1=250, y1=700, x2=750, y2=900, width=500, height=200)
    panel_wide = localize_damage_panel(wide_box, roi, "front")
    assert panel_wide in ("front bumper cover", "front_bumper_cover")

    # 2. Front view, tall narrow vertical box in right outer third -> front right fender
    fender_box = BoundingBox(x1=800, y1=300, x2=950, y2=800, width=150, height=500)
    panel_fender = localize_damage_panel(fender_box, roi, "front")
    assert panel_fender in ("right front fender", "front_right_fender")

    # 3. Rear view, bottom center -> rear bumper cover
    rear_box = BoundingBox(x1=300, y1=700, x2=700, y2=900, width=400, height=200)
    panel_rear = localize_damage_panel(rear_box, roi, "rear")
    assert panel_rear in ("rear bumper cover", "rear_bumper_cover")

    # 4. Fallback on close-up view without ROI
    panel_close, fallback, conf, grid_cell = localize_damage_panel_details(wide_box, None, "close_up")
    assert panel_close in ("front bumper cover", "front_bumper_cover", "hood")


def test_knowledge_graph_seeding_and_idempotency():
    kg = get_knowledge_graph()
    assert kg is not None
    initial_nodes = len(kg.nodes)
    initial_rels = len(kg.relationships)

    assert initial_nodes >= 30, f"Expected >=30 nodes, got {initial_nodes}"
    assert initial_rels >= 35, f"Expected >=35 relationships, got {initial_rels}"

    # Verify idempotency: re-running seeding must NOT duplicate nodes or edges
    kg.seed_schema()
    assert len(kg.nodes) == initial_nodes
    assert len(kg.relationships) == initial_rels


def test_structural_reasoning_front_bumper_traversal():
    recs = recommend_inspections(
        damage_type="dent",
        detected_panel="front bumper cover",
        deformation_score=0.85,
        area_percentage=6.0,
        max_hops=3,
    )
    assert len(recs) > 0

    comp_names = [r.component_name for r in recs]
    assert any("reinforcement" in name for name in comp_names)
    assert any("crush" in name for name in comp_names)
    assert any("radiator support" in name for name in comp_names)

    # Check that reinforcement bar has 1-hop path
    bar_rec = next(r for r in recs if "reinforcement" in r.component_name)
    assert bar_rec.load_path[0] in ("front bumper cover", "front_bumper_cover")
    assert "reinforcement" in bar_rec.load_path[1]
    assert bar_rec.risk_score > 0.5
    assert bar_rec.safety_risk_level in ("HIGH", "MEDIUM")

    # Check hop attenuation: 1-hop risk >= 2-hop risk (due to transmission weight and lambda=0.75)
    crush_rec = next(r for r in recs if "crush" in r.component_name)
    assert bar_rec.risk_score >= crush_rec.risk_score


def test_risk_formula_bounds_and_deformation_contrast():
    # Deep severe dent vs superficial minor scratch
    recs_severe = recommend_inspections("dent", "front bumper cover", deformation_score=0.90, area_percentage=7.0)
    recs_minor = recommend_inspections("scratch", "front bumper cover", deformation_score=0.05, area_percentage=0.2)

    assert len(recs_severe) > 0
    assert len(recs_minor) > 0

    # Bounds check
    for r in recs_severe:
        assert 0.0 <= r.risk_score <= 1.0
    for r in recs_minor:
        assert 0.0 <= r.risk_score <= 1.0

    # Contrast check: severe damage must produce higher structural risk than minor scratch
    assert recs_severe[0].risk_score > recs_minor[0].risk_score


def test_aggregate_claim_structural_risks_noisy_or():
    from backend.app.models.schemas import DamageDeformation, SegmentationMask

    det_bumper = DamageDetection(
        class_id=0,
        damage_type="dent",
        confidence=0.88,
        bbox=BoundingBox(x1=200, y1=650, x2=800, y2=900, width=600, height=250),
        classified=True,
        detected_panel="front bumper cover",
        segmentation=SegmentationMask(damage_type="dent", area_pixels=15000, area_percentage=6.0, polygon_points=[[200, 650], [800, 650], [800, 900]]),
        deformation=DamageDeformation(relative_deformation_score=0.80, severity_tier="severe"),
    )

    det_hood = DamageDetection(
        class_id=0,
        damage_type="dent",
        confidence=0.82,
        bbox=BoundingBox(x1=300, y1=200, x2=700, y2=500, width=400, height=300),
        classified=True,
        detected_panel="hood",
        segmentation=SegmentationMask(damage_type="dent", area_pixels=12000, area_percentage=5.0, polygon_points=[[300, 200], [700, 200], [700, 500]]),
        deformation=DamageDeformation(relative_deformation_score=0.70, severity_tier="severe"),
    )

    matrix = aggregate_claim_structural_risks([det_bumper, det_hood])
    assert len(matrix) > 0

    # Radiator support assembly is reachable from both front bumper cover and hood
    rad_recs = [m for m in matrix if "radiator support" in m.component_name]
    assert len(rad_recs) == 1, "Expected deduplicated entry for radiator support assembly"

    rad_support = rad_recs[0]
    # Noisy-OR should yield high risk and compound explanation
    assert rad_support.risk_score > 0.4
    assert "Flagged via" in rad_support.rationale
    assert "front bumper cover" in rad_support.rationale and "hood" in rad_support.rationale
    assert rad_support.source_panel == "multi-panel convergence"



def test_report_generator_functional():
    summary = DamageSummary(
        total_damages_count=0,
        damage_counts_by_type={},
        unique_damage_types=[],
        damages_by_view={},
        severity_assessment="No Damage Detected",
    )
    claim_resp = ClaimInspectionResponse(
        claim_id="CLM-TEST-000",
        status="COMPLETED",
        created_at="2026-08-31T00:00:00",
        images=[],
        damage_summary=summary,
    )
    report = generate_claim_report(claim_resp)
    assert report is not None
    assert hasattr(report, "executive_summary")


# ---------------------------------------------------------------------------
# 2. Vision Detector Tests
# ---------------------------------------------------------------------------

def test_damage_detector_initialization():
    detector = get_detector()
    assert detector is not None
    assert len(detector.classes) == 6
    assert detector.classes[0] == "dent"
    assert detector.classes[1] == "scratch"
    assert detector.classes[2] == "crack"
    assert detector.classes[3] == "shattered_glass"
    assert detector.classes[4] == "broken_lamp"
    assert detector.classes[5] == "flat_tire"


def test_damage_detector_inference_on_sample():
    detector = get_detector()
    sample_path = "backend/static/samples/1.png"
    if os.path.exists(sample_path):
        annotated, detections = detector.predict(sample_path, confidence_threshold=0.20)
        assert isinstance(annotated, Image.Image)
        assert isinstance(detections, list)
        assert len(detections) > 0
        first_det = detections[0]
        assert first_det.damage_type in detector.classes.values()
        assert 0.0 <= first_det.confidence <= 1.0
        assert first_det.bbox.width > 0
        assert first_det.bbox.height > 0


# ---------------------------------------------------------------------------
# 3. Storage Repository Tests
# ---------------------------------------------------------------------------

def test_local_json_repository(tmp_path):
    db_file = os.path.join(tmp_path, "test_claims.json")
    repo = LocalJsonRepository(storage_path=db_file)

    claim_data = {
        "claim_id": "CLM-UNIT-999",
        "status": "COMPLETED",
        "created_at": "2026-08-31T12:00:00",
        "damage_summary": {
            "total_damages_count": 2,
            "damage_counts_by_type": {"dent": 1, "scratch": 1},
            "unique_damage_types": ["dent", "scratch"],
            "damages_by_view": {"front": ["dent", "scratch"]},
            "severity_assessment": "Moderate",
        },
        "images": [],
    }

    # Save
    repo.save_claim(claim_data)

    # Retrieve
    retrieved = repo.get_claim("CLM-UNIT-999")
    assert retrieved is not None
    assert retrieved["claim_id"] == "CLM-UNIT-999"

    # List
    all_claims = repo.list_claims()
    assert len(all_claims) >= 1

    # Delete
    deleted = repo.delete_claim("CLM-UNIT-999")
    assert deleted is True
    assert repo.get_claim("CLM-UNIT-999") is None


# ---------------------------------------------------------------------------
# 4. FastAPI Endpoints Integration Tests
# ---------------------------------------------------------------------------

def test_health_endpoint():
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "HEALTHY"
    assert data["vision_pipeline"]["damage_detector_yolo11m"]["loaded"] is True
    assert data["vision_pipeline"]["damage_detector_yolo11m"]["classes_count"] == 6
    assert data["vision_pipeline"]["damage_segmenter_sam2"]["loaded"] is True
    assert data["vision_pipeline"]["depth_anything_v2"]["loaded"] is True


def test_inspect_and_retrieve_claim():
    sample_path = "backend/static/samples/1.png"
    if not os.path.exists(sample_path):
        pytest.skip("Sample image 1.png not found")

    with open(sample_path, "rb") as f:
        img_bytes = f.read()

    claim_id = "CLM-AUTO-TEST-888"

    # POST Inspect
    res = client.post(
        f"/api/claims/{claim_id}/inspect",
        data={
            "view_angles": ["front"],
            "vehicle_reg_number": "KA 01 MJ 9999",
            "surveyor_notes": "Automated test inspection",
            "confidence_threshold": "0.25",
        },
        files=[("files", ("test_car.png", img_bytes, "image/png"))],
    )
    assert res.status_code == 200
    res_data = res.json()
    assert res_data["claim_id"] == claim_id
    assert len(res_data["images"]) == 1
    assert res_data["images"][0]["damage_count"] >= 1
    assert "annotated_image_url" in res_data["images"][0]
    assert "segmented_image_url" in res_data["images"][0]
    assert "depth_colormap_url" in res_data["images"][0]
    assert "depth_result" in res_data["images"][0]

    # Check that detections have segmentation and deformation data attached
    first_det = res_data["images"][0]["detections"][0]
    assert "segmentation" in first_det
    if first_det["segmentation"]:
        assert first_det["segmentation"]["area_pixels"] > 0
        assert first_det["segmentation"]["area_percentage"] > 0.0

    assert "deformation" in first_det
    if first_det["deformation"]:
        assert first_det["deformation"]["reference_available"] in (True, False)
        assert first_det["deformation"]["severity_tier"] in ("minor", "moderate", "severe")

    # GET Retrieve
    get_res = client.get(f"/api/claims/{claim_id}")
    assert get_res.status_code == 200
    get_data = get_res.json()
    assert get_data["claim_id"] == claim_id
    assert get_data["vehicle_reg_number"] == "KA 01 MJ 9999"
    assert "segmented_image_url" in get_data["images"][0]
    assert "depth_colormap_url" in get_data["images"][0]


# ---------------------------------------------------------------------------
# 5. Phase 2 Refinement Tests: High-Recall, Padded Boxes, ROI, AMG & IoA
# ---------------------------------------------------------------------------

from backend.app.pipeline.damage_segmentation import compute_edge_aware_padded_box
from backend.app.vision.vehicle_detector import get_vehicle_detector


def test_edge_aware_padded_box_heuristic():
    img_w, img_h = 1000, 800
    # Standard box in center: 100x100, far from edges and no neighbors
    center_box = BoundingBox(x1=400, y1=300, x2=500, y2=400, width=100, height=100)
    padded = compute_edge_aware_padded_box(
        b=center_box,
        img_w=img_w,
        img_h=img_h,
        all_boxes=[center_box],
        base_padding_pct=0.10,
        edge_padding_pct=0.15,
    )
    # 10% of 100 is 10px on each side -> [390, 290, 510, 410]
    assert padded[0] == pytest.approx(390.0)
    assert padded[1] == pytest.approx(290.0)
    assert padded[2] == pytest.approx(510.0)
    assert padded[3] == pytest.approx(410.0)

    # Near lower edge box (e.g. y2 = 780, within 5% of img_h=800) -> 15% edge padding + downward reach
    near_edge_box = BoundingBox(x1=400, y1=680, x2=500, y2=780, width=100, height=100)
    padded_edge = compute_edge_aware_padded_box(
        b=near_edge_box,
        img_w=img_w,
        img_h=img_h,
        all_boxes=[near_edge_box],
        base_padding_pct=0.10,
        edge_padding_pct=0.15,
        edge_margin_ratio=0.05,
    )
    # 15% of 100 is 15px -> px1 = 385, py1 = 665, px2 = 515, py2 clamped or with extra downward
    assert padded_edge[0] == pytest.approx(385.0)
    assert padded_edge[1] == pytest.approx(665.0)
    assert padded_edge[2] == pytest.approx(515.0)
    assert padded_edge[3] > 795.0

    # Adjacency check: two neighboring boxes within 20px
    box_a = BoundingBox(x1=200, y1=200, x2=250, y2=250, width=50, height=50)
    box_b = BoundingBox(x1=260, y1=200, x2=310, y2=250, width=50, height=50)  # dx = 10px <= 20px
    padded_adj = compute_edge_aware_padded_box(
        b=box_a,
        img_w=img_w,
        img_h=img_h,
        all_boxes=[box_a, box_b],
        base_padding_pct=0.10,
        edge_padding_pct=0.15,
    )
    # 15% of 50 is 7.5px -> px1 = 192.5
    assert padded_adj[0] == pytest.approx(192.5)


def test_vehicle_detector_roi():
    v_det = get_vehicle_detector()
    assert v_det is not None
    sample_path = "backend/static/samples/1.png"
    if os.path.exists(sample_path):
        crop_img, roi_bbox, crop_coords = v_det.detect_vehicle_roi(sample_path)
        assert crop_img is not None
        assert roi_bbox is not None
        assert crop_coords[2] > crop_coords[0]
        assert crop_coords[3] > crop_coords[1]
        assert roi_bbox.width > 0
        assert roi_bbox.height > 0


def test_confidence_tier_assignment():
    det_high = DamageDetection(
        class_id=0,
        damage_type="dent",
        confidence=0.72,
        confidence_tier="high",
        classified=True,
        bbox=BoundingBox(x1=10, y1=10, x2=50, y2=50, width=40, height=40),
    )
    assert det_high.confidence_tier == "high"
    assert det_high.classified is True

    det_low = DamageDetection(
        class_id=1,
        damage_type="scratch",
        confidence=0.22,
        confidence_tier="low",
        classified=True,
        bbox=BoundingBox(x1=10, y1=10, x2=50, y2=50, width=40, height=40),
    )
    assert det_low.confidence_tier == "low"
    assert det_low.classified is True


def test_amg_redundancy_and_downstream_safety():
    sample_path = "backend/static/samples/1.png"
    if not os.path.exists(sample_path):
        pytest.skip("Sample image 1.png not found")

    with open(sample_path, "rb") as f:
        img_bytes = f.read()

    claim_id = "CLM-TEST-REFINEMENT-P2"

    res = client.post(
        f"/api/claims/{claim_id}/inspect",
        data={
            "view_angles": ["front"],
            "vehicle_reg_number": "KA 05 AB 1234",
            "confidence_threshold": "0.15",
            "enable_amg": "true",
        },
        files=[("files", ("front_car.png", img_bytes, "image/png"))],
    )
    assert res.status_code == 200
    data = res.json()
    assert data["claim_id"] == claim_id
    assert len(data["images"]) == 1

    img_res = data["images"][0]
    # Verify vehicle ROI was found and attached
    assert img_res["vehicle_roi_bbox"] is not None
    # Verify latency metrics were collected
    assert img_res["latency_metrics"] is not None
    assert img_res["latency_metrics"]["yolo_detection_ms"] > 0.0
    assert img_res["latency_metrics"]["box_prompted_sam2_ms"] > 0.0
    assert img_res["latency_metrics"]["total_pipeline_ms"] > 0.0

    # Verify confidence tiers are set on classified detections
    for d in img_res["detections"]:
        assert d["confidence_tier"] in ["high", "low"]
        assert d["classified"] is True

    # Verify downstream safety (Refinement 1):
    # damage_summary.total_damages_count equals classified detections count ONLY!
    assert data["damage_summary"]["total_damages_count"] == len(img_res["detections"])
    # Any unclassified regions are isolated
    assert "unclassified_candidates_count" in data["damage_summary"]
    for u in img_res["unclassified_regions"]:
        assert u["classified"] is False
        assert u["damage_type"] == "unclassified"
        assert "unclassified" in u["label"].lower()
        # Verify no tiny noise fragment receives spurious SEVERE tag
        if u.get("segmentation") and u["segmentation"]["area_percentage"] < 1.0:
            if u.get("deformation"):
                assert u["deformation"]["severity_tier"] != "severe"


def test_deformation_size_gate_unclassified():
    """Verify that unclassified fragments below 1.0% skip scoring and never get severe."""
    from backend.app.pipeline.depth_estimation import get_depth_estimator

    estimator = get_depth_estimator()
    depth_map = np.full((300, 300), 0.5, dtype=np.float32)
    # Create tiny mask (e.g. 5x5 pixels = 25 pixels in 300x300 = 0.027% of image)
    tiny_mask = np.zeros((300, 300), dtype=np.uint8)
    tiny_mask[100:105, 100:105] = 1

    deform = estimator.compute_instance_deformation(
        depth_map=depth_map,
        mask=tiny_mask,
        area_percentage=0.32,  # The exact 0.32% case that caused the bug
        damage_type="unclassified",
    )
    assert deform.reference_available is False
    assert deform.relative_deformation_score is None
    assert deform.severity_tier is None
    assert deform.deformation_status == "too_small_for_reliable_severity_estimate"


def test_amg_config_named_values():
    """Verify AMGConfig exposes named parameters with correct default thresholds."""
    from backend.app.pipeline.amg_config import default_amg_config

    assert default_amg_config.points_per_side in [16, 20, 24]
    assert 0.90 <= default_amg_config.pred_iou_thresh <= 0.95
    assert 0.95 <= default_amg_config.stability_score_thresh <= 0.98
    assert default_amg_config.min_crop_area_pct == 1.0
    assert default_amg_config.dedup_iou_thresh == 0.80
    assert default_amg_config.min_deform_scoring_area_pct == 1.0
    assert default_amg_config.vehicle_crop_margin_pct == 0.02
    assert default_amg_config.display_cap_n == 5


