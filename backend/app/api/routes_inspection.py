"""
Inspection and Damage Assessment API Endpoints.
Handles multi-view vehicle photo uploads, YOLO11m inference, damage aggregation,
and claim inspection record persistence.
"""

from datetime import datetime, timezone
import io
import os
import sys
from pathlib import Path
import uuid
from typing import Any, Dict, List, Optional

# Ensure package context and sys.path if this submodule is executed or loaded directly
if not __package__:
    project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    __package__ = "backend.app.api"

import numpy as np
from fastapi import APIRouter, Body, File, Form, HTTPException, Query, UploadFile, status
from fastapi.responses import JSONResponse
from PIL import Image

from ..models.schemas import (
    ClaimDocuments,
    ClaimInspectionResponse,
    DamageDeformation,
    DamageDetection,
    DamageSummary,
    DepthMapResult,
    DocumentExtractionResult,
    ExtractedFieldItem,
    ImageInspectionResult,
    LLMSurveyReport,
    PipelineLatencyMetrics,
)
from ..storage.repository import get_repository
from ..vision.damage_detector import get_detector
from ..vision.vehicle_detector import get_vehicle_detector
from ..pipeline.amg_config import default_amg_config
from ..pipeline.damage_segmentation import get_segmenter
from ..pipeline.depth_estimation import get_depth_estimator
from ..pipeline.knowledge_graph import (
    get_knowledge_graph,
    localize_damage_panel,
)
from ..pipeline.document_ocr import extract_document_fields, extract_document_hybrid
from ..pipeline.groq_client import GroqClientError, get_groq_client
from ..pipeline.report_generator import (
    ReportGenerationError,
    generate_claim_report,
    generate_survey_report,
)
import logging
import time

logger = logging.getLogger("routes_inspection")

router = APIRouter(prefix="/api/claims", tags=["Vehicle Damage Inspection"])

# Configurable constants
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/jpg"}
MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB per image


def _get_storage_directories(claim_id: str) -> tuple[str, str, str]:
    """Ensure static directories exist for a given claim."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    static_dir = os.path.join(base_dir, "static")
    claim_upload_dir = os.path.join(static_dir, "uploads", claim_id)
    claim_annotated_dir = os.path.join(static_dir, "annotated", claim_id)

    os.makedirs(claim_upload_dir, exist_ok=True)
    os.makedirs(claim_annotated_dir, exist_ok=True)

    return static_dir, claim_upload_dir, claim_annotated_dir


def _compute_severity_assessment(total_damages: int, counts_by_type: Dict[str, int]) -> str:
    """Rule-based preliminary severity index."""
    if total_damages == 0:
        return "No Damage Detected"

    has_severe = any(
        counts_by_type.get(c, 0) > 0
        for c in ["shattered_glass", "broken_lamp", "flat_tire"]
    )
    dent_count = counts_by_type.get("dent", 0)

    if total_damages >= 5 or dent_count >= 3 or (has_severe and total_damages >= 3):
        return "Severe (Major Structural/Component Impact)"
    elif total_damages >= 2 or has_severe or dent_count >= 1:
        return "Moderate (Body Panel & Component Repair Needed)"
    else:
        return "Minor (Cosmetic Scratches / Surface Level)"


@router.post(
    "/{claim_id}/inspect",
    response_model=ClaimInspectionResponse,
    status_code=status.HTTP_200_OK,
    summary="Upload multi-view vehicle images and execute damage assessment",
)
async def inspect_vehicle_damages(
    claim_id: str,
    files: List[UploadFile] = File(..., description="1 or more vehicle photographs (front, rear, side, etc.)"),
    view_angles: Optional[List[str]] = Form(None, description="Optional view tags corresponding to each file"),
    vehicle_reg_number: Optional[str] = Form(None, description="Optional vehicle registration / plate number"),
    surveyor_notes: Optional[str] = Form(None, description="Optional notes from surveyor"),
    confidence_threshold: float = Form(0.15, description="YOLO confidence threshold [0.05 - 0.95] (default: 0.15 for high recall)"),
    enable_amg: bool = Form(True, description="Enable SAM2 Automatic Mask Generation on vehicle ROI to surface unclassified fragments"),
):
    """
    Process multi-view vehicle images through the high-recall damage detection and segmentation pipeline:
      1. General Vehicle Detector isolates vehicle bounding box ROI crop.
      2. YOLO11m detects damage bounding boxes with high recall (conf=0.15, iou=0.45).
      3. SAM2 segments classified damages using edge-aware padded box prompts (10-15%).
      4. SAM2 Automatic Mask Generation (AMG) runs on the vehicle ROI to surface fragmented pieces.
      5. IoA & IoU Redundancy Filter drops overlapping fragments, keeping novel unclassified regions.
      6. Depth Anything V2 estimates full-scene monocular depth and computes relative surface deformation per instance.
      7. Downstream Safety: Only classified damages enter structural graphs; unclassified regions are isolated.
    """
    clean_claim_id = claim_id.strip()
    if not clean_claim_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid claim_id provided."
        )

    if not files or len(files) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one vehicle image file must be uploaded."
        )

    # Resolve view angles list
    views = view_angles if view_angles else []
    if len(views) < len(files):
        default_labels = ["front", "rear", "left", "right", "close_up", "overview"]
        for idx in range(len(views), len(files)):
            label = default_labels[idx] if idx < len(default_labels) else f"view_{idx + 1}"
            views.append(label)

    detector = get_detector()
    vehicle_detector = get_vehicle_detector()
    segmenter = get_segmenter()
    depth_estimator = get_depth_estimator()
    kg = get_knowledge_graph()
    repo = get_repository()
    _, claim_upload_dir, claim_annotated_dir = _get_storage_directories(clean_claim_id)

    image_results: List[ImageInspectionResult] = []
    all_classified_detections: List[DamageDetection] = []
    all_unclassified_regions: List[DamageDetection] = []
    damages_by_view: Dict[str, List[str]] = {}
    damage_counts_by_type: Dict[str, int] = {}

    for index, file in enumerate(files):
        # Validate content type
        if file.content_type and file.content_type not in ALLOWED_CONTENT_TYPES:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"File '{file.filename}' has unsupported content type '{file.content_type}'. Allowed: JPEG, PNG, WebP.",
            )

        # Read image bytes
        image_bytes = await file.read()
        if len(image_bytes) > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File '{file.filename}' exceeds maximum allowed size of 20MB.",
            )
        if len(image_bytes) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File '{file.filename}' is empty (0 bytes).",
            )

        # Verify image validity
        try:
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to decode image file '{file.filename}': {str(e)}",
            )

        img_w, img_h = pil_image.size
        view_tag = views[index].strip().lower()
        image_uid = f"img_{index + 1}_{uuid.uuid4().hex[:8]}"

        # Save original image
        orig_filename = f"{image_uid}_orig.jpg"
        orig_filepath = os.path.join(claim_upload_dir, orig_filename)
        pil_image.save(orig_filepath, format="JPEG", quality=92)
        orig_url = f"/static/uploads/{clean_claim_id}/{orig_filename}"

        # -------------------------------------------------------------------
        # Step A: Detect Vehicle ROI (COCO-pretrained YOLO11n)
        # -------------------------------------------------------------------
        t_crop_start = time.perf_counter()
        try:
            cropped_vehicle, roi_bbox, crop_coords = vehicle_detector.detect_car(
                image_input=pil_image,
                margin_pct=default_amg_config.vehicle_crop_margin_pct,
                min_confidence=0.25,
            )
        except ValueError as e:
            logger.warning("Car detection validation failed for file '%s': %s", file.filename, str(e))
            raise HTTPException(
                status_code=400,
                detail=str(e),
            )
        vehicle_crop_ms = round((time.perf_counter() - t_crop_start) * 1000.0, 2)

        # -------------------------------------------------------------------
        # Step B: YOLO11m Damage Detection with configurable thresholds
        # -------------------------------------------------------------------
        t_yolo_start = time.perf_counter()
        annotated_pil, detections = detector.predict(
            image_input=pil_image,
            confidence_threshold=confidence_threshold,
            iou_threshold=0.45,
            view_angle=view_tag,
        )
        yolo_detection_ms = round((time.perf_counter() - t_yolo_start) * 1000.0, 2)

        # Save annotated image (bounding boxes)
        annot_filename = f"{image_uid}_annotated.jpg"
        annot_filepath = os.path.join(claim_annotated_dir, annot_filename)
        annotated_pil.save(annot_filepath, format="JPEG", quality=92)
        annot_url = f"/static/annotated/{clean_claim_id}/{annot_filename}"

        # -------------------------------------------------------------------
        # Step C: Padded Box-Prompted SAM2 Segmentation
        # -------------------------------------------------------------------
        t_box_sam2_start = time.perf_counter()
        box_segmented_pil, classified_masks, classified_binary_masks = segmenter.segment_image_detections(
            image_input=pil_image,
            detections=detections,
            vehicle_roi_coords=crop_coords,
            view_angle=view_tag,
        )
        box_prompted_sam2_ms = round((time.perf_counter() - t_box_sam2_start) * 1000.0, 2)

        # -------------------------------------------------------------------
        # Step D: SAM2 Automatic Mask Generation (AMG) on Vehicle ROI (Optional)
        # -------------------------------------------------------------------
        unclassified_detections: List[DamageDetection] = []
        unclassified_binary_masks: List[np.ndarray] = []
        amg_sam2_ms = 0.0

        if enable_amg:
            t_amg_start = time.perf_counter()
            unclass_dets, unclass_masks = segmenter.segment_automatic_roi_masks(
                cropped_vehicle_img=cropped_vehicle,
                crop_coords=crop_coords,
                full_img_size=(img_w, img_h),
                classified_detections=detections,
                classified_binary_masks=classified_binary_masks,
                points_per_side=default_amg_config.points_per_side,
                pred_iou_thresh=default_amg_config.pred_iou_thresh,
                stability_score_thresh=default_amg_config.stability_score_thresh,
                min_crop_area_pct=default_amg_config.min_crop_area_pct,
                max_crop_area_pct=default_amg_config.max_crop_area_pct,
                dedup_iou_thresh=default_amg_config.dedup_iou_thresh,
                ioa_redundancy_thresh=default_amg_config.ioa_redundancy_thresh,
                iou_redundancy_thresh=default_amg_config.iou_redundancy_thresh,
            )
            amg_sam2_ms = round((time.perf_counter() - t_amg_start) * 1000.0, 2)
            unclassified_detections = unclass_dets
            unclassified_binary_masks = unclass_masks

            # Render combined visual overlay containing both classified and unclassified masks
            final_segmented_pil = segmenter.render_combined_overlay(
                base_img=pil_image,
                classified_detections=detections,
                classified_masks=classified_binary_masks,
                unclassified_detections=unclassified_detections,
                unclassified_masks=unclassified_binary_masks,
            )
        else:
            final_segmented_pil = box_segmented_pil

        # Save segmented image (mask overlays)
        segmented_filename = f"{image_uid}_segmented.jpg"
        segmented_filepath = os.path.join(claim_annotated_dir, segmented_filename)
        final_segmented_pil.save(segmented_filepath, format="JPEG", quality=92)
        segmented_url = f"/static/annotated/{clean_claim_id}/{segmented_filename}"

        # -------------------------------------------------------------------
        # Step E: Depth Anything V2 Monocular Depth & Surface Deformation
        # Run on the FULL original image to preserve monocular context.
        # Skip inference if no damages (classified or unclassified) were detected.
        # -------------------------------------------------------------------
        has_damages = (len(detections) > 0 or len(unclassified_detections) > 0)
        depth_estimation_ms = 0.0
        depth_colormap_url = None
        depth_result = None

        if has_damages:
            t_depth_start = time.perf_counter()
            try:
                # 1. Full-image monocular depth estimation & pseudo-colored colormap
                depth_map_norm, depth_colormap_pil, depth_result = depth_estimator.estimate_scene_depth(pil_image)
                depth_estimation_ms = round((time.perf_counter() - t_depth_start) * 1000.0, 2)

                # 2. Save depth colormap
                depth_filename = f"{image_uid}_depth.jpg"
                depth_filepath = os.path.join(claim_annotated_dir, depth_filename)
                depth_colormap_pil.save(depth_filepath, format="JPEG", quality=92)
                depth_colormap_url = f"/static/annotated/{clean_claim_id}/{depth_filename}"
                depth_result.depth_colormap_url = depth_colormap_url

                # 3. Compute instance surface deformation for each classified detection
                all_masks = list(classified_binary_masks) + list(unclassified_binary_masks)
                for idx, det in enumerate(detections):
                    if idx < len(classified_binary_masks):
                        mask = classified_binary_masks[idx]
                        other_masks = [m for j, m in enumerate(all_masks) if not (j == idx and m is mask)]
                        det.deformation = depth_estimator.compute_instance_deformation(
                            depth_map=depth_map_norm,
                            mask=mask,
                            bbox=det.bbox,
                            area_percentage=det.segmentation.area_percentage if det.segmentation else 0.0,
                            damage_type=det.damage_type,
                            other_damage_masks=other_masks,
                        )

                # 4. Compute instance surface deformation for each unclassified AMG candidate
                for idx, u_det in enumerate(unclassified_detections):
                    if idx < len(unclassified_binary_masks):
                        mask = unclassified_binary_masks[idx]
                        mask_global_idx = len(classified_binary_masks) + idx
                        other_masks = [m for j, m in enumerate(all_masks) if not (j == mask_global_idx and m is mask)]
                        u_area_pct = u_det.segmentation.area_percentage if u_det.segmentation else 0.0

                        # Step 4: Minimum-size gate before deformation scoring
                        if u_area_pct < default_amg_config.min_deform_scoring_area_pct:
                            u_det.deformation = DamageDeformation(
                                reference_available=False,
                                relative_deformation_score=None,
                                max_relative_deformation=None,
                                surface_irregularity=None,
                                depth_std=None,
                                deformation_type=None,
                                severity_tier=None,
                                deformation_status="too_small_for_reliable_severity_estimate",
                            )
                        else:
                            u_det.deformation = depth_estimator.compute_instance_deformation(
                                depth_map=depth_map_norm,
                                mask=mask,
                                bbox=u_det.bbox,
                                area_percentage=u_area_pct,
                                damage_type="unclassified",
                                other_damage_masks=other_masks,
                            )
            except Exception as e:
                logger.error(f"[!] Depth estimation failed for image {file.filename}: {e}", exc_info=True)
                depth_result = DepthMapResult(
                    depth_colormap_url=None,
                    mean_scene_depth=0.0,
                    min_scene_depth=0.0,
                    max_scene_depth=0.0,
                    inference_skipped=True,
                )
        else:
            logger.info(f"[*] Claim {clean_claim_id} - Image {file.filename}: Zero damages detected. Skipping Depth Anything V2.")
            depth_result = DepthMapResult(
                depth_colormap_url=None,
                mean_scene_depth=0.0,
                min_scene_depth=0.0,
                max_scene_depth=0.0,
                inference_skipped=True,
            )

        # -------------------------------------------------------------------
        # Step F: Panel Localization & Vehicle Structural Knowledge Graph (Phase 4)
        # Localize each classified damage to an external vehicle panel using
        # 3x3 normalized ROI grid with aspect-ratio disambiguation, then traverse
        # force transmission paths to identify concealed internal components.
        # (Unclassified AMG regions are skipped per safety rule).
        # -------------------------------------------------------------------
        t_kg_start = time.perf_counter()
        for det in detections:
            if det.classified and det.damage_type != "unclassified":
                panel_name = localize_damage_panel(
                    damage_bbox=det.bbox,
                    vehicle_roi_bbox=roi_bbox,
                    view_angle=view_tag,
                )
                det.detected_panel = panel_name

                deform_score = det.deformation.relative_deformation_score if det.deformation else None
                area_pct = det.segmentation.area_percentage if det.segmentation else 1.0

                det.recommendations = kg.recommend_inspections(
                    panel_name=panel_name,
                    deformation_score=deform_score,
                    damaged_area_pct=area_pct,
                    damage_type=det.damage_type,
                )
            else:
                det.detected_panel = "unclassified candidate region"
                det.recommendations = []

        knowledge_graph_ms = round((time.perf_counter() - t_kg_start) * 1000.0, 2)

        total_ms = round(vehicle_crop_ms + yolo_detection_ms + box_prompted_sam2_ms + amg_sam2_ms + depth_estimation_ms + knowledge_graph_ms, 2)
        latency_metrics = PipelineLatencyMetrics(
            yolo_detection_ms=yolo_detection_ms,
            box_prompted_sam2_ms=box_prompted_sam2_ms,
            vehicle_crop_ms=vehicle_crop_ms,
            amg_sam2_ms=amg_sam2_ms,
            depth_estimation_ms=depth_estimation_ms,
            knowledge_graph_ms=knowledge_graph_ms,
            total_pipeline_ms=total_ms,
        )

        logger.info(
            f"[*] Claim {clean_claim_id} - Image {file.filename}: "
            f"YOLO={yolo_detection_ms}ms, Box-SAM2={box_prompted_sam2_ms}ms, "
            f"Crop={vehicle_crop_ms}ms, AMG={amg_sam2_ms}ms, Depth={depth_estimation_ms}ms, "
            f"KG={knowledge_graph_ms}ms, Total={total_ms}ms "
            f"(Classified: {len(detections)}, Unclassified AMG: {len(unclassified_detections)})"
        )

        # -------------------------------------------------------------------
        # REFINEMENT 1: Downstream Safety Filter
        # Only classified damages feed severity index and component aggregates!
        # -------------------------------------------------------------------
        classified_damages = [d for d in detections if d.classified]
        for d in classified_damages:
            all_classified_detections.append(d)
            damage_counts_by_type[d.damage_type] = damage_counts_by_type.get(d.damage_type, 0) + 1

        for u in unclassified_detections:
            all_unclassified_regions.append(u)

        if view_tag not in damages_by_view:
            damages_by_view[view_tag] = []
        for d in classified_damages:
            damages_by_view[view_tag].append(d.damage_type)

        image_results.append(
            ImageInspectionResult(
                image_id=image_uid,
                filename=file.filename or f"image_{index + 1}.jpg",
                view_angle=view_tag,
                original_image_url=orig_url,
                annotated_image_url=annot_url,
                segmented_image_url=segmented_url,
                depth_colormap_url=depth_colormap_url,
                damage_count=len(classified_damages),
                detections=classified_damages,
                unclassified_regions=unclassified_detections,
                vehicle_roi_bbox=roi_bbox,
                amg_enabled=enable_amg,
                latency_metrics=latency_metrics,
                depth_result=depth_result,
            )
        )

    # Build Damage Summary (Computed ONLY on classified damages per Refinement 1)
    total_classified_count = len(all_classified_detections)
    unique_types = sorted(list(damage_counts_by_type.keys()))
    severity = _compute_severity_assessment(total_classified_count, damage_counts_by_type)

    summary = DamageSummary(
        total_damages_count=total_classified_count,
        damage_counts_by_type=damage_counts_by_type,
        unique_damage_types=unique_types,
        damages_by_view=damages_by_view,
        unclassified_candidates_count=len(all_unclassified_regions),
        severity_assessment=severity,
    )

    structural_risk_matrix = kg.aggregate_claim_structural_risks(all_classified_detections)

    existing_claim = repo.get_claim(clean_claim_id) or {}
    existing_docs = existing_claim.get("documents")
    existing_report = existing_claim.get("report")

    response_payload = ClaimInspectionResponse(
        claim_id=clean_claim_id,
        vehicle_reg_number=vehicle_reg_number or existing_claim.get("vehicle_reg_number"),
        surveyor_notes=surveyor_notes or existing_claim.get("surveyor_notes"),
        status="COMPLETED",
        created_at=existing_claim.get("created_at") or datetime.now(timezone.utc).isoformat(),
        images=image_results,
        damage_summary=summary,
        structural_risk_matrix=structural_risk_matrix,
        documents=existing_docs,
        report=existing_report,
    )

    # Persist in storage repository
    repo.save_claim(response_payload.model_dump())

    return response_payload


@router.get(
    "/{claim_id}",
    response_model=ClaimInspectionResponse,
    summary="Retrieve stored claim inspection record and damage summary",
)
async def get_claim_inspection(claim_id: str):
    """Retrieve an existing inspection record by claim ID."""
    clean_claim_id = claim_id.strip()
    repo = get_repository()
    record = repo.get_claim(clean_claim_id)

    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Claim '{clean_claim_id}' was not found.",
        )

    return ClaimInspectionResponse(**record)


@router.get(
    "",
    response_model=List[ClaimInspectionResponse],
    summary="List all processed vehicle inspection claims",
)
async def list_all_claims(limit: int = Query(50, ge=1, le=200)):
    """Retrieve recent claim records."""
    repo = get_repository()
    claims = repo.list_claims(limit=limit)
    return [ClaimInspectionResponse(**c) for c in claims]


@router.delete(
    "/{claim_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete a claim inspection record",
)
async def delete_claim_inspection(claim_id: str):
    """Delete a claim record from the repository."""
    clean_claim_id = claim_id.strip()
    repo = get_repository()
    deleted = repo.delete_claim(clean_claim_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Claim '{clean_claim_id}' not found.",
        )
    return {"message": f"Claim '{clean_claim_id}' deleted successfully."}


# ---------------------------------------------------------------------------
# Phase 5 Endpoints: Document Understanding (OCR) & Survey Report Synthesis
# ---------------------------------------------------------------------------

@router.post(
    "/{claim_id}/documents",
    response_model=DocumentExtractionResult,
    status_code=status.HTTP_200_OK,
    summary="Upload and extract structured fields from RC, Policy, or Driving Licence",
)
async def upload_claim_document(
    claim_id: str,
    file: UploadFile = File(..., description="Document image file (JPEG, PNG, WebP)"),
    document_type: str = Form(..., description="Document type: rc_book, insurance_policy, or driving_licence"),
):
    """
    Process an uploaded claim verification document via Groq Vision LLM:
      1. Validates document type and file extension.
      2. Saves original document image to disk under static uploads.
      3. Dispatches image bytes to Groq Vision OCR with strict anti-hallucination prompt.
      4. Executes independent formatting and sanity checks on extracted fields.
      5. Persists extraction record to claim repository and cross-checks registration numbers.
    """
    clean_claim_id = claim_id.strip()
    clean_doc_type = document_type.strip().lower()

    if not clean_claim_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid claim_id.")

    valid_types = {"rc_book", "insurance_policy", "driving_licence", "rc", "policy", "licence", "license"}
    if clean_doc_type not in valid_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid document_type '{document_type}'. Allowed: rc_book, insurance_policy, driving_licence.",
        )

    # Prepare storage directory
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    static_dir = os.path.join(base_dir, "static")
    doc_dir = os.path.join(static_dir, "uploads", clean_claim_id, "documents")
    os.makedirs(doc_dir, exist_ok=True)

    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")

    # Save to disk
    file_ext = os.path.splitext(file.filename or "document.jpg")[1] or ".jpg"
    safe_filename = f"{clean_doc_type}_{uuid.uuid4().hex[:6]}{file_ext}"
    saved_file_path = os.path.join(doc_dir, safe_filename)
    with open(saved_file_path, "wb") as f:
        f.write(image_bytes)

    doc_static_url = f"/static/uploads/{clean_claim_id}/documents/{safe_filename}"

    # Execute two-stage PaddleOCR + Groq Text hybrid extraction with validation (never use mock for real uploads)
    extraction_result = extract_document_hybrid(
        image_bytes=image_bytes,
        image_path=saved_file_path,
        document_type=clean_doc_type,
        filename=file.filename or safe_filename,
        mime_type=file.content_type or "image/jpeg",
        image_url=doc_static_url,
        allow_mock=False,
    )

    # Persist in storage repository
    repo = get_repository()
    repo.update_claim_document(clean_claim_id, clean_doc_type, extraction_result.model_dump())

    return extraction_result


@router.get(
    "/{claim_id}/documents",
    response_model=ClaimDocuments,
    summary="Retrieve all verified documents for an insurance claim",
)
async def get_claim_documents(claim_id: str):
    """Retrieve all KYC and insurance documents associated with a claim."""
    clean_claim_id = claim_id.strip()
    repo = get_repository()
    record = repo.get_claim(clean_claim_id)

    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Claim '{clean_claim_id}' not found.")

    docs_data = record.get("documents") or {}
    return ClaimDocuments(**docs_data)


@router.put(
    "/{claim_id}/documents/{document_type}",
    response_model=DocumentExtractionResult,
    summary="Manually confirm or override extracted document field values",
)
async def update_claim_document_fields(
    claim_id: str,
    document_type: str,
    fields_update: Dict[str, Any] = Body(..., description="Dictionary of field names to updated string values"),
):
    """
    Allow the human surveyor to correct or confirm AI-extracted fields
    (Human-in-the-Loop Requirement).
    """
    clean_claim_id = claim_id.strip()
    clean_doc_type = document_type.strip().lower()
    repo = get_repository()
    record = repo.get_claim(clean_claim_id)

    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Claim '{clean_claim_id}' not found.")

    docs = record.get("documents") or {}
    norm_key = (
        "rc_book" if "rc" in clean_doc_type or "reg" in clean_doc_type
        else "insurance_policy" if "policy" in clean_doc_type or "insur" in clean_doc_type
        else "driving_licence"
    )

    doc_entry = docs.get(norm_key)
    if not doc_entry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No document record found for '{document_type}' in claim '{clean_claim_id}'.",
        )

    existing_fields = doc_entry.get("fields", {})

    # Apply surveyor manual corrections
    for k, v in fields_update.items():
        val_str = str(v).strip() if v is not None else None
        if k in existing_fields:
            existing_fields[k]["value"] = val_str
            existing_fields[k]["confidence"] = 1.0
            existing_fields[k]["is_valid"] = True
            existing_fields[k]["validation_error"] = None
            existing_fields[k]["surveyor_overridden"] = True
        else:
            existing_fields[k] = {
                "value": val_str,
                "source_snippet": "Manual Surveyor Entry",
                "confidence": 1.0,
                "is_valid": True,
                "validation_error": None,
                "surveyor_overridden": True,
            }

    doc_entry["fields"] = existing_fields
    doc_entry["needs_manual_review"] = False
    doc_entry["status"] = "SUCCESS"
    doc_entry["validation_warnings"] = [w for w in doc_entry.get("validation_warnings", []) if "Required" not in w]

    # Update in repository
    repo.update_claim_document(clean_claim_id, norm_key, doc_entry)

    return DocumentExtractionResult(**doc_entry)


@router.post(
    "/{claim_id}/generate-report",
    response_model=LLMSurveyReport,
    status_code=status.HTTP_200_OK,
    summary="Synthesize Explainable Insurance Survey Report using Groq LLM",
)
async def generate_claim_survey_report(
    claim_id: str,
    allow_mock: Optional[bool] = Query(None, description="Explicit mock toggle; if None, adopts client default"),
):
    """
    Final Pipeline Step:
      Synthesizes visual damage detections, SAM2 segmentation metrics,
      Depth Anything V2 monocular deformations, Neo4j knowledge graph recommendations,
      and verified document credentials into an explainable, audit-ready survey dossier.
    """
    clean_claim_id = claim_id.strip()
    repo = get_repository()
    record = repo.get_claim(clean_claim_id)

    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Claim '{clean_claim_id}' not found.")

    claim_data = ClaimInspectionResponse(**record)
    docs_data = record.get("documents")
    documents = ClaimDocuments(**docs_data) if docs_data else None

    # Synthesize report via Groq LLM
    try:
        mock_flag = allow_mock if allow_mock is not None else get_groq_client().is_mock_mode()
        report = generate_survey_report(claim_data=claim_data, documents=documents, allow_mock=mock_flag)
    except (GroqClientError, ReportGenerationError) as e:
        logger.error(f"[X] Report generation failed for claim '{clean_claim_id}': {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Survey report generation failed: {e}. No fabricated fallback was generated.",
        )

    # Persist in storage repository
    repo.update_claim_report(clean_claim_id, report.model_dump())

    return report


@router.get(
    "/{claim_id}/report",
    response_model=LLMSurveyReport,
    summary="Retrieve existing generated survey report for a claim",
)
async def get_claim_survey_report(claim_id: str):
    """Fetch stored survey report dossier for an insurance claim."""
    clean_claim_id = claim_id.strip()
    repo = get_repository()
    record = repo.get_claim(clean_claim_id)

    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Claim '{clean_claim_id}' not found.")

    report_data = record.get("report")
    if not report_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Survey report has not been generated yet for claim '{clean_claim_id}'.",
        )

    return LLMSurveyReport(**report_data)


@router.post(
    "/{claim_id}/finalize-report",
    response_model=LLMSurveyReport,
    summary="Surveyor sign-off and finalize the AI-assisted draft survey report",
)
async def finalize_survey_report(
    claim_id: str,
    signoff_notes: Optional[str] = Body(None, embed=True, description="Surveyor remarks or final approval notes"),
):
    """Mark an AI-assisted survey report as finalized with human surveyor sign-off."""
    clean_claim_id = claim_id.strip()
    repo = get_repository()
    record = repo.get_claim(clean_claim_id)

    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Claim '{clean_claim_id}' not found.")

    report_data = record.get("report")
    if not report_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No survey report exists to finalize for claim '{clean_claim_id}'.",
        )

    report_data["status"] = "FINALIZED"
    report_data["surveyor_signoff_notes"] = signoff_notes or "Approved and verified by Insurance Claim Surveyor."
    report_data["signed_off_at"] = datetime.now(timezone.utc).isoformat()

    repo.update_claim_report(clean_claim_id, report_data)

    return LLMSurveyReport(**report_data)


if __name__ == "__main__":
    import uvicorn
    print("ℹ️ Note: routes_inspection.py contains APIRouter endpoints and is part of the FastAPI backend.")
    print("🚀 Starting the full application server via backend.app.main:app at http://localhost:8000 ...")
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)


