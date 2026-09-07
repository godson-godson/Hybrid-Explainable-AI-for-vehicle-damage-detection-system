"""
Pydantic Schemas for Vehicle Damage Assessment and Insurance Claim Assistance.
Defines data contracts for Phase 1 vision detection and future module stubs.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Phase 1 Vision & Damage Schemas
# ---------------------------------------------------------------------------

class BoundingBox(BaseModel):
    """Bounding box coordinates in absolute pixels and normalized scale."""
    x1: float = Field(..., description="Top-left X coordinate (pixels)")
    y1: float = Field(..., description="Top-left Y coordinate (pixels)")
    x2: float = Field(..., description="Bottom-right X coordinate (pixels)")
    y2: float = Field(..., description="Bottom-right Y coordinate (pixels)")
    width: float = Field(..., description="Box width (pixels)")
    height: float = Field(..., description="Box height (pixels)")
    norm_x1: Optional[float] = Field(None, description="Normalized top-left X [0.0 - 1.0]")
    norm_y1: Optional[float] = Field(None, description="Normalized top-left Y [0.0 - 1.0]")
    norm_x2: Optional[float] = Field(None, description="Normalized bottom-right X [0.0 - 1.0]")
    norm_y2: Optional[float] = Field(None, description="Normalized bottom-right Y [0.0 - 1.0]")


class SegmentationMask(BaseModel):
    """Schema for SAM2 Damage Segmentation Module."""
    damage_type: str = Field(..., description="Target damage class")
    polygon_points: List[List[float]] = Field(default_factory=list, description="Contour polygon coordinates [[x,y],...]")
    area_pixels: int = Field(..., description="Pixel area of the segmented mask")
    area_percentage: float = Field(0.0, description="Damaged area as a percentage of total image area [0.0 - 100.0]")
    relative_surface_ratio: float = Field(0.0, description="Damaged area ratio relative to full image [0.0 - 1.0]")
    mask_confidence: Optional[float] = Field(None, description="SAM2 predicted IoU score")
    part_name: Optional[str] = Field(None, description="Associated vehicle body part (e.g. front_bumper, door)")


class DamageDeformation(BaseModel):
    """
    Relative surface deformation features derived from Depth Anything V2.
    IMPORTANT: All scores are unitless relative metrics, not metric millimeters or centimeters.
    """
    reference_available: bool = Field(True, description="True if a valid undamaged reference ring was extracted around the damage")
    relative_deformation_score: Optional[float] = Field(None, description="Unitless relative indentation depth (+ = recessed/dent, - = protruding/bent outward)")
    max_relative_deformation: Optional[float] = Field(None, description="Unitless 95th-percentile peak deformation relative to reference")
    surface_irregularity: Optional[float] = Field(None, description="Depth variance within the damage mask (rough crumple vs flat scratch)")
    depth_std: Optional[float] = Field(None, description="Depth standard deviation within the damage mask")
    deformation_type: Optional[str] = Field(None, description="Deformation characterization: 'recessed' (dent), 'protruding', or 'planar' (scratch)")
    severity_tier: Optional[str] = Field(None, description="Heuristic proxy severity tier: 'minor', 'moderate', or 'severe'")
    deformation_status: str = Field("computed", description="Calculation status: 'computed' or 'reference_unavailable'")


class InspectionRecommendation(BaseModel):
    """Concealed internal vehicle component flagged for physical inspection via structural graph traversal."""
    component_name: str = Field(..., description="Vehicle internal component name (e.g. radiator support assembly)")
    source_panel: str = Field("unknown", description="External panel where primary impact/damage was localized")
    impact_zone: str = Field("front", description="Vehicle zone: front, rear, left, right")
    risk_score: float = Field(0.0, description="Heuristic force transmission risk probability [0.0 - 1.0]")
    safety_risk_level: str = Field("LOW", description="Risk tier: HIGH (>=0.60), MEDIUM (0.35-0.59), LOW (<0.35)")
    recommended_action: str = Field("Visual Inspection", description="Action: Teardown, Borescope Check, Secondary Check")
    estimated_labor_hours: float = Field(1.0, description="Estimated technician inspection labor hours")
    load_path: List[str] = Field(default_factory=list, description="Sequence of graph nodes forming the collision load path")
    rationale: str = Field(..., description="Explainable reasoning log narrative detailing force propagation")
    detected_damage: str = Field("dent", description="Primary surface damage type triggering the load path")


class DamageDetection(BaseModel):
    """Individual damage detection instance from YOLO11m with optional SAM2 segmentation or unclassified AMG region."""
    class_id: int = Field(..., description="Numeric class index from YOLO model (-1 for unclassified)")
    damage_type: str = Field(..., description="Damage class name (e.g. dent, scratch, broken_lamp, unclassified)")
    confidence: float = Field(..., description="Prediction confidence score [0.0 - 1.0]")
    confidence_tier: str = Field("high", description="Confidence tier: 'high' (>=0.50) or 'low' (<0.50) to guide surveyor scrutiny")
    classified: bool = Field(True, description="True for YOLO-classified detections, False for unclassified candidate regions")
    label: Optional[str] = Field(None, description="Human-readable display label (e.g. 'unclassified — needs review')")
    bbox: BoundingBox = Field(..., description="Bounding box localization")
    view_angle: Optional[str] = Field(None, description="View angle tag (e.g. front, rear, left, right, close-up)")
    detected_panel: Optional[str] = Field(None, description="Localized external vehicle panel from 3x3 ROI grid heuristic")
    segmentation: Optional[SegmentationMask] = Field(None, description="Pixel-level SAM2 segmentation mask and metrics")
    deformation: Optional[DamageDeformation] = Field(None, description="Relative depth surface deformation metrics from Depth Anything V2")
    recommendations: List[InspectionRecommendation] = Field(default_factory=list, description="Internal components at risk from this specific damage instance")


class DepthMapResult(BaseModel):
    """Monocular scene depth estimation result from Depth Anything V2."""
    depth_colormap_url: Optional[str] = Field(None, description="URL of generated pseudo-colored depth colormap image")
    mean_scene_depth: float = Field(0.0, description="Mean relative depth value across the full scene")
    min_scene_depth: float = Field(0.0, description="Minimum relative depth value (closest point)")
    max_scene_depth: float = Field(0.0, description="Maximum relative depth value (farthest point)")
    inference_skipped: bool = Field(False, description="True if depth estimation was skipped because no damage detections were found")


class PipelineLatencyMetrics(BaseModel):
    """Execution timing profiling in milliseconds for transparency and benchmarking."""
    yolo_detection_ms: float = Field(0.0, description="YOLO11m damage detection execution time (ms)")
    box_prompted_sam2_ms: float = Field(0.0, description="Padded box-prompted SAM2 segmentation time (ms)")
    vehicle_crop_ms: float = Field(0.0, description="YOLO vehicle ROI detection and crop time (ms)")
    amg_sam2_ms: float = Field(0.0, description="SAM2 Automatic Mask Generation on ROI time (ms)")
    depth_estimation_ms: float = Field(0.0, description="Depth Anything V2 monocular depth & deformation time (ms)")
    knowledge_graph_ms: float = Field(0.0, description="Neo4j / Structural Knowledge Graph reasoning latency (ms)")
    total_pipeline_ms: float = Field(0.0, description="Total image pipeline processing time (ms)")


class ImageInspectionResult(BaseModel):
    """Inference result for a single uploaded vehicle photograph."""
    image_id: str = Field(..., description="Unique identifier for the image")
    filename: str = Field(..., description="Original uploaded filename")
    view_angle: str = Field("unknown", description="Vehicle view perspective")
    original_image_url: str = Field(..., description="Static URL to original image")
    annotated_image_url: str = Field(..., description="Static URL to annotated image with bounding boxes")
    segmented_image_url: Optional[str] = Field(None, description="Static URL to annotated image with segmentation mask overlays")
    depth_colormap_url: Optional[str] = Field(None, description="Static URL to Depth Anything V2 pseudo-colored colormap")
    damage_count: int = Field(..., description="Total classified detections in this image")
    detections: List[DamageDetection] = Field(default_factory=list, description="List of classified damage detections")
    unclassified_regions: List[DamageDetection] = Field(default_factory=list, description="Candidate damage regions surfaced by SAM2 AMG")
    vehicle_roi_bbox: Optional[BoundingBox] = Field(None, description="Vehicle bounding box ROI crop if detected")
    amg_enabled: bool = Field(True, description="Whether SAM2 Automatic Mask Generation was executed")
    latency_metrics: Optional[PipelineLatencyMetrics] = Field(None, description="Latency profiling breakdown")
    depth_result: Optional[DepthMapResult] = Field(None, description="Scene depth estimation metrics")


class DamageSummary(BaseModel):
    """Aggregated summary of damages across all uploaded multi-view photos."""
    total_damages_count: int = Field(..., description="Total classified damage instances detected")
    damage_counts_by_type: Dict[str, int] = Field(default_factory=dict, description="Damage count grouped by class")
    unique_damage_types: List[str] = Field(default_factory=list, description="Unique damage classes found")
    damages_by_view: Dict[str, List[str]] = Field(default_factory=dict, description="Damage classes mapped per view")
    unclassified_candidates_count: int = Field(0, description="Total unclassified candidate damage fragments surfaced by AMG")
    severity_assessment: str = Field(..., description="Rule-based preliminary severity index (Minor, Moderate, Severe)")


# ---------------------------------------------------------------------------
# Phase 5: Document Understanding (OCR) & Explainable Report Schemas
# ---------------------------------------------------------------------------

class ExtractedFieldItem(BaseModel):
    """Individual extracted field with source snippet and validation tracking."""
    value: Optional[str] = Field(None, description="Extracted string value (null if illegible/absent)")
    source_snippet: Optional[str] = Field(None, description="Literal character snippet read from original image")
    confidence: float = Field(0.0, description="Extraction confidence score [0.0 - 1.0]")
    is_valid: bool = Field(True, description="True if field satisfies heuristic formatting checks")
    validation_error: Optional[str] = Field(None, description="Explanation if heuristic validation flagged this field")
    surveyor_overridden: bool = Field(False, description="True if field value was manually confirmed/modified by surveyor")


class DocumentExtractionResult(BaseModel):
    """Complete extraction and validation result for an uploaded document."""
    document_id: str = Field(..., description="Unique document upload ID")
    document_type: str = Field(..., description="rc_book, insurance_policy, or driving_licence")
    filename: str = Field(..., description="Original filename of uploaded document")
    file_url: Optional[str] = Field(None, description="Static URL to uploaded document image")
    uploaded_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat(), description="Upload ISO timestamp")
    fields: Dict[str, ExtractedFieldItem] = Field(default_factory=dict, description="Dictionary of extracted field objects")
    needs_manual_review: bool = Field(False, description="Flagged true if required fields are missing, malformed, or low-confidence")
    validation_warnings: List[str] = Field(default_factory=list, description="List of specific heuristic validation alerts")
    latency_ms: float = Field(0.0, description="Vision LLM extraction execution time (ms)")
    extraction_method: str = Field("paddle_hybrid", description="Method: paddle_hybrid, vision_fallback, surveyor_override")
    raw_ocr_corpus: Optional[str] = Field(None, description="Raw OCR corpus extracted by PaddleOCR prior to LLM semantic mapping")
    status: str = Field("SUCCESS", description="Extraction status: SUCCESS, NEEDS_REVIEW, EXTRACTION_FAILED")


class ClaimDocuments(BaseModel):
    """Aggregated container of verified documents for a claim."""
    rc_book: Optional[DocumentExtractionResult] = Field(None, description="Registration Certificate record")
    insurance_policy: Optional[DocumentExtractionResult] = Field(None, description="Insurance Policy record")
    driving_licence: Optional[DocumentExtractionResult] = Field(None, description="Driving Licence record")
    cross_validation_passed: bool = Field(False, description="True if registration numbers match across documents")
    cross_validation_notes: List[str] = Field(default_factory=list, description="Cross-document consistency audit notes")


class LLMSurveyReport(BaseModel):
    """Schema for Chapter 4.6.5 Explainable Survey Report Dossier."""
    report_id: str = Field(..., description="Unique generated report dossier ID")
    claim_id: str = Field(..., description="Associated claim ID")
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat(), description="Generation timestamp")
    model_used: str = Field(..., description="Underlying Groq model utilized for synthesis")
    latency_ms: float = Field(0.0, description="Report generation execution time in milliseconds")
    executive_summary: str = Field(..., description="Surveyor-grade executive summary narrative")
    damage_summary: Dict[str, Any] = Field(default_factory=dict, description="Structured damage summary counts and panels")
    surface_deformation_narrative: str = Field(..., description="Synthesis of Depth Anything V2 indentation metrics")
    internal_inspection_plan: List[Dict[str, Any]] = Field(default_factory=list, description="Ranked internal components directly from Knowledge Graph")
    explainable_reasoning: str = Field(..., description="Plain-language narrative tying visual impact to internal component risks")
    claim_disposition: str = Field(..., description="Recommended verdict: Approve, Conditional Approval, Teardown Required, Reject")
    estimated_repair_cost_min: float = Field(0.0, description="Estimated minimum repair expense in INR")
    estimated_repair_cost_max: float = Field(0.0, description="Estimated maximum repair expense in INR")
    surveyor_action_items: List[str] = Field(default_factory=list, description="Immediate checklist items for physical surveyor")
    markdown_dossier: str = Field(..., description="Publication-grade Markdown report for human display and printing")
    status: str = Field("DRAFT", description="Dossier status: DRAFT, FINALIZED")
    surveyor_signoff_notes: Optional[str] = Field(None, description="Optional surveyor signoff / approval remarks")
    signed_off_at: Optional[str] = Field(None, description="Timestamp when report was finalized")


class ClaimInspectionResponse(BaseModel):
    """Complete claim assessment response returned by the backend."""
    claim_id: str = Field(..., description="Unique Claim/Session Identifier")
    vehicle_reg_number: Optional[str] = Field(None, description="Vehicle Registration / License Plate Number")
    surveyor_notes: Optional[str] = Field(None, description="Optional surveyor notes")
    status: str = Field("COMPLETED", description="Assessment pipeline status")
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat(), description="ISO Timestamp")
    images: List[ImageInspectionResult] = Field(default_factory=list, description="Per-image inspection results")
    damage_summary: Optional[DamageSummary] = Field(None, description="Aggregated damage summary")
    structural_risk_matrix: List[InspectionRecommendation] = Field(default_factory=list, description="Claim-level aggregated & ranked internal component inspection risk matrix")
    documents: Optional[ClaimDocuments] = Field(None, description="Verified claim documents (RC, Policy, Driving Licence)")
    report: Optional[LLMSurveyReport] = Field(None, description="Generated AI-assisted survey dossier")


# Legacy stubs retained for backwards compatibility
class DocumentFields(BaseModel):
    """Legacy stub schema."""
    document_type: str = Field(..., description="Document type")
    policy_number: Optional[str] = None
    claim_number: Optional[str] = None
    vehicle_reg_number: Optional[str] = None
    claimant_name: Optional[str] = None
    date_of_incident: Optional[str] = None
    insured_declared_value: Optional[float] = None
    confidence_score: float = 0.0
    raw_text: Optional[str] = None


class LLMExplainableReport(BaseModel):
    """Legacy stub schema."""
    report_id: str
    executive_summary: str
    damage_breakdown: List[Dict[str, str]] = Field(default_factory=list)
    causality_consistency_analysis: str = ""
    estimated_repair_cost_min: float = 0.0
    estimated_repair_cost_max: float = 0.0
    surveyor_verdict: str = ""


# Rebuild model to resolve any remaining forward annotations
ClaimInspectionResponse.model_rebuild()

