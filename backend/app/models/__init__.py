"""Models package initialization."""
from .schemas import (
    BoundingBox,
    DamageDetection,
    ImageInspectionResult,
    DamageSummary,
    ClaimInspectionResponse,
    DocumentFields,
    SegmentationMask,
    DepthMapResult,
    InspectionRecommendation,
    LLMExplainableReport,
)

__all__ = [
    "BoundingBox",
    "DamageDetection",
    "ImageInspectionResult",
    "DamageSummary",
    "ClaimInspectionResponse",
    "DocumentFields",
    "SegmentationMask",
    "DepthMapResult",
    "InspectionRecommendation",
    "LLMExplainableReport",
]
