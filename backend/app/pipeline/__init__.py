"""Pipeline modules package initialization."""
from .document_ocr import extract_document_fields
from .damage_segmentation import segment_damaged_parts
from .depth_estimation import estimate_damage_depth
from .knowledge_graph import recommend_inspections
from .report_generator import generate_claim_report

__all__ = [
    "extract_document_fields",
    "segment_damaged_parts",
    "estimate_damage_depth",
    "recommend_inspections",
    "generate_claim_report",
]
