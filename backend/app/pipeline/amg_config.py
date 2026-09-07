"""
Configuration module for SAM2 Automatic Mask Generation (AMG) unclassified pathway.
Surfaces all tuning parameters as named, configurable constants with environment variable overrides.
"""

import os
from dataclasses import dataclass


@dataclass
class AMGConfig:
    """Named configuration parameters for SAM2 AMG generation, filtering, and presentation."""

    # Step 1: AMG Generation Hyperparameters
    # Reduced points_per_side produces fewer near-duplicate candidate proposals
    points_per_side: int = int(os.getenv("AMG_POINTS_PER_SIDE", "16"))
    # Higher IoU and stability score thresholds discard low-quality proposals at source
    pred_iou_thresh: float = float(os.getenv("AMG_PRED_IOU_THRESH", "0.91"))
    stability_score_thresh: float = float(os.getenv("AMG_STABILITY_SCORE_THRESH", "0.96"))

    # Step 2: Candidate Mask Area Floor & Ceiling
    # Floor: percentage of the vehicle ROI crop area (e.g. 1.0% floor drops sub-1% pixel noise)
    min_crop_area_pct: float = float(os.getenv("AMG_MIN_CROP_AREA_PCT", "1.0"))
    # Ceiling: percentage of crop area to discard entire car body or background panel masks
    max_crop_area_pct: float = float(os.getenv("AMG_MAX_CROP_AREA_PCT", "35.0"))

    # Step 3: Overlap Deduplication
    # Pairwise IoU threshold: if two surviving masks overlap > threshold, retain only higher stability/larger
    dedup_iou_thresh: float = float(os.getenv("AMG_DEDUP_IOU_THRESH", "0.80"))

    # Step 4: Phase 3 Deformation Scoring Size Gate
    # Fragments smaller than this image area percentage skip depth deformation scoring
    min_deform_scoring_area_pct: float = float(os.getenv("MIN_DEFORM_SCORING_AREA_PCT", "1.0"))

    # Step 5: Vehicle ROI Crop Margin
    # Tighter margin prevents background passersby, adjacent cars, and pavement from entering AMG
    vehicle_crop_margin_pct: float = float(os.getenv("VEHICLE_CROP_MARGIN_PCT", "0.02"))

    # Step 6: Surveyor UI Display Cap
    # Maximum number of top unclassified fragments displayed individually before collapsing remainder
    display_cap_n: int = int(os.getenv("TOP_N_UNCLASSIFIED_DISPLAY", "5"))

    # Classified vs AMG Reconciliation Thresholds
    ioa_redundancy_thresh: float = float(os.getenv("AMG_IOA_REDUNDANCY_THRESH", "0.60"))
    iou_redundancy_thresh: float = float(os.getenv("AMG_IOU_REDUNDANCY_THRESH", "0.35"))


# Default global instance
default_amg_config = AMGConfig()
