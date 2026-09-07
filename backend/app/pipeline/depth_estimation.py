"""
===============================================================================
DEPTH ANYTHING V2 MONOCULAR DEPTH ESTIMATION & SURFACE DEFORMATION MODULE
===============================================================================

SIGN CONVENTION (EMPIRICALLY VERIFIED ON DEPTH ANYTHING V2):
-------------------------------------------------------------------------------
Raw output tensor values from Depth Anything V2 represent relative DISTANCE
from the camera sensor:
  - HIGHER VALUES = FARTHER FROM CAMERA (pushed inward / recessed into body)
  - LOWER VALUES  = CLOSER TO CAMERA (protruding outward / toward camera)

Therefore, when comparing the damaged mask region against the immediately
surrounding undamaged reference panel ring:

    relative_deformation_score = median(Depth_mask) - median(Depth_reference)

  - When score > 0 : Damaged surface is RECESSED / INDENTED (e.g., DENT)
  - When score < 0 : Damaged surface is PROTRUDING / BENT OUTWARD
  - When score ≈ 0 : Damaged surface is PLANAR / SURFACE-LEVEL (e.g., SCRATCH)

IMPORTANT NOTE ON UNITS:
All deformation measurements are strictly UNITLESS, RELATIVE comparative
scores derived from monocular depth predictions. They are NOT metric millimeters
or centimeters, as monocular depth lacks absolute metric scale calibration.
===============================================================================
"""

import io
import logging
import os
from typing import List, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from ..models.schemas import BoundingBox, DamageDeformation, DepthMapResult

logger = logging.getLogger("depth_estimation")
logging.basicConfig(level=logging.INFO)

# Default model configuration
DEFAULT_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"


def compute_deformation_severity_tier(
    relative_deformation: Optional[float],
    depth_std: Optional[float],
    area_percentage: float = 0.0,
) -> str:
    """
    Heuristic Proxy Severity Tier Function.

    HEURISTIC RATIONALE & TUNING GUIDELINES:
    This function combines the unitless relative deformation score (indentation depth),
    depth standard deviation (surface roughness/crumple), and damaged surface area
    percentage into an advisory severity tier: 'minor', 'moderate', or 'severe'.

    NOTE:
    This is an assistive heuristic proxy for insurance surveyor decision support,
    NOT a deterministic or calibrated physical damage metric.
    Surveyors should treat this as a prioritisation indicator.

    Thresholds:
      - Severe: Deep structural recess (|score| >= 0.35) OR moderate recess (|score| >= 0.18)
        combined with substantial panel deformation (area >= 1.5% or depth_std >= 0.20).
      - Moderate: Noticeable indentation (|score| >= 0.12) OR crumple texture (depth_std >= 0.15)
        or surface area >= 0.5%.
      - Minor: Superficial surface scratch or minimal displacement (|score| < 0.12).
    """
    if relative_deformation is None:
        return "minor"

    abs_def = abs(relative_deformation)
    std = depth_std if depth_std is not None else 0.0

    # Guard against spurious SEVERE on small boundary/noise fragments:
    # Severe structural rating requires substantial damaged area (>= 1.5%) OR confirmed physical crumple (std >= 0.20)
    if (abs_def >= 0.35 and (area_percentage >= 1.5 or std >= 0.15)) or (abs_def >= 0.18 and (area_percentage >= 1.5 and std >= 0.20)):
        return "severe"
    elif abs_def >= 0.12 or std >= 0.15 or (abs_def >= 0.06 and area_percentage >= 0.5):
        return "moderate"
    else:
        return "minor"


class DepthAnythingV2Estimator:
    """
    Singleton wrapper for Depth Anything V2 monocular relative depth estimation.
    Loads model weights once at startup, executes inference over the full scene image
    to preserve global context, and computes per-damage deformation metrics.
    """

    def __init__(self, model_id: Optional[str] = None, device: Optional[str] = None):
        self.model_id = model_id or os.getenv("DEPTH_ANYTHING_MODEL", DEFAULT_MODEL_ID)
        self.model_type = "Depth-Anything-V2"
        self.device = self._resolve_device(device)

        logger.info(f"[*] Initializing Depth Anything V2 from: {self.model_id} on {self.device}")
        self.image_processor = AutoImageProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(self.model_id)
        self.model.to(self.device)
        self.model.eval()
        logger.info("[+] Depth Anything V2 Estimator loaded successfully.")

    def _resolve_device(self, user_device: Optional[str]) -> str:
        """Resolve inference device (CUDA if available, else CPU)."""
        if user_device:
            dev = user_device.strip().lower()
        else:
            dev = os.getenv("DEPTH_DEVICE", "").strip().lower()

        if dev in ["cuda", "cuda:0"] and torch.cuda.is_available():
            return "cuda"
        elif torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def estimate_scene_depth(
        self,
        image_input: Union[bytes, Image.Image, np.ndarray, str],
    ) -> Tuple[np.ndarray, Image.Image, DepthMapResult]:
        """
        Execute monocular relative depth estimation over the FULL scene image.

        CRITICAL CONTEXT REQUIREMENT (Step 3):
        Monocular depth estimation models rely on full-scene semantic context (horizon,
        ground contact, lighting, vehicle perspective) to judge depth accurately.
        Inference must run on the full original image, NOT a cropped damage patch.

        Returns:
            Tuple containing:
                - Raw 2D float32 depth map array of shape (H, W).
                - Pseudo-colored PIL Image (Inferno colormap) for visualization.
                - DepthMapResult schema containing scene depth statistics.
        """
        if isinstance(image_input, bytes):
            pil_img = Image.open(io.BytesIO(image_input)).convert("RGB")
        elif isinstance(image_input, str):
            pil_img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            pil_img = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        img_w, img_h = pil_img.size

        # Preprocess full image
        inputs = self.image_processor(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self.model(**inputs)
            post_processed = self.image_processor.post_process_depth_estimation(
                outputs,
                target_sizes=[(img_h, img_w)],
            )

        depth_tensor = post_processed[0]["predicted_depth"]
        depth_np = depth_tensor.detach().cpu().numpy().astype(np.float32)

        # Scene depth metrics
        min_val = float(depth_np.min())
        max_val = float(depth_np.max())
        mean_val = float(depth_np.mean())

        # Generate normalized pseudo-colored colormap for visualization (Inferno)
        denom = (max_val - min_val) if (max_val - min_val) > 1e-6 else 1.0
        normalized_depth = np.clip((depth_np - min_val) / denom, 0.0, 1.0)
        depth_uint8 = (normalized_depth * 255.0).astype(np.uint8)

        # Invert so closer objects (smaller distance) appear hot/bright in colormap
        colormap_bgr = cv2.applyColorMap(255 - depth_uint8, cv2.COLORMAP_INFERNO)
        colormap_rgb = cv2.cvtColor(colormap_bgr, cv2.COLOR_BGR2RGB)
        colormap_pil = Image.fromarray(colormap_rgb)

        result_schema = DepthMapResult(
            depth_colormap_url=None,
            mean_scene_depth=round(mean_val, 4),
            min_scene_depth=round(min_val, 4),
            max_scene_depth=round(max_val, 4),
            inference_skipped=False,
        )

        return normalized_depth, colormap_pil, result_schema

    def compute_instance_deformation(
        self,
        depth_map: np.ndarray,
        target_mask: Optional[np.ndarray] = None,
        all_other_masks: Optional[List[np.ndarray]] = None,
        bbox: Optional[BoundingBox] = None,
        area_percentage: float = 0.0,
        dilation_ratio: float = 0.20,
        min_reference_pixels: int = 30,
        border_margin_px: int = 5,
        mask: Optional[np.ndarray] = None,
        other_damage_masks: Optional[List[np.ndarray]] = None,
        damage_type: Optional[str] = None,
    ) -> DamageDeformation:
        """
        Compute relative surface deformation for a single damage mask.

        ALGORITHM (Step 4):
        1. Dilate target mask by a margin proportional to damage bounding box dimensions.
        2. Create reference ring = Dilated(Mask) - Mask.
        3. Exclude all OTHER damage masks to avoid contaminating undamaged reference baseline.
        4. Exclude outer image borders (border_margin_px).
        5. Check reference validity: if reference pixels < min_reference_pixels, mark as
           'reference_unavailable' and return None for scores (prevents fabricated metrics).
        6. Compute:
           - relative_deformation_score = median(Depth_mask) - median(Depth_reference)
           - surface_irregularity = variance(Depth_mask)
           - depth_std = std(Depth_mask)
           - severity_tier = compute_deformation_severity_tier(...)
        """
        if target_mask is None:
            target_mask = mask
        if target_mask is None:
            raise ValueError("target_mask or mask must be provided")

        if all_other_masks is None:
            all_other_masks = other_damage_masks if other_damage_masks is not None else []

        img_h, img_w = depth_map.shape
        mask_bool = (target_mask > 0).astype(bool)

        if not np.any(mask_bool):
            return DamageDeformation(
                reference_available=False,
                relative_deformation_score=None,
                max_relative_deformation=None,
                surface_irregularity=None,
                depth_std=None,
                deformation_type=None,
                severity_tier=None if damage_type == "unclassified" else "minor",
                deformation_status="reference_unavailable",
            )

        # Step 4: Minimum-size gate for unclassified fragments
        min_deform_gate = float(os.getenv("MIN_DEFORM_SCORING_AREA_PCT", "1.0"))
        if damage_type == "unclassified" and area_percentage < min_deform_gate:
            logger.info(
                f"[*] Skipping deformation scoring for unclassified fragment (area={area_percentage:.2f}% < {min_deform_gate}%). "
                "Marking as too small for reliable severity estimate."
            )
            return DamageDeformation(
                reference_available=False,
                relative_deformation_score=None,
                max_relative_deformation=None,
                surface_irregularity=None,
                depth_std=None,
                deformation_type=None,
                severity_tier=None,
                deformation_status="too_small_for_reliable_severity_estimate",
            )

        # 1. Compute dynamic dilation kernel size based on damage bounding box dimensions
        if bbox is not None:
            box_dim = max(bbox.width, bbox.height)
        else:
            ys, xs = np.where(mask_bool)
            y1, y2 = int(np.min(ys)), int(np.max(ys))
            x1, x2 = int(np.min(xs)), int(np.max(xs))
            box_dim = max(y2 - y1, x2 - x1)

        kernel_size = max(7, int(box_dim * dilation_ratio))
        # Ensure odd kernel size
        if kernel_size % 2 == 0:
            kernel_size += 1

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        dilated_mask = cv2.dilate(mask_bool.astype(np.uint8), kernel, iterations=1).astype(bool)

        # 2. Reference ring = Dilated - Original
        reference_ring = dilated_mask & (~mask_bool)

        # 3. Exclude all other damage masks in the same image
        for other in all_other_masks:
            other_bool = (other > 0).astype(bool)
            reference_ring = reference_ring & (~other_bool)

        # 4. Exclude boundary margins to prevent background/sky bleed
        reference_ring[:border_margin_px, :] = False
        reference_ring[-border_margin_px:, :] = False
        reference_ring[:, :border_margin_px] = False
        reference_ring[:, -border_margin_px:] = False

        ref_pixel_count = int(np.sum(reference_ring))
        dynamic_min_ref = max(min_reference_pixels, 50, int(box_dim * 1.5))

        # 5. Reference Availability Check (Step 4 requirement)
        if ref_pixel_count < dynamic_min_ref:
            coord_str = f"[{bbox.x1}, {bbox.y1}]" if bbox else "unknown"
            logger.info(
                f"[!] Reference unavailable for damage at {coord_str} "
                f"(valid ref pixels={ref_pixel_count} < {dynamic_min_ref}). Skipping score."
            )
            # Compute internal variance even if reference ring is unavailable
            mask_depth_vals = depth_map[mask_bool]
            std_internal = round(float(np.std(mask_depth_vals)), 4) if len(mask_depth_vals) > 0 else None
            var_internal = round(float(np.var(mask_depth_vals)), 5) if len(mask_depth_vals) > 0 else None

            return DamageDeformation(
                reference_available=False,
                relative_deformation_score=None,
                max_relative_deformation=None,
                surface_irregularity=var_internal,
                depth_std=std_internal,
                deformation_type="unknown",
                severity_tier=None if damage_type == "unclassified" else compute_deformation_severity_tier(None, std_internal, area_percentage),
                deformation_status="reference_unavailable",
            )

        # Sample depth values
        ref_depths = depth_map[reference_ring]
        ref_std = float(np.std(ref_depths))

        # Tighten reference validity: if reference ring variance is high, it spans an edge/sky/ground boundary
        if ref_std > 0.25:
            coord_str = f"[{bbox.x1}, {bbox.y1}]" if bbox else "unknown"
            logger.info(
                f"[!] Reference ring depth variance too high ({ref_std:.3f} > 0.25) at {coord_str}. "
                "Baseline spans depth boundary. Skipping score."
            )
            mask_depth_vals = depth_map[mask_bool]
            std_internal = round(float(np.std(mask_depth_vals)), 4) if len(mask_depth_vals) > 0 else None
            var_internal = round(float(np.var(mask_depth_vals)), 5) if len(mask_depth_vals) > 0 else None

            return DamageDeformation(
                reference_available=False,
                relative_deformation_score=None,
                max_relative_deformation=None,
                surface_irregularity=var_internal,
                depth_std=std_internal,
                deformation_type="unknown",
                severity_tier=None if damage_type == "unclassified" else "minor",
                deformation_status="reference_unavailable",
            )

        # 6. Sample depth values
        mask_depths = depth_map[mask_bool]
        ref_depths = depth_map[reference_ring]

        med_mask = float(np.median(mask_depths))
        med_ref = float(np.median(ref_depths))

        # Sign convention: med_mask - med_ref > 0 means recessed into vehicle (Dent)
        rel_deformation = round(med_mask - med_ref, 4)
        max_deformation = round(float(np.percentile(mask_depths, 95)) - med_ref, 4)
        depth_std = round(float(np.std(mask_depths)), 4)
        surface_irregularity = round(float(np.var(mask_depths)), 5)

        if rel_deformation >= 0.05:
            deformation_type = "recessed"
        elif rel_deformation <= -0.05:
            deformation_type = "protruding"
        else:
            deformation_type = "planar"

        tier = compute_deformation_severity_tier(
            relative_deformation=rel_deformation,
            depth_std=depth_std,
            area_percentage=area_percentage,
        )

        return DamageDeformation(
            reference_available=True,
            relative_deformation_score=rel_deformation,
            max_relative_deformation=max_deformation,
            surface_irregularity=surface_irregularity,
            depth_std=depth_std,
            deformation_type=deformation_type,
            severity_tier=tier,
            deformation_status="computed",
        )


# Global singleton instance
_depth_estimator_instance: Optional[DepthAnythingV2Estimator] = None


def get_depth_estimator() -> DepthAnythingV2Estimator:
    """Singleton getter for DepthAnythingV2Estimator."""
    global _depth_estimator_instance
    if _depth_estimator_instance is None:
        _depth_estimator_instance = DepthAnythingV2Estimator()
    return _depth_estimator_instance


def estimate_damage_depth(
    image_bytes: bytes,
    mask: Optional[np.ndarray] = None,
) -> DepthMapResult:
    """
    Standard interface function matching Phase 2 signature.
    """
    estimator = get_depth_estimator()
    _, _, result = estimator.estimate_scene_depth(image_bytes)
    return result
