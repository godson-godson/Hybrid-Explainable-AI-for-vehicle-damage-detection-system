"""
SAM2 (Segment Anything Model 2) Damage Segmentation Module.
Generates pixel-accurate binary masks, surface area metrics, and contour polygons
for vehicle damages prompted by YOLO11m bounding boxes, and surfaces unclassified
fragmented damage via Automatic Mask Generation (AMG) on the vehicle ROI.
"""

import io
import logging
import os
import time
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from ..models.schemas import BoundingBox, DamageDetection, SegmentationMask
from .amg_config import AMGConfig, default_amg_config

logger = logging.getLogger("damage_segmentation")
logging.basicConfig(level=logging.INFO)

# Damage class color palette (RGB)
CLASS_COLOR_PALETTE: Dict[str, Tuple[int, int, int]] = {
    "dent": (255, 140, 0),             # Vivid Orange
    "scratch": (230, 40, 60),           # Crimson Red
    "crack": (0, 180, 216),             # Cyan / Teal
    "shattered_glass": (142, 68, 173),    # Purple
    "broken_lamp": (231, 76, 60),       # Bright Red
    "flat_tire": (243, 156, 18),        # Golden Amber
}
DEFAULT_COLOR: Tuple[int, int, int] = (52, 152, 219)        # Fallback Blue
UNCLASSIFIED_COLOR: Tuple[int, int, int] = (100, 116, 139)  # Neutral Slate Gray


def compute_edge_aware_padded_box(
    b: BoundingBox,
    img_w: int,
    img_h: int,
    all_boxes: List[BoundingBox],
    vehicle_roi_coords: Optional[Tuple[int, int, int, int]] = None,
    view_angle: Optional[str] = None,
    base_padding_pct: float = 0.10,
    edge_padding_pct: float = 0.15,
    edge_margin_ratio: float = 0.05,
) -> np.ndarray:
    """
    Apply edge-aware bounding box padding prior to SAM2 prompt execution.

    GEOMETRIC HEURISTIC & RATIONALE (Steps 3 & 4):
    YOLO models trained on vehicle damage tend to draw tight bounding boxes around the
    prominent center of a dent or crumpled surface (such as a bent hood leading edge).
    However, kinetic impact typically fractures or distorts adjacent mounting clips,
    subframe brackets, or lower bumper grilles immediately underneath.
    
    Rule:
    1. Base expansion: Expand box by base_padding_pct (10%) on each side.
    2. Edge & Adjacency Heuristic:
       - If the box's lower edge is within edge_margin_ratio (5% of image height) of the
         bottom of the vehicle frontal ROI or image, OR
       - If the box directly borders/overlaps another detected damage box (distance < 20px),
       the expansion is increased to the top of the range (edge_padding_pct = 15%).
       When near the lower boundary of a frontal panel, extra downward reach is permitted
       so SAM2 can segment the crumpled bumper/subframe directly underneath.
    3. Clamping: All coordinates are clamped within image boundaries [0, 0, img_w, img_h].
    """
    box_w = max(1.0, b.width)
    box_h = max(1.0, b.height)

    # Check proximity to bottom edge of vehicle ROI or image
    is_near_lower_edge = False
    if vehicle_roi_coords is not None:
        _, _, _, roi_y2 = vehicle_roi_coords
        if abs(roi_y2 - b.y2) <= (edge_margin_ratio * img_h):
            is_near_lower_edge = True
    else:
        if (img_h - b.y2) <= (edge_margin_ratio * img_h):
            is_near_lower_edge = True

    # Check adjacency: does this box sit within 20px of any other detected damage box?
    is_adjacent_to_other = False
    for other in all_boxes:
        if other is b:
            continue
        # Distance between bounding boxes
        dx = max(0.0, max(b.x1, other.x1) - min(b.x2, other.x2))
        dy = max(0.0, max(b.y1, other.y1) - min(b.y2, other.y2))
        if dx <= 20.0 and dy <= 20.0:
            is_adjacent_to_other = True
            break

    # Determine padding percentage
    if is_near_lower_edge or is_adjacent_to_other:
        pad_ratio_x = edge_padding_pct
        pad_ratio_y = edge_padding_pct
        # Extra downward reach if sitting at the lower frontal threshold
        extra_down = (box_h * 0.05) if is_near_lower_edge else 0.0
    else:
        pad_ratio_x = base_padding_pct
        pad_ratio_y = base_padding_pct
        extra_down = 0.0

    pad_x = box_w * pad_ratio_x
    pad_y = box_h * pad_ratio_y

    px1 = max(0.0, b.x1 - pad_x)
    py1 = max(0.0, b.y1 - pad_y)
    px2 = min(float(img_w), b.x2 + pad_x)
    py2 = min(float(img_h), b.y2 + pad_y + extra_down)

    return np.array([px1, py1, px2, py2], dtype=np.float32)


class SAM2DamageSegmenter:
    """
    Singleton wrapper for Meta's SAM2 (Segment Anything Model 2).
    Provides:
      1. Box-prompted segmentation with edge-aware padding for classified YOLO detections.
      2. Dense Automatic Mask Generation (AMG) on the cropped vehicle ROI for unclassified fragments.
      3. IoA/IoU redundancy reconciliation and combined visual rendering.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        config_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.device = self._resolve_device(device)
        self.checkpoint_path = self._resolve_checkpoint_path(checkpoint_path)
        self.config_path = self._resolve_config_path(config_path, self.checkpoint_path)

        logger.info(f"[*] Initializing SAM2 Predictor on device: {self.device}")
        logger.info(f"[*] SAM2 Checkpoint: {self.checkpoint_path}")
        logger.info(f"[*] SAM2 Config: {self.config_path}")

        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self.model = build_sam2(self.config_path, self.checkpoint_path, device=self.device)
        self.predictor = SAM2ImagePredictor(self.model)
        
        # Automatic Mask Generator lazy instance (shares self.model to avoid duplicate RAM/VRAM)
        self._amg_generator = None
        self._amg_params: Optional[Tuple[int, float, float]] = None

        logger.info("[+] SAM2 Damage Segmenter loaded and ready.")

    def _resolve_device(self, user_device: Optional[str]) -> str:
        """Resolve device preference with automatic GPU detection and CPU fallback warning."""
        if user_device:
            dev = user_device.strip().lower()
        else:
            dev = os.getenv("SAM2_DEVICE", "").strip().lower()

        if dev in ["cuda", "cuda:0"] and torch.cuda.is_available():
            return "cuda"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            logger.warning(
                "[WARNING] CUDA is not available. SAM2 is falling back to CPU inference. "
                "Inference on CPU will be slower than on an NVIDIA RTX GPU."
            )
            return "cpu"

    def _resolve_checkpoint_path(self, provided_path: Optional[str]) -> str:
        """Find the SAM2 checkpoint weights file."""
        if provided_path and os.path.exists(provided_path):
            return os.path.abspath(provided_path)

        env_path = os.getenv("SAM2_CHECKPOINT_PATH")
        if env_path and os.path.exists(env_path):
            return os.path.abspath(env_path)

        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))

        candidate_paths = [
            os.path.join(project_root, "backend", "weights", "sam2", "sam2.1_hiera_tiny.pt"),
            os.path.join(project_root, "backend", "weights", "sam2", "sam2.1_hiera_small.pt"),
            os.path.join(project_root, "backend", "weights", "sam2.1_hiera_tiny.pt"),
            os.path.join(os.getcwd(), "backend", "weights", "sam2", "sam2.1_hiera_tiny.pt"),
        ]

        for path in candidate_paths:
            if os.path.exists(path):
                return os.path.abspath(path)

        raise FileNotFoundError(
            f"Could not locate SAM2 checkpoint in: {candidate_paths}. "
            "Please set SAM2_CHECKPOINT_PATH environment variable."
        )

    def _resolve_config_path(self, user_config: Optional[str], checkpoint_path: str) -> str:
        """Resolve SAM2 model YAML configuration file."""
        if user_config:
            return user_config

        env_config = os.getenv("SAM2_MODEL_CONFIG")
        if env_config:
            return env_config

        ckpt_name = os.path.basename(checkpoint_path).lower()
        if "hiera_s" in ckpt_name:
            return "configs/sam2.1/sam2.1_hiera_s.yaml"
        elif "hiera_b+" in ckpt_name or "base_plus" in ckpt_name:
            return "configs/sam2.1/sam2.1_hiera_b+.yaml"
        elif "hiera_l" in ckpt_name or "large" in ckpt_name:
            return "configs/sam2.1/sam2.1_hiera_l.yaml"
        else:
            return "configs/sam2.1/sam2.1_hiera_t.yaml"

    def _get_or_create_amg_generator(
        self,
        points_per_side: Optional[int] = None,
        pred_iou_thresh: Optional[float] = None,
        stability_score_thresh: Optional[float] = None,
    ):
        """
        Instantiate or return cached SAM2AutomaticMaskGenerator.
        Shares self.model (the exact same weights as SAM2Predictor) to prevent duplicate
        model weight instantiation and VRAM/RAM collision. Re-instantiates if parameters change.
        """
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

        cfg_pts = points_per_side if points_per_side is not None else default_amg_config.points_per_side
        cfg_iou = pred_iou_thresh if pred_iou_thresh is not None else default_amg_config.pred_iou_thresh
        cfg_stab = stability_score_thresh if stability_score_thresh is not None else default_amg_config.stability_score_thresh
        target_params = (cfg_pts, cfg_iou, cfg_stab)

        if self._amg_generator is None or self._amg_params != target_params:
            logger.info(
                f"[*] Initializing shared SAM2AutomaticMaskGenerator "
                f"(points_per_side={cfg_pts}, pred_iou_thresh={cfg_iou}, stability_score_thresh={cfg_stab})"
            )
            self._amg_generator = SAM2AutomaticMaskGenerator(
                model=self.model,
                points_per_side=cfg_pts,
                pred_iou_thresh=cfg_iou,
                stability_score_thresh=cfg_stab,
                crop_n_layers=0,
                min_mask_region_area=50,
            )
            self._amg_params = target_params
        return self._amg_generator

    def segment_image_detections(
        self,
        image_input: Union[bytes, Image.Image, np.ndarray, str],
        detections: List[DamageDetection],
        vehicle_roi_coords: Optional[Tuple[int, int, int, int]] = None,
        view_angle: Optional[str] = None,
    ) -> Tuple[Image.Image, List[SegmentationMask], List[np.ndarray]]:
        """
        Execute padded box-prompted SAM2 segmentation for classified YOLO detections.

        Returns:
            Tuple of:
                - Visual overlay image with colored masks.
                - List of SegmentationMask objects.
                - List of raw binary masks (np.ndarray of shape [H, W]).
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
        total_image_pixels = max(1, img_w * img_h)

        if not detections:
            return pil_img.copy(), [], []

        # Set image in SAM2 image predictor
        img_np = np.array(pil_img)
        self.predictor.set_image(img_np)

        masks_result: List[SegmentationMask] = []
        raw_binary_masks: List[np.ndarray] = []
        all_bboxes = [d.bbox for d in detections]

        for det in detections:
            b = det.bbox
            # Step 3 & 4: Compute edge-aware padded box prompt
            box_prompt = compute_edge_aware_padded_box(
                b=b,
                img_w=img_w,
                img_h=img_h,
                all_boxes=all_bboxes,
                vehicle_roi_coords=vehicle_roi_coords,
                view_angle=view_angle,
                base_padding_pct=0.10,
                edge_padding_pct=0.15,
                edge_margin_ratio=0.05,
            )

            with torch.inference_mode():
                masks, scores, _ = self.predictor.predict(
                    box=box_prompt,
                    multimask_output=False,
                )

            # SAM2 returns masks of shape (1, H, W)
            raw_mask = masks[0]
            binary_mask = (raw_mask > 0.0).astype(bool)
            raw_binary_masks.append(binary_mask)

            area_pixels = int(np.sum(binary_mask))
            area_pct = round((area_pixels / total_image_pixels) * 100.0, 3)
            surface_ratio = round(area_pixels / total_image_pixels, 5)
            mask_score = float(scores[0]) if len(scores) > 0 else None

            # Extract contour polygon points
            mask_uint8 = (binary_mask.astype(np.uint8) * 255)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            polygon_points: List[List[float]] = []
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                epsilon = max(1.0, 0.003 * cv2.arcLength(largest_contour, True))
                approx = cv2.approxPolyDP(largest_contour, epsilon, True)
                if len(approx) >= 3:
                    polygon_points = approx.reshape(-1, 2).astype(float).tolist()
                else:
                    polygon_points = largest_contour.reshape(-1, 2).astype(float).tolist()

            seg_mask = SegmentationMask(
                damage_type=det.damage_type,
                polygon_points=polygon_points,
                area_pixels=area_pixels,
                area_percentage=area_pct,
                relative_surface_ratio=surface_ratio,
                mask_confidence=mask_score,
                part_name=None,
            )

            det.segmentation = seg_mask
            masks_result.append(seg_mask)

        # Generate visual overlay image with colored semi-transparent mask fills
        annotated_segmented_img = self._render_mask_overlay(
            base_img=pil_img,
            detections=detections,
            binary_masks=raw_binary_masks,
        )

        return annotated_segmented_img, masks_result, raw_binary_masks

    def segment_automatic_roi_masks(
        self,
        cropped_vehicle_img: Image.Image,
        crop_coords: Tuple[int, int, int, int],
        full_img_size: Tuple[int, int],
        classified_detections: List[DamageDetection],
        classified_binary_masks: List[np.ndarray],
        points_per_side: Optional[int] = None,
        pred_iou_thresh: Optional[float] = None,
        stability_score_thresh: Optional[float] = None,
        min_crop_area_pct: Optional[float] = None,
        max_crop_area_pct: Optional[float] = None,
        dedup_iou_thresh: Optional[float] = None,
        ioa_redundancy_thresh: Optional[float] = None,
        iou_redundancy_thresh: Optional[float] = None,
        min_area_ratio: Optional[float] = None,
        max_area_ratio: Optional[float] = None,
    ) -> Tuple[List[DamageDetection], List[np.ndarray]]:
        """
        Run SAM2 Automatic Mask Generation (AMG) on the cropped vehicle ROI.

        MULTI-STAGE FILTERING PIPELINE:
        Step 1: Tightened AMG Generation (points_per_side, pred_iou_thresh, stability_score_thresh).
        Step 2: Immediate minimum-area floor (% of vehicle crop area, default 1.0%).
        Step 3: Deduplicate overlapping AMG proposals (IoU > 0.80, keep higher stability/larger).
        Step 4: Full-image coordinate mapping & reconciliation against classified detections.
        Step 5: Area-descending sorting for surveyor prioritization.
        """
        img_w, img_h = full_img_size
        crop_x1, crop_y1, crop_x2, crop_y2 = crop_coords
        crop_w = max(1, crop_x2 - crop_x1)
        crop_h = max(1, crop_y2 - crop_y1)
        crop_area = crop_w * crop_h
        total_image_pixels = max(1, img_w * img_h)

        # Resolve named configuration parameters
        pts = points_per_side if points_per_side is not None else default_amg_config.points_per_side
        p_iou = pred_iou_thresh if pred_iou_thresh is not None else default_amg_config.pred_iou_thresh
        s_score = stability_score_thresh if stability_score_thresh is not None else default_amg_config.stability_score_thresh

        # Area floor resolution (backward-compatible with min_area_ratio if supplied)
        if min_crop_area_pct is not None:
            min_floor_pct = min_crop_area_pct
        elif min_area_ratio is not None:
            min_floor_pct = min_area_ratio * 100.0
        else:
            min_floor_pct = default_amg_config.min_crop_area_pct

        if max_crop_area_pct is not None:
            max_floor_pct = max_crop_area_pct
        elif max_area_ratio is not None:
            max_floor_pct = max_area_ratio * 100.0
        else:
            max_floor_pct = default_amg_config.max_crop_area_pct

        d_iou_th = dedup_iou_thresh if dedup_iou_thresh is not None else default_amg_config.dedup_iou_thresh
        ioa_red_th = ioa_redundancy_thresh if ioa_redundancy_thresh is not None else default_amg_config.ioa_redundancy_thresh
        iou_red_th = iou_redundancy_thresh if iou_redundancy_thresh is not None else default_amg_config.iou_redundancy_thresh

        # Step 1: Tightened AMG generation
        amg = self._get_or_create_amg_generator(
            points_per_side=pts,
            pred_iou_thresh=p_iou,
            stability_score_thresh=s_score,
        )

        crop_np = np.array(cropped_vehicle_img)
        with torch.inference_mode():
            raw_amg_results = amg.generate(crop_np)

        raw_count = len(raw_amg_results)
        logger.info(
            f"[*] Step 1 (AMG Generation): Generated {raw_count} raw candidate masks "
            f"(points_per_side={pts}, pred_iou={p_iou}, stability={s_score}) before filtering."
        )

        # Step 2: Immediate minimum-area floor applied to candidate masks
        # Computed as a percentage of vehicle ROI crop area, NOT full image or raw pixel count
        area_floor_survivors = []
        for item in raw_amg_results:
            mask_area_pixels = int(item["area"])
            crop_pct = (mask_area_pixels / crop_area) * 100.0
            if crop_pct < min_floor_pct or crop_pct > max_floor_pct:
                continue
            item["crop_area_pct"] = crop_pct
            area_floor_survivors.append(item)

        logger.info(
            f"[*] Step 2 (Area Floor): {len(area_floor_survivors)} / {raw_count} masks survived "
            f"area floor [{min_floor_pct:.2f}% - {max_floor_pct:.2f}% of vehicle crop]."
        )

        # Step 3: Deduplicate overlapping AMG masks
        # Dense point grids produce multiple overlapping proposals for the same region.
        # Sort candidates by stability_score descending (with area as tie-breaker)
        area_floor_survivors.sort(
            key=lambda x: (float(x.get("stability_score", 0.0)), int(x.get("area", 0))),
            reverse=True,
        )

        deduped_candidates = []
        for cand in area_floor_survivors:
            cand_mask = cand["segmentation"]
            cand_area = cand["area"]
            bx1, by1, bw1, bh1 = cand["bbox"]
            is_dup = False

            for kept in deduped_candidates:
                kx1, ky1, kw1, kh1 = kept["bbox"]
                # Fast bounding-box disjoint check
                if bx1 > kx1 + kw1 or kx1 > bx1 + bw1 or by1 > ky1 + kh1 or ky1 > by1 + bh1:
                    continue
                intersection = int(np.sum(cand_mask & kept["segmentation"]))
                if intersection == 0:
                    continue
                union = cand_area + kept["area"] - intersection
                iou = intersection / union if union > 0 else 0.0
                if iou > d_iou_th:
                    is_dup = True
                    break

            if not is_dup:
                deduped_candidates.append(cand)

        logger.info(
            f"[*] Step 3 (Overlap Dedup): {len(deduped_candidates)} / {len(area_floor_survivors)} masks retained "
            f"after IoU dedup (threshold > {d_iou_th:.2f})."
        )

        # Step 4: Full-image coordinate translation & classified reconciliation
        surviving_detections: List[DamageDetection] = []
        surviving_binary_masks: List[np.ndarray] = []

        for item in deduped_candidates:
            crop_mask = item["segmentation"]
            mask_area_pixels = int(item["area"])

            # Map crop mask back to full image coordinate space
            full_binary_mask = np.zeros((img_h, img_w), dtype=bool)
            full_binary_mask[crop_y1:crop_y2, crop_x1:crop_x2] = crop_mask

            # Bounding box translation
            bx, by, bw, bh = item["bbox"]
            fx1 = max(0.0, float(crop_x1 + bx))
            fy1 = max(0.0, float(crop_y1 + by))
            fx2 = min(float(img_w), float(crop_x1 + bx + bw))
            fy2 = min(float(img_h), float(crop_y1 + by + bh))

            # Redundancy Reconciliation against classified YOLO damages
            is_redundant = False
            for c_det, c_mask in zip(classified_detections, classified_binary_masks):
                intersection = int(np.sum(full_binary_mask & c_mask))
                if intersection == 0:
                    continue

                c_area = int(np.sum(c_mask))
                amg_area = mask_area_pixels
                union = amg_area + c_area - intersection

                ioa_amg = intersection / amg_area if amg_area > 0 else 0.0
                ioa_classified = intersection / c_area if c_area > 0 else 0.0
                iou = intersection / union if union > 0 else 0.0

                if (
                    ioa_amg > ioa_red_th
                    or ioa_classified > ioa_red_th
                    or iou > iou_red_th
                ):
                    is_redundant = True
                    break

            if is_redundant:
                continue

            # Extract contour polygon points
            mask_uint8 = (full_binary_mask.astype(np.uint8) * 255)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            polygon_points: List[List[float]] = []
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                epsilon = max(1.0, 0.003 * cv2.arcLength(largest_contour, True))
                approx = cv2.approxPolyDP(largest_contour, epsilon, True)
                if len(approx) >= 3:
                    polygon_points = approx.reshape(-1, 2).astype(float).tolist()
                else:
                    polygon_points = largest_contour.reshape(-1, 2).astype(float).tolist()

            area_pct = round((mask_area_pixels / total_image_pixels) * 100.0, 3)
            surface_ratio = round(mask_area_pixels / total_image_pixels, 5)

            seg_mask = SegmentationMask(
                damage_type="unclassified",
                polygon_points=polygon_points,
                area_pixels=mask_area_pixels,
                area_percentage=area_pct,
                relative_surface_ratio=surface_ratio,
                mask_confidence=round(float(item.get("predicted_iou", 0.85)), 3),
                part_name="fragment_candidate",
            )

            bbox = BoundingBox(
                x1=round(fx1, 2),
                y1=round(fy1, 2),
                x2=round(fx2, 2),
                y2=round(fy2, 2),
                width=round(fx2 - fx1, 2),
                height=round(fy2 - fy1, 2),
                norm_x1=round(fx1 / img_w, 4),
                norm_y1=round(fy1 / img_h, 4),
                norm_x2=round(fx2 / img_w, 4),
                norm_y2=round(fy2 / img_h, 4),
            )

            unclass_det = DamageDetection(
                class_id=-1,
                damage_type="unclassified",
                confidence=round(float(item.get("predicted_iou", 0.75)), 3),
                confidence_tier="low",
                classified=False,
                label="unclassified — needs review",
                bbox=bbox,
                view_angle="roi_fragment",
                segmentation=seg_mask,
            )

            surviving_detections.append(unclass_det)
            surviving_binary_masks.append(full_binary_mask)

        # Sort surviving unclassified detections by area descending
        if surviving_detections:
            paired = list(zip(surviving_detections, surviving_binary_masks))
            paired.sort(
                key=lambda p: (p[0].segmentation.area_percentage if p[0].segmentation else 0.0),
                reverse=True,
            )
            surviving_detections = [p[0] for p in paired]
            surviving_binary_masks = [p[1] for p in paired]

        logger.info(
            f"[*] Post-reconciliation surviving unclassified fragments: {len(surviving_detections)} "
            f"(sorted by area descending)."
        )

        return surviving_detections, surviving_binary_masks

    def render_combined_overlay(
        self,
        base_img: Image.Image,
        classified_detections: List[DamageDetection],
        classified_masks: List[np.ndarray],
        unclassified_detections: List[DamageDetection],
        unclassified_masks: List[np.ndarray],
    ) -> Image.Image:
        """
        Render both classified damage masks and unclassified AMG candidate regions.
        Classified masks are rendered with class colors; unclassified fragments are rendered
        with a distinctive neutral slate gray overlay and 'Unclassified — Needs Review' badges.
        """
        all_detections = classified_detections + unclassified_detections
        all_masks = classified_masks + unclassified_masks
        return self._render_mask_overlay(
            base_img=base_img,
            detections=all_detections,
            binary_masks=all_masks,
        )

    def _render_mask_overlay(
        self,
        base_img: Image.Image,
        detections: List[DamageDetection],
        binary_masks: List[np.ndarray],
    ) -> Image.Image:
        """Render semi-transparent damage masks, crisp contour strokes, and badges."""
        img_w, img_h = base_img.size
        overlay_np = np.array(base_img.copy(), dtype=np.float32)

        for det, b_mask in zip(detections, binary_masks):
            if not det.classified:
                color = UNCLASSIFIED_COLOR
                alpha = 0.35  # Subtle opacity for unclassified fragments
            else:
                color = CLASS_COLOR_PALETTE.get(det.damage_type, DEFAULT_COLOR)
                alpha = 0.45

            color_rgb = np.array(color, dtype=np.float32)
            mask_region = b_mask == True
            overlay_np[mask_region] = (
                (1.0 - alpha) * overlay_np[mask_region] + alpha * color_rgb
            )

        blended_img = Image.fromarray(np.clip(overlay_np, 0, 255).astype(np.uint8))
        draw = ImageDraw.Draw(blended_img)

        font_size = max(12, int(min(img_w, img_h) / 48))
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
        except IOError:
            try:
                font = ImageFont.truetype("Arial.ttf", font_size)
            except IOError:
                font = ImageFont.load_default()

        line_width = max(2, int(min(img_w, img_h) / 300))

        for det, b_mask in zip(detections, binary_masks):
            if not det.classified:
                color = UNCLASSIFIED_COLOR
            else:
                color = CLASS_COLOR_PALETTE.get(det.damage_type, DEFAULT_COLOR)

            b = det.bbox

            # Draw outer contour line with OpenCV
            mask_uint8 = (b_mask.astype(np.uint8) * 255)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            contour_img_np = np.array(blended_img)
            cv2.drawContours(contour_img_np, contours, -1, color, line_width)
            blended_img = Image.fromarray(contour_img_np)
            draw = ImageDraw.Draw(blended_img)

            # Draw bounding box context (dashed/thin)
            box_width = 1 if (not det.classified or det.confidence_tier == "low") else 2
            draw.rectangle(
                [b.x1, b.y1, b.x2, b.y2],
                outline=(*color, 140),
                width=box_width,
            )

            # Label badge
            seg = det.segmentation
            area_str = f" | {seg.area_percentage:.2f}%" if seg else ""

            if not det.classified:
                label_text = f"Unclassified Fragment{area_str}"
            else:
                tier_str = " [LOW]" if det.confidence_tier == "low" else ""
                label_text = f"{det.damage_type.replace('_', ' ').title()} {det.confidence * 100:.0f}%{tier_str}{area_str}"

            try:
                bbox_text = draw.textbbox((0, 0), label_text, font=font)
                text_w = bbox_text[2] - bbox_text[0]
                text_h = bbox_text[3] - bbox_text[1]
            except AttributeError:
                text_w = len(label_text) * 8
                text_h = 16

            pad_x, pad_y = 6, 4
            badge_y1 = max(0.0, b.y1 - text_h - pad_y * 2)
            badge_y2 = badge_y1 + text_h + pad_y * 2
            badge_x1 = b.x1
            badge_x2 = min(float(img_w), b.x1 + text_w + pad_x * 2)

            draw.rectangle([badge_x1, badge_y1, badge_x2, badge_y2], fill=color)
            draw.text(
                (badge_x1 + pad_x, badge_y1 + pad_y),
                label_text,
                fill=(255, 255, 255),
                font=font,
            )

        return blended_img


# Module-level singleton instance
_segmenter_instance: Optional[SAM2DamageSegmenter] = None


def get_segmenter() -> SAM2DamageSegmenter:
    """Singleton getter for the SAM2DamageSegmenter."""
    global _segmenter_instance
    if _segmenter_instance is None:
        _segmenter_instance = SAM2DamageSegmenter()
    return _segmenter_instance


def segment_damaged_parts(
    image_bytes: bytes,
    detections: List[DamageDetection],
) -> List[SegmentationMask]:
    """
    Standard interface function matching Phase 1 signature.
    Generates pixel-level segmentation masks for detected damage regions.
    """
    segmenter = get_segmenter()
    _, masks, _ = segmenter.segment_image_detections(image_bytes, detections)
    return masks
