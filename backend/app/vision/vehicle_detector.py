"""
General Vehicle Detector Module.
Uses a lightweight COCO-pretrained YOLO model (yolo11n.pt) to locate and isolate the vehicle
bounding box in the frame. Crops the vehicle ROI with a configurable margin before running
dense SAM2 Automatic Mask Generation (AMG), significantly reducing background noise and compute.
"""

import os
from typing import Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO

from ..models.schemas import BoundingBox

# COCO classes representing vehicles:
# 2: car, 3: motorcycle, 5: bus, 7: truck
VEHICLE_CLASS_IDS = {2, 3, 5, 7}


class VehicleDetector:
    """Detects overall vehicle body region in an image using COCO-pretrained YOLO11n."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = self._resolve_model_path(model_path)
        print(f"[*] Initializing VehicleDetector from: {self.model_path}")
        self.model = YOLO(self.model_path)
        # Ensure model is in eval mode to avoid training overhead/gradient tracking
        if hasattr(self.model, "model") and hasattr(self.model.model, "eval"):
            self.model.model.eval()
        print("[+] VehicleDetector loaded successfully.")

    def _resolve_model_path(self, provided_path: Optional[str]) -> str:
        """Locate yolo11n.pt checkpoint or download fallback."""
        if provided_path and os.path.exists(provided_path):
            return os.path.abspath(provided_path)

        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        backend_dir = os.path.dirname(os.path.dirname(current_file_dir))
        project_root = os.path.dirname(backend_dir)

        candidates = [
            os.path.join(project_root, "yolo11n.pt"),
            os.path.join(backend_dir, "weights", "yolo11n.pt"),
            os.path.join(os.getcwd(), "yolo11n.pt"),
            "yolo11n.pt",
        ]

        for cand in candidates:
            if os.path.exists(cand):
                return os.path.abspath(cand)

        # Fallback to model name which Ultralytics will auto-fetch if needed
        return "yolo11n.pt"

    def detect_vehicle_roi(
        self,
        image_input: Union[Image.Image, np.ndarray, str],
        margin_pct: Optional[float] = None,
        min_confidence: float = 0.25,
    ) -> Tuple[Image.Image, Optional[BoundingBox], Tuple[int, int, int, int]]:
        """
        Locate the primary vehicle in the image and return a cropped image and its coordinates.

        Args:
            image_input: PIL Image or path to image.
            margin_pct: Expansion margin around vehicle boundary (default 2%, tight crop).
            min_confidence: Confidence threshold to consider a vehicle detection.

        Returns:
            Tuple containing:
                - Cropped PIL Image around vehicle (or original if no vehicle detected).
                - BoundingBox of detected vehicle ROI in full image coordinates.
                - Tuple of integer coordinates: (crop_x1, crop_y1, crop_x2, crop_y2).
        """
        if isinstance(image_input, str):
            pil_img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            pil_img = Image.fromarray(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        img_w, img_h = pil_img.size

        # Run inference under torch.inference_mode to prevent gradient/VRAM retention
        with torch.inference_mode():
            results = self.model.predict(
                source=pil_img,
                conf=min_confidence,
                classes=list(VEHICLE_CLASS_IDS),
                verbose=False,
            )

        best_box = None
        max_area = 0.0

        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                cls_id = int(box.cls[0].item())
                if cls_id in VEHICLE_CLASS_IDS:
                    xyxy = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = xyxy[0], xyxy[1], xyxy[2], xyxy[3]
                    area = (x2 - x1) * (y2 - y1)
                    if area > max_area:
                        max_area = area
                        best_box = (x1, y1, x2, y2)

        # Fallback: if no vehicle was detected (e.g. extreme close-up of a dented door/fender),
        # treat the entire image as the ROI.
        if best_box is None:
            crop_coords = (0, 0, img_w, img_h)
            roi_bbox = BoundingBox(
                x1=0.0,
                y1=0.0,
                x2=float(img_w),
                y2=float(img_h),
                width=float(img_w),
                height=float(img_h),
                norm_x1=0.0,
                norm_y1=0.0,
                norm_x2=1.0,
                norm_y2=1.0,
            )
            return pil_img, roi_bbox, crop_coords

        # Expand box by margin_pct (tight margin 0.02 avoids background artifacts)
        if margin_pct is None:
            margin_pct = float(os.getenv("VEHICLE_CROP_MARGIN_PCT", "0.02"))

        bx1, by1, bx2, by2 = best_box
        box_w = bx2 - bx1
        box_h = by2 - by1

        pad_x = box_w * margin_pct
        pad_y = box_h * margin_pct

        crop_x1 = max(0, int(bx1 - pad_x))
        crop_y1 = max(0, int(by1 - pad_y))
        crop_x2 = min(img_w, int(bx2 + pad_x))
        crop_y2 = min(img_h, int(by2 + pad_y))

        crop_w = crop_x2 - crop_x1
        crop_h = crop_y2 - crop_y1

        cropped_img = pil_img.crop((crop_x1, crop_y1, crop_x2, crop_y2))

        roi_bbox = BoundingBox(
            x1=float(crop_x1),
            y1=float(crop_y1),
            x2=float(crop_x2),
            y2=float(crop_y2),
            width=float(crop_w),
            height=float(crop_h),
            norm_x1=round(crop_x1 / img_w, 4),
            norm_y1=round(crop_y1 / img_h, 4),
            norm_x2=round(crop_x2 / img_w, 4),
            norm_y2=round(crop_y2 / img_h, 4),
        )

        return cropped_img, roi_bbox, (crop_x1, crop_y1, crop_x2, crop_y2)


# Global singleton instance
_vehicle_detector_instance: Optional[VehicleDetector] = None


def get_vehicle_detector() -> VehicleDetector:
    """Singleton getter for the VehicleDetector."""
    global _vehicle_detector_instance
    if _vehicle_detector_instance is None:
        _vehicle_detector_instance = VehicleDetector()
    return _vehicle_detector_instance
