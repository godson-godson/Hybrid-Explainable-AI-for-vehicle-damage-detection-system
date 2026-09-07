"""
YOLO11m Damage Detector Wrapper.
Loads the trained YOLO11m weights and executes inference on vehicle damage photographs.
Generates structured damage detections and annotated visualizations with bounding boxes.
"""

import io
import os
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from ..models.schemas import BoundingBox, DamageDetection


# Color palette for damage classes (RGB format for PIL / Drawing)
CLASS_COLOR_PALETTE: Dict[str, Tuple[int, int, int]] = {
    "dent": (255, 140, 0),          # Vivid Orange
    "scratch": (230, 40, 60),        # Crimson Red
    "crack": (0, 180, 216),          # Cyan / Teal
    "shattered_glass": (142, 68, 173), # Purple
    "broken_lamp": (231, 76, 60),    # Bright Red
    "flat_tire": (243, 156, 18),     # Golden Amber
}

DEFAULT_COLOR: Tuple[int, int, int] = (52, 152, 219)  # Blue fallback


class DamageDetector:
    """Wrapper around YOLO11m vehicle damage detection model."""

    def __init__(self, weights_path: Optional[str] = None):
        self.weights_path = self._resolve_weights_path(weights_path)
        print(f"[*] Loading YOLO11m Damage Detector from: {self.weights_path}")
        self.model = YOLO(self.weights_path)
        self.classes: Dict[int, str] = {int(k): str(v) for k, v in self.model.names.items()}
        print(f"[+] Model loaded successfully with {len(self.classes)} classes: {self.classes}")

    def _resolve_weights_path(self, provided_path: Optional[str]) -> str:
        """Locate weights file from provided argument, env var, or candidate locations."""
        if provided_path and os.path.exists(provided_path):
            return os.path.abspath(provided_path)

        env_path = os.getenv("YOLO_WEIGHTS_PATH")
        if env_path and os.path.exists(env_path):
            return os.path.abspath(env_path)

        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_dir)))

        candidate_paths = [
            os.path.join(project_root, "YOLO11m-Car-Damage-Detector-main", "trained.pt"),
            os.path.join(project_root, "backend", "weights", "trained.pt"),
            os.path.join(os.getcwd(), "YOLO11m-Car-Damage-Detector-main", "trained.pt"),
            "YOLO11m-Car-Damage-Detector-main/trained.pt",
        ]

        for path in candidate_paths:
            if os.path.exists(path):
                return os.path.abspath(path)

        raise FileNotFoundError(
            f"Could not locate 'trained.pt' weights file in candidates: {candidate_paths}. "
            "Set YOLO_WEIGHTS_PATH environment variable."
        )

    def predict(
        self,
        image_input: Union[bytes, Image.Image, np.ndarray, str],
        confidence_threshold: float = 0.15,
        iou_threshold: float = 0.45,
        view_angle: Optional[str] = None,
    ) -> Tuple[Image.Image, List[DamageDetection]]:
        """
        Execute YOLO11m inference on a single vehicle image.

        NOTE ON THRESHOLDS:
        The default confidence threshold is set to 0.15 (down from 0.25) and IoU to 0.45.
        Lowering the confidence threshold intentionally trades precision for recall.
        This design decision is critical because this is an assistive decision-support
        tool reviewed by a certified insurance surveyor, NOT an autonomous approval system.
        In claim assessment, false negatives (missed damages like cracked brackets, dented
        subframes, or hairline fractures) are far more costly to adjust later than false
        positives, which a human surveyor can quickly scrutinize or dismiss in the UI.

        Args:
            image_input: Binary bytes, PIL Image, OpenCV ndarray, or file path.
            confidence_threshold: Minimum confidence score to accept detection (default 0.15).
            iou_threshold: NMS IoU threshold for overlapping bounding boxes (default 0.45).
            view_angle: Optional view perspective (front, rear, left, right, etc.).

        Returns:
            Tuple containing:
                - Annotated PIL Image with custom-styled damage bounding boxes.
                - List of structured DamageDetection objects with confidence tiers.
        """
        # Convert input to PIL Image and NumPy RGB array
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

        img_width, img_height = pil_img.size

        # Run inference in eval/inference_mode to prevent gradient/VRAM retention
        import torch
        with torch.inference_mode():
            results = self.model.predict(
                source=pil_img,
                conf=confidence_threshold,
                iou=iou_threshold,
                verbose=False
            )

        detections: List[DamageDetection] = []
        result = results[0]

        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                cls_name = self.classes.get(cls_id, f"damage_{cls_id}")
                conf = float(box.conf[0].item())
                xyxy = box.xyxy[0].tolist()
                x1, y1, x2, y2 = xyxy[0], xyxy[1], xyxy[2], xyxy[3]

                # Ensure non-negative and bounded coordinates
                x1 = max(0.0, min(float(x1), float(img_width)))
                y1 = max(0.0, min(float(y1), float(img_height)))
                x2 = max(0.0, min(float(x2), float(img_width)))
                y2 = max(0.0, min(float(y2), float(img_height)))

                w = max(0.0, x2 - x1)
                h = max(0.0, y2 - y1)

                bbox = BoundingBox(
                    x1=round(x1, 2),
                    y1=round(y1, 2),
                    x2=round(x2, 2),
                    y2=round(y2, 2),
                    width=round(w, 2),
                    height=round(h, 2),
                    norm_x1=round(x1 / img_width, 4) if img_width else 0.0,
                    norm_y1=round(y1 / img_height, 4) if img_height else 0.0,
                    norm_x2=round(x2 / img_width, 4) if img_width else 0.0,
                    norm_y2=round(y2 / img_height, 4) if img_height else 0.0,
                )

                # Confidence tier: 'high' (>= 0.50) vs 'low' (< 0.50)
                tier = "high" if conf >= 0.50 else "low"
                tier_label = f"{cls_name.replace('_', ' ').title()}"

                detections.append(
                    DamageDetection(
                        class_id=cls_id,
                        damage_type=cls_name,
                        confidence=round(conf, 4),
                        confidence_tier=tier,
                        classified=True,
                        label=tier_label,
                        bbox=bbox,
                        view_angle=view_angle,
                    )
                )

        # Generate annotated visualization
        annotated_img = self._render_annotations(pil_img, detections)
        return annotated_img, detections

    def _render_annotations(
        self,
        base_img: Image.Image,
        detections: List[DamageDetection]
    ) -> Image.Image:
        """Render high-contrast, professional bounding boxes and labels with tier distinctions."""
        annotated = base_img.copy()
        draw = ImageDraw.Draw(annotated)
        img_w, img_h = base_img.size

        # Dynamic line thickness based on image size
        line_width = max(3, int(min(img_w, img_h) / 250))
        font_size = max(14, int(min(img_w, img_h) / 40))

        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
        except IOError:
            try:
                font = ImageFont.truetype("Arial.ttf", font_size)
            except IOError:
                font = ImageFont.load_default()

        for det in detections:
            color = CLASS_COLOR_PALETTE.get(det.damage_type, DEFAULT_COLOR)
            b = det.bbox
            is_low_conf = (det.confidence_tier == "low")

            # Draw outer rectangle - for low confidence, use thinner border
            box_width = max(1, line_width - 1) if is_low_conf else line_width
            for i in range(box_width):
                draw.rectangle(
                    [b.x1 - i, b.y1 - i, b.x2 + i, b.y2 + i],
                    outline=color
                )

            # Label text with tier indicator
            tier_badge = " [LOW]" if is_low_conf else ""
            label_text = f"{det.damage_type.replace('_', ' ').title()} {det.confidence * 100:.1f}%{tier_badge}"

            # Compute label background bounding box
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

            # Draw badge background
            draw.rectangle([badge_x1, badge_y1, badge_x2, badge_y2], fill=color)

            # Draw white label text
            draw.text(
                (badge_x1 + pad_x, badge_y1 + pad_y),
                label_text,
                fill=(255, 255, 255),
                font=font
            )

        return annotated


# Global detector instance singleton
_detector_instance: Optional[DamageDetector] = None


def get_detector() -> DamageDetector:
    """Singleton getter for the YOLO11m DamageDetector."""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = DamageDetector()
    return _detector_instance
