"""
Phase 3 Benchmark & Verification Script: Depth Anything V2 Surface Deformation.
Tests real end-to-end damage detection, SAM2 segmentation, Depth Anything V2 estimation,
and per-mask deformation scores on sample images.
"""

import time
import os
from PIL import Image
import numpy as np

from backend.app.vision.damage_detector import get_detector
from backend.app.vision.vehicle_detector import get_vehicle_detector
from backend.app.pipeline.damage_segmentation import get_segmenter
from backend.app.pipeline.depth_estimation import get_depth_estimator


def benchmark_phase3():
    print("=" * 70)
    print("PHASE 3 VERIFICATION: DEPTH ANYTHING V2 MONOCULAR SURFACE DEFORMATION")
    print("=" * 70)

    t0 = time.perf_counter()
    detector = get_detector()
    vehicle_detector = get_vehicle_detector()
    segmenter = get_segmenter()
    depth_estimator = get_depth_estimator()
    print(f"[+] All models loaded in {time.perf_counter() - t0:.2f}s")
    print(f"[+] Depth Anything V2 Model: {depth_estimator.model_id} on {depth_estimator.device}")

    sample_paths = ["backend/static/samples/1.png", "backend/static/samples/2.png", "backend/static/samples/3.png"]

    for sample_path in sample_paths:
        if not os.path.exists(sample_path):
            continue

        print("\n" + "-" * 60)
        print(f"IMAGE: {sample_path}")
        print("-" * 60)

        pil_img = Image.open(sample_path).convert("RGB")
        w, h = pil_img.size
        print(f"Image Resolution: {w}x{h}")

        # 1. Vehicle ROI Crop
        t_crop = time.perf_counter()
        cropped_vehicle, roi_bbox, crop_coords = vehicle_detector.detect_vehicle_roi(pil_img)
        ms_crop = (time.perf_counter() - t_crop) * 1000

        # 2. YOLO Detection
        t_yolo = time.perf_counter()
        annotated_pil, detections = detector.predict(pil_img, confidence_threshold=0.15, iou_threshold=0.45)
        ms_yolo = (time.perf_counter() - t_yolo) * 1000
        print(f"Detections (conf>=0.15): {len(detections)} damages found")

        # 3. SAM2 Segmentation
        t_sam = time.perf_counter()
        box_segmented_pil, classified_masks, binary_masks = segmenter.segment_image_detections(
            image_input=pil_img,
            detections=detections,
            vehicle_roi_coords=crop_coords,
        )
        ms_sam = (time.perf_counter() - t_sam) * 1000

        # 4. Depth Anything V2
        t_depth = time.perf_counter()
        depth_map_norm, colormap_pil, depth_result = depth_estimator.estimate_scene_depth(pil_img)
        ms_depth = (time.perf_counter() - t_depth) * 1000

        print(f"Depth Scene Stats: min={depth_result.min_scene_depth}, max={depth_result.max_scene_depth}, mean={depth_result.mean_scene_depth}")
        print(f"Normalized Array: min={depth_map_norm.min():.4f}, max={depth_map_norm.max():.4f}, mean={depth_map_norm.mean():.4f}")

        # 5. Deformation per Detection
        for i, det in enumerate(detections):
            if i < len(binary_masks):
                mask = binary_masks[i]
                other_masks = [m for j, m in enumerate(binary_masks) if j != i]
                deform = depth_estimator.compute_instance_deformation(
                    depth_map=depth_map_norm,
                    mask=mask,
                    bbox=det.bbox,
                    area_percentage=det.segmentation.area_percentage if det.segmentation else 0.0,
                    damage_type=det.damage_type,
                    other_damage_masks=other_masks,
                )
                det.deformation = deform
                score_str = f"{deform.relative_deformation_score:+.4f}" if deform.relative_deformation_score is not None else "N/A"
                max_str = f"{deform.max_relative_deformation:+.4f}" if deform.max_relative_deformation is not None else "N/A"
                var_str = f"{deform.surface_irregularity:.4f}" if deform.surface_irregularity is not None else "N/A"
                print(
                    f"  #{i+1} [{det.damage_type.upper()}] (conf={det.confidence:.2f}): "
                    f"rel_deform={score_str} | max_deform={max_str} | "
                    f"roughness(σ²)={var_str} | type={deform.deformation_type} | "
                    f"tier={deform.severity_tier} | status={deform.deformation_status}"
                )

        print("\nLatency Profiling Breakdown:")
        print(f"  Vehicle ROI Crop : {ms_crop:6.1f} ms")
        print(f"  YOLO11m Detection: {ms_yolo:6.1f} ms")
        print(f"  SAM2 Segment     : {ms_sam:6.1f} ms")
        print(f"  Depth Anything V2: {ms_depth:6.1f} ms")
        print(f"  Total Image Time : {ms_crop + ms_yolo + ms_sam + ms_depth:6.1f} ms ({(ms_crop + ms_yolo + ms_sam + ms_depth)/1000:.2f} s)")


if __name__ == "__main__":
    benchmark_phase3()
