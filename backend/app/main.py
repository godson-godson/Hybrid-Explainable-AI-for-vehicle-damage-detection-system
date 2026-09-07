"""
Main FastAPI Application Entrypoint.
Hybrid Explainable AI Framework for Intelligent Vehicle Damage Assessment and Insurance Claim Assistance.
"""

import os
import sys
from pathlib import Path

# Ensure package context and sys.path for direct script execution or worker spawn
if not __package__:
    project_root = str(Path(__file__).resolve().parent.parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    __package__ = "backend.app"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .api.routes_inspection import router as inspection_router
from .vision.damage_detector import get_detector
from .vision.vehicle_detector import get_vehicle_detector
from .pipeline.damage_segmentation import get_segmenter
from .pipeline.depth_estimation import get_depth_estimator
from .pipeline.knowledge_graph import get_knowledge_graph
from .pipeline.groq_client import get_groq_client

# Determine base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # backend/
PROJECT_ROOT = os.path.dirname(BASE_DIR)                              # project root
STATIC_DIR = os.path.join(BASE_DIR, "static")
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")

# Ensure static and upload directories exist
os.makedirs(os.path.join(STATIC_DIR, "uploads"), exist_ok=True)
os.makedirs(os.path.join(STATIC_DIR, "annotated"), exist_ok=True)

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-warm YOLO11m detector, VehicleDetector, SAM2 segmenter, Depth Anything V2, and Knowledge Graph singletons."""
    try:
        get_detector()
    except Exception as e:
        print(f"[!] Warning: Could not pre-load YOLO damage detector: {e}")

    try:
        get_vehicle_detector()
    except Exception as e:
        print(f"[!] Warning: Could not pre-load Vehicle detector: {e}")

    try:
        get_segmenter()
    except Exception as e:
        print(f"[!] Warning: Could not pre-load SAM2 segmenter: {e}")

    try:
        get_depth_estimator()
    except Exception as e:
        print(f"[!] Warning: Could not pre-load Depth Anything V2 estimator: {e}")

    try:
        get_knowledge_graph()
    except Exception as e:
        print(f"[!] Warning: Could not pre-load Vehicle Structural Knowledge Graph: {e}")
    yield


app = FastAPI(
    title="Explainable Vehicle Damage Assessment API",
    description=(
        "Phase 2: Computer Vision Damage Pipeline (YOLO11m Detection + "
        "SAM2 Pixel-Level Damage Segmentation), Multi-View Aggregation, "
        "and Claim Persistence for Insurance Surveyors."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for annotated images and uploads
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Mount frontend if directory exists
if os.path.exists(FRONTEND_DIR):
    app.mount("/portal", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


# Include API routers
app.include_router(inspection_router)


@app.get("/", tags=["Root"])
async def root_redirect():
    """Root route redirecting to the surveyor portal or API overview."""
    index_html = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_html):
        return FileResponse(index_html)
    return {
        "project": "Hybrid Explainable AI Framework for Intelligent Vehicle Damage Assessment",
        "phase": "Phase 1 (Vision Pipeline + Backend + Surveyor Portal)",
        "docs": "/docs",
        "surveyor_portal": "/portal/",
        "api_inspect": "/api/claims/{claim_id}/inspect",
    }


@app.get("/api/health", tags=["System Health"])
async def health_check():
    """System health and AI model readiness check."""
    try:
        detector = get_detector()
        yolo_status = {
            "loaded": True,
            "weights_path": detector.weights_path,
            "classes_count": len(detector.classes),
            "classes": detector.classes,
        }
    except Exception as e:
        yolo_status = {
            "loaded": False,
            "error": str(e),
        }

    try:
        vehicle_detector = get_vehicle_detector()
        vehicle_status = {
            "loaded": True,
            "model_path": vehicle_detector.model_path,
        }
    except Exception as e:
        vehicle_status = {
            "loaded": False,
            "error": str(e),
        }

    try:
        segmenter = get_segmenter()
        sam2_status = {
            "loaded": True,
            "device": segmenter.device,
            "checkpoint_path": segmenter.checkpoint_path,
            "config_path": segmenter.config_path,
            "amg_support": True,
        }
    except Exception as e:
        sam2_status = {
            "loaded": False,
            "error": str(e),
        }

    try:
        depth_est = get_depth_estimator()
        depth_status = {
            "loaded": True,
            "device": depth_est.device,
            "model_type": depth_est.model_type,
            "model_id": depth_est.model_id,
        }
    except Exception as e:
        depth_status = {
            "loaded": False,
            "error": str(e),
        }

    try:
        kg = get_knowledge_graph()
        kg_status = {
            "loaded": True,
            "mode": "neo4j_live" if kg.is_neo4j_connected else "embedded_graph",
            "neo4j_connected": kg.is_neo4j_connected,
            "uri": kg.uri,
            "nodes_count": len(kg.embedded_store.nodes),
        }
    except Exception as e:
        kg_status = {
            "loaded": False,
            "error": str(e),
        }

    try:
        groq_client = get_groq_client()
        groq_status = {
            "loaded": True,
            "mock_mode": groq_client.is_mock_mode(),
            "vision_model": groq_client.vision_model,
            "text_model": groq_client.text_model,
        }
    except Exception as e:
        groq_status = {
            "loaded": False,
            "error": str(e),
        }

    is_healthy = (
        yolo_status.get("loaded", False)
        and sam2_status.get("loaded", False)
        and vehicle_status.get("loaded", False)
        and depth_status.get("loaded", False)
        and kg_status.get("loaded", False)
        and groq_status.get("loaded", False)
    )

    return {
        "status": "HEALTHY" if is_healthy else "DEGRADED",
        "vision_pipeline": {
            "damage_detector_yolo11m": yolo_status,
            "vehicle_detector_yolo11n": vehicle_status,
            "damage_segmenter_sam2": sam2_status,
            "depth_anything_v2": depth_status,
            "structural_knowledge_graph": kg_status,
        },
        "phase5_llm_pipeline": {
            "groq_service": groq_status,
            "document_understanding_ocr": "Operational (Two-Stage: PaddleOCR + Groq Text LLM)",
            "llm_explainable_report": "Operational (Groq Text LLM)",
        },
        "storage": {
            "type": "LocalJsonRepository (Thread-Safe Persistence)",
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)


