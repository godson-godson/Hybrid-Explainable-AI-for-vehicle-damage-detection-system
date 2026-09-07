# Hybrid Explainable AI Framework for Intelligent Vehicle Damage Assessment and Insurance Claim Assistance

An academic MCA project integrating deep computer vision, multi-view damage aggregation, structural reasoning, and explainable AI for insurance claim surveying.

---

## 📌 Phase 1 Scope (Implemented)

Phase 1 delivers the operational **Computer Vision Pipeline**, **FastAPI Backend**, and **Surveyor Web Portal** powered by the custom fine-tuned **YOLO11m** damage detection model.

```
┌───────────────────────────────┐
│     Surveyor Web Portal       │ ◄─── Browser UI (Multi-view photo upload & inspection)
└──────────────┬────────────────┘
               │  POST /api/claims/{claim_id}/inspect
               ▼
┌───────────────────────────────┐
│     FastAPI Backend           │ ◄─── Validation, aggregation, & persistence
└──────────────┬────────────────┘
               │
       ┌───────┴───────────────────────────────┐
       ▼                                       ▼
┌───────────────────────────────┐   ┌───────────────────────────────┐
│   YOLO11m Damage Detector     │   │     Storage Repository        │
│   (dent, scratch, crack,      │   │   (Local JSON in Phase 1,     │
│    glass, lamp, flat_tire)    │   │    MongoDB in Phase 2)        │
└───────────────────────────────┘   └───────────────────────────────┘
               │
               ▼
┌───────────────────────────────────────────────────────────────────┐
│                  Phase 2 Pipeline Stubs (Ready)                   │
│   • document_ocr.py        (Document Understanding / KYC OCR)     │
│   • damage_segmentation.py (SAM2 Pixel-Level Segmentation)        │
│   • depth_estimation.py    (Depth Anything V2 3D Indentation)     │
│   • knowledge_graph.py     (Neo4j Structural Dependency Reasoning)│
│   • report_generator.py    (LLM-based Explainable Report Dossier) │
└───────────────────────────────────────────────────────────────────┘
```

---

## 🔍 Model Information

- **Architecture:** YOLO11m (`DetectionModel`, 231 layers, ~20M parameters)
- **Weights File:** `YOLO11m-Car-Damage-Detector-main/trained.pt`
- **Trained Classes:**
  1. `0: dent`
  2. `1: scratch`
  3. `2: crack`
  4. `3: shattered_glass`
  5. `4: broken_lamp`
  6. `5: flat_tire`

---

## 📂 Project Structure

```
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   └── routes_inspection.py  # Upload, YOLO inference & claim query endpoints
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   └── schemas.py            # Pydantic schemas for Phase 1 & Phase 2 stubs
│   │   ├── pipeline/                 # Downstream modular stubs
│   │   │   ├── __init__.py
│   │   │   ├── document_ocr.py       # STUB: OCR document extraction
│   │   │   ├── damage_segmentation.py# STUB: SAM2 polygon segmentation
│   │   │   ├── depth_estimation.py   # STUB: Depth Anything V2 3D depth
│   │   │   ├── knowledge_graph.py    # STUB: Neo4j component graph reasoning
│   │   │   └── report_generator.py   # STUB: LLM explainable claim dossier
│   │   ├── storage/
│   │   │   ├── __init__.py
│   │   │   └── repository.py         # Abstract persistence layer (JSON / MongoDB)
│   │   ├── vision/
│   │   │   ├── __init__.py
│   │   │   └── damage_detector.py    # YOLO11m model wrapper & visual annotator
│   │   ├── __init__.py
│   │   └── main.py                   # FastAPI application entrypoint
│   ├── data/                         # Local JSON database storage
│   └── static/                       # Uploaded and annotated images
├── frontend/                         # Surveyor Web Portal
│   ├── index.html                    # Responsive Surveyor UI
│   ├── css/
│   │   └── style.css                 # Dark theme dashboard styles
│   └── js/
│       └── app.js                    # Multi-view upload, preview & API client
├── YOLO11m-Car-Damage-Detector-main/ # Model weights and training assets
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation
```

---

## 🚀 Quickstart Guide

### 1. Environment Setup

Make sure Python 3.10+ is installed:

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Install dependencies (CPU PyTorch + Ultralytics + FastAPI)
pip install -r requirements.txt
```

### 2. Start the FastAPI Backend & Surveyor Portal

From the project root:

```bash
# Run FastAPI server
.venv/bin/python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Open the Surveyor Portal

Navigate to:
👉 **[http://localhost:8000](http://localhost:8000)** or **[http://localhost:8000/portal/](http://localhost:8000/portal/)**

Interactive Swagger API Docs are available at:
👉 **[http://localhost:8000/docs](http://localhost:8000/docs)**

---

## 🧪 API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/claims/{claim_id}/inspect` | Upload multi-view photos, run YOLO11m, return detections & damage summary |
| `GET` | `/api/claims/{claim_id}` | Retrieve stored claim assessment and annotated image links |
| `GET` | `/api/claims` | List recently processed insurance claims |
| `DELETE`| `/api/claims/{claim_id}` | Delete a claim record |
| `GET` | `/api/health` | Check YOLO11m model status and stub readiness |

---

## 🔮 Pipeline Modules Status (Phase 2 & Phase 3)

- **Module 1 (Document OCR):** `document_ocr.extract_document_fields(image_bytes)` *(Phase 3 Stub)*
- **Module 2 (SAM2 Damage Segmentation):** **[OPERATIONAL]** Powered by Meta's Segment Anything Model 2 (`sam2.1_hiera_tiny.pt`). Prompts SAM2 using YOLO bounding boxes to generate pixel-level masks, damage surface area in pixels, frame area percentage, and contour polygon points.
- **Module 3 (Depth Anything V2):** `depth_estimation.estimate_damage_depth(image_bytes, mask)` *(Phase 3 Stub)*
- **Module 4 (Neo4j Graph):** `knowledge_graph.recommend_inspections(damage_summary)` *(Phase 3 Stub)*
- **Module 5 (LLM Report):** `report_generator.generate_claim_report(claim_data)` *(Phase 3 Stub)*

### SAM2 Configuration Options

Configure SAM2 behavior via environment variables:

| Environment Variable | Default | Description |
| :--- | :--- | :--- |
| `SAM2_CHECKPOINT_PATH` | `backend/weights/sam2/sam2.1_hiera_tiny.pt` | Path to SAM 2.1 model checkpoint (`.pt`) |
| `SAM2_MODEL_CONFIG` | `configs/sam2.1/sam2.1_hiera_t.yaml` | Model configuration YAML file |
| `SAM2_DEVICE` | Auto (`cuda` if available, else `cpu`) | Inference device (`cuda` or `cpu`) |

