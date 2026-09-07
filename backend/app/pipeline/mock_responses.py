"""
Deterministic Mock Responses for Phase 5 (Groq LLM Fallback).
Provides hardcoded, schema-compliant responses for:
1. Registration Certificate (RC Book)
2. Motor Insurance Policy
3. Driving Licence
4. Explainable Insurance Survey Report (Chapter 4.6.5)

Ensures zero UI breakage during offline demonstrations, viva evaluations,
or when GROQ_API_KEY is unset.
"""

import json
from typing import Dict, Any, Optional

MOCK_RC_JSON = {
    "registration_number": {
        "value": "DL 03 CC 4921",
        "source_snippet": "REGN NO: DL 03 CC 4921",
        "confidence": 0.96
    },
    "owner_name": {
        "value": "Rajesh Kumar Sharma",
        "source_snippet": "OWNER NAME: RAJESH KUMAR SHARMA",
        "confidence": 0.94
    },
    "vehicle_make_model": {
        "value": "Hyundai i20 Asta 1.2 Petrol",
        "source_snippet": "MAKER / MODEL: HYUNDAI I20 ASTA 1.2",
        "confidence": 0.93
    },
    "chassis_number": {
        "value": "MALBB51BLAM123456",
        "source_snippet": "CHASSIS NO: MALBB51BLAM123456",
        "confidence": 0.95
    },
    "engine_number": {
        "value": "G4LAJ123456",
        "source_snippet": "ENGINE NO: G4LAJ123456",
        "confidence": 0.92
    },
    "registration_date": {
        "value": "14/08/2021",
        "source_snippet": "DATE OF REGN: 14/08/2021",
        "confidence": 0.91
    },
    "fuel_type": {
        "value": "PETROL",
        "source_snippet": "FUEL: PETROL",
        "confidence": 0.98
    },
    "vehicle_class": {
        "value": "LMV (Light Motor Vehicle)",
        "source_snippet": "CLASS OF VEHICLE: LMV (MOTOR CAR)",
        "confidence": 0.95
    }
}

MOCK_POLICY_JSON = {
    "policy_number": {
        "value": "POL-2026-9874120",
        "source_snippet": "POLICY NO: POL-2026-9874120",
        "confidence": 0.97
    },
    "insurer_name": {
        "value": "National General Insurance Corp",
        "source_snippet": "INSURER: NATIONAL GENERAL INSURANCE CORP LTD",
        "confidence": 0.98
    },
    "policy_start_date": {
        "value": "15/08/2025",
        "source_snippet": "EFFECTIVE FROM: 15/08/2025 00:00 HRS",
        "confidence": 0.94
    },
    "policy_expiry_date": {
        "value": "14/08/2026",
        "source_snippet": "EXPIRY DATE: 14/08/2026 MIDNIGHT",
        "confidence": 0.95
    },
    "insured_name": {
        "value": "Rajesh Kumar Sharma",
        "source_snippet": "INSURED NAME: RAJESH KUMAR SHARMA",
        "confidence": 0.96
    },
    "vehicle_reg_number": {
        "value": "DL 03 CC 4921",
        "source_snippet": "VEHICLE REGN: DL 03 CC 4921",
        "confidence": 0.97
    },
    "insured_declared_value": {
        "value": "650000",
        "source_snippet": "INSURED DECLARED VALUE (IDV): RS. 6,50,000/-",
        "confidence": 0.93
    },
    "coverage_type": {
        "value": "Comprehensive Private Car Package (Zero Depreciation)",
        "source_snippet": "COVERAGE: COMPREHENSIVE ZERO DEP ADD-ON",
        "confidence": 0.94
    }
}

MOCK_LICENCE_JSON = {
    "licence_number": {
        "value": "DL-0420180054321",
        "source_snippet": "DL NO: DL-0420180054321",
        "confidence": 0.96
    },
    "holder_name": {
        "value": "Rajesh Kumar Sharma",
        "source_snippet": "NAME: RAJESH KUMAR SHARMA",
        "confidence": 0.95
    },
    "validity_date": {
        "value": "12/05/2038",
        "source_snippet": "VALID TILL (NT): 12/05/2038",
        "confidence": 0.93
    },
    "vehicle_classes_authorized": {
        "value": "MCWG, LMV",
        "source_snippet": "COV: MCWG, LMV",
        "confidence": 0.95
    },
    "issue_date": {
        "value": "13/05/2018",
        "source_snippet": "DATE OF ISSUE: 13/05/2018",
        "confidence": 0.94
    }
}

MOCK_SURVEY_REPORT_JSON = {
    "executive_summary": "Intelligent automated survey synthesis for vehicle DL 03 CC 4921 confirms significant frontal and right-side impact with multiple localized dents, cracks, and scratches across the front bumper cover, hood, and right front fender. Deep surface indentation metrics derived from Depth Anything V2 indicate plastic deformation along structural load paths, warranting internal borescope inspection of frame rails and radiator support before final claim settlement.",
    "damage_summary": {
        "total_classified_instances": 11,
        "counts_by_type": {
            "dent": 5,
            "crack": 3,
            "scratch": 2,
            "flat_tire": 1
        },
        "primary_impact_zones": ["front", "right"],
        "severity_tier": "Severe (Major Structural/Component Impact)"
    },
    "surface_deformation_narrative": "Depth Anything V2 monocular depth analysis surfaces relative indentation values ranging up to +0.07 to +0.16 relative units in the front bumper and hood areas. The right front fender exhibits severe surface irregularity (depth std deviation 0.089) and recessed dent deformation. These deformation profiles exceed standard cosmetic thresholds, signaling kinetic energy transmission past the outer fascia.",
    "internal_inspection_plan": [
        {
            "component_name": "front longitudinal frame rails",
            "priority": "HIGH",
            "risk_score": 0.496,
            "recommended_action": "Visual Borescope & Mount Tolerance Inspection",
            "estimated_labor_hours": 0.5,
            "rationale": "Convergence of forces from right front fender (DENT) and front bumper cover (CRACK) traversing bumper reinforcement bar and crush cans into frame rails."
        },
        {
            "component_name": "radiator support assembly",
            "priority": "MEDIUM",
            "risk_score": 0.359,
            "recommended_action": "Visual Borescope & Mount Tolerance Inspection",
            "estimated_labor_hours": 0.5,
            "rationale": "Impact energy localized on front bumper cover and hood propagating to radiator upper support brackets."
        },
        {
            "component_name": "hood latch and release cable",
            "priority": "LOW",
            "risk_score": 0.312,
            "recommended_action": "Secondary Check during Reassembly",
            "estimated_labor_hours": 0.5,
            "rationale": "Direct frontal hood deformation creates potential misalignment of latching safety mechanisms."
        }
    ],
    "explainable_reasoning": "The visual pipeline localized primary damage clusters on the front bumper cover and hood. The neuro-symbolic knowledge graph traversed collision load paths connecting these external panels to internal structural members. Because the bumper reinforcement bar directly shields the front crush cans, the high deformation score (+0.16) on the bumper implies that kinetic forces were transmitted deeper than superficial paint damage. Consequently, teardown and borescope verification of the front longitudinal frame rails is recommended to ensure vehicle roadworthiness before authorization.",
    "claim_disposition": "Conditional Approval Pending Teardown Inspection",
    "estimated_repair_cost_min": 28500.0,
    "estimated_repair_cost_max": 42000.0,
    "surveyor_action_items": [
        "Perform physical borescope examination of front longitudinal frame rails.",
        "Check radiator support assembly alignment and coolant line tolerances.",
        "Confirm zero-depreciation policy coverage validity for bumper and headlamp assemblies."
    ]
}

MOCK_SURVEY_REPORT_MARKDOWN = """# 📋 Intelligent Vehicle Damage Assessment & Survey Dossier
**AI-Assisted Claim Survey Report (Chapter 4.6.5 Compliance)**
*Status: DRAFT — Pending Surveyor Final Sign-Off*

---

### 1. Executive Summary
Intelligent automated survey synthesis for vehicle **DL 03 CC 4921** (Claim ID: `{claim_id}`) confirms significant frontal and right-quarter impact. Multi-view computer vision (YOLO11m + SAM2) and monocular depth deformation modeling (Depth Anything V2) identified **11 classified damage instances** across the front bumper cover, hood, and right front fender. Structural reasoning via the Vehicle Knowledge Graph recommends targeted borescope inspection of internal load-bearing members prior to authorization.

---

### 2. Verified Document & Policy Credentials
| Document | Reference Number | Holder / Insured | Validity / Expiry | Verification Status |
| :--- | :--- | :--- | :--- | :--- |
| **RC Book** | `DL 03 CC 4921` | Rajesh Kumar Sharma | Regn: 14/08/2021 | ✅ Match Confirmed |
| **Insurance Policy** | `POL-2026-9874120` | Rajesh Kumar Sharma | Exp: 14/08/2026 | ✅ Active (Zero-Dep) |
| **Driving Licence** | `DL-0420180054321` | Rajesh Kumar Sharma | Val: 12/05/2038 | ✅ Valid LMV |

*Cross-Validation Audit:* Registration numbers across Registration Certificate and Policy match with 100% concordance. Policy is in active force on date of inspection.

---

### 3. Comprehensive Damage Breakdown (Vision + SAM2)
* **Overall Assessed Severity:** `Severe (Major Structural/Component Impact)`
* **Total Damage Count:** 11 instances
* **Damage Class Distribution:**
  - **Dents:** 5 instances (Front Bumper, Hood, Right Fender)
  - **Cracks:** 3 instances (Front Bumper Cover, Lower Grille)
  - **Scratches:** 2 instances (Right Fender, Front Bumper)
  - **Flat Tire:** 1 instance (Right Front Wheel Assembly)

---

### 4. Surface Deformation Analysis (Depth Anything V2)
Depth Anything V2 relative monocular depth calculations demonstrate plastic deformation with peak indentation metrics between **+0.07 and +0.16 relative units**. 
* **Hood & Front Bumper:** Severe indentation along bumper center line indicates high kinetic impact exceeding elastic recovery limits.
* **Surface Irregularity:** Depth standard deviation on the right fender mask is elevated (0.089), confirming crumpled metal geometry rather than superficial abrasive paint transfer.

---

### 5. Probabilistic Internal Component Inspection Plan (Knowledge Graph)
*Concealed safety-critical components flagged via neuro-symbolic force propagation traversal:*

| Priority | Component | Load Path Propagation | Rec. Action | Est. Labor |
| :--- | :--- | :--- | :--- | :--- |
| 🔴 **HIGH** | **Front Longitudinal Frame Rails** | `[Right Fender -> Frame Rails]` & `[Bumper -> Rebar -> Crush Cans -> Rails]` | Visual Borescope & Mount Tolerance | 0.5 hrs |
| 🟡 **MEDIUM** | **Radiator Support Assembly** | `[Hood -> Rad Support]` & `[Front Bumper -> Rad Support]` | Alignment & Fluid Line Check | 0.5 hrs |
| 🟢 **LOW** | **Hood Latch & Release Cable** | `[Hood -> Hood Latch]` | Secondary Check during Reassembly | 0.5 hrs |

---

### 6. Explainable Reasoning & Surveyor Disposition
* **Causal Linkage:** Frontal impact forced the bumper cover against the reinforcement bar, propagating stresses to the radiator mounting brackets and crush cans.
* **Estimated Preliminary Repair Range:** ₹28,500 – ₹42,000 (Parts Replacement & Paint Labor)
* **Surveyor Recommendation:** **CONDITIONAL APPROVAL PENDING TEARDOWN INSPECTION**
* **Action Directive:** Authorize external cosmetic repairs only after garage technician completes borescope verification of the front longitudinal frame rails.
"""


MOCK_RC_OCR_LINES = [
    "FORM 23 - CERTIFICATE OF REGISTRATION",
    "REGISTRATION NUMBER: DL 03 CC 4921",
    "OWNER NAME: RAJESH KUMAR SHARMA",
    "CHASSIS NUMBER: MALBB51BLAM123456",
    "ENGINE NUMBER: G4LAJ123456",
    "REGISTRATION DATE: 14/08/2021",
    "FUEL: PETROL",
    "CLASS: LMV-MOTOR CAR",
]

MOCK_POLICY_OCR_LINES = [
    "MOTOR VEHICLE INSURANCE CERTIFICATE CUM POLICY SCHEDULE",
    "POLICY NUMBER: POL-2026-9874120",
    "INSURER: ICICI LOMBARD GENERAL INSURANCE CO LTD",
    "PERIOD OF INSURANCE: 15/08/2025 TO 14/08/2026",
    "INSURED NAME: RAJESH KUMAR SHARMA",
    "REGISTRATION NO: DL 03 CC 4921",
    "INSURED DECLARED VALUE (IDV): INR 645000",
    "POLICY TYPE: COMPREHENSIVE ZERO DEPRECIATION",
]

MOCK_LICENCE_OCR_LINES = [
    "UNION OF INDIA - DRIVING LICENCE",
    "LICENCE NO: DL-0420180054321",
    "NAME: RAJESH KUMAR SHARMA",
    "VALID TILL: 12/05/2038",
    "VEHICLE CLASS: LMV, MCWG",
    "DATE OF ISSUE: 12/05/2018",
]


def get_mock_document_response(document_type: str) -> Dict[str, Any]:
    """Return mock extracted fields for a given document type."""
    doc_type = document_type.lower()
    if "rc" in doc_type or "reg" in doc_type:
        return MOCK_RC_JSON
    elif "policy" in doc_type or "insur" in doc_type:
        return MOCK_POLICY_JSON
    elif "licence" in doc_type or "license" in doc_type or "dl" in doc_type:
        return MOCK_LICENCE_JSON
    else:
        return MOCK_RC_JSON


def get_mock_ocr_corpus(document_type: str) -> Dict[str, Any]:
    """Return mock OCR lines and corpus for deterministic testing."""
    doc_type = document_type.lower()
    if "rc" in doc_type or "reg" in doc_type:
        lines = MOCK_RC_OCR_LINES
    elif "policy" in doc_type or "insur" in doc_type:
        lines = MOCK_POLICY_OCR_LINES
    elif "licence" in doc_type or "license" in doc_type or "dl" in doc_type:
        lines = MOCK_LICENCE_OCR_LINES
    else:
        lines = MOCK_RC_OCR_LINES
    corpus = "\n".join(lines)
    return {
        "lines": lines,
        "scores": [0.98] * len(lines),
        "corpus": corpus,
        "char_count": len(corpus),
    }


def get_mock_survey_report(claim_id: str = "CLAIM-MOCK-001") -> Dict[str, Any]:
    """Return mock structured report and populated markdown."""
    md = MOCK_SURVEY_REPORT_MARKDOWN.replace("{claim_id}", claim_id)
    return {
        "structured_data": MOCK_SURVEY_REPORT_JSON,
        "markdown_report": md
    }
