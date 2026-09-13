"""
Explainable Survey Report Generation Module — Phase 5.
Synthesizes multi-view YOLO11m detections, SAM2 segmentation metrics,
Depth Anything V2 monocular deformations, Neo4j structural knowledge graph risks,
and verified KYC/policy credentials into an audit-ready, surveyor-grade report dossier.
(Chapter 4.6.5 Compliance).
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..models.schemas import (
    ClaimDocuments,
    ClaimInspectionResponse,
    DamageDetection,
    InspectionRecommendation,
    LLMSurveyReport,
)
from .groq_client import GroqClientError, get_groq_client
from .mock_responses import get_mock_survey_report

logger = logging.getLogger("report_generator")


class ReportGenerationError(Exception):
    """Raised when survey report generation fails without fabricated fallbacks."""
    pass


# ---------------------------------------------------------------------------
# Clearly Named System and Instruction Prompts
# ---------------------------------------------------------------------------

REPORT_GENERATOR_SYSTEM_PROMPT = """You are synthesizing an insurance survey report from pipeline outputs provided below. Only use the facts, figures, classifications, and recommendations provided in the input. Do not invent additional damage, statistics, components, or conclusions not present in the input. If something relevant is missing from the input, state that plainly rather than filling the gap.

DECISION-MAKING AND PRICING CONSTRAINTS:
1. This system is an explainable decision-support tool (Chapter 4.6.5). It does NOT possess legal or adjudicative authority to approve or deny insurance claims. The licensed insurance surveyor retains sole and independent decision-making authority. You must NEVER output "Approve" or "Reject".
2. Financial repair estimation requires physical parts-pricing and garage labor rate schedules not available to this pipeline. Do NOT generate financial repair estimates.
3. The internal component inspection plan and damage metrics are maintained with deterministic precision by Python from the Knowledge Graph — do not re-emit component tables or lists.

Your synthesis must provide high-quality analytical narratives adhering to the academic standards of Chapter 4.6.5 (Explainable Survey Reporting):
1. Executive Summary: High-level overview of the incident, vehicle credentials, and assessed severity.
2. Surface Deformation Analysis narrative: Coherent explanation of monocular indentation depth metrics, plastic deformation, and surface irregularity.
3. Explainable Reasoning Narrative: Plain-language cause-and-effect narrative connecting external panel impact points, force transmission load paths, and concealed internal component vulnerabilities so a non-technical insurance surveyor can immediately understand the engineering justification.
4. Surveyor Recommendation: Professional assistive audit directive based on structural risk and damage severity. Valid choices:
   - "Detailed Teardown Audit Required" (when high-risk concealed structural components or severe deformations exist)
   - "Physical Verification Recommended" (when moderate damage or document audit flags require surveyor verification)
   - "Cosmetic Survey Verification" (when only superficial low-severity cosmetic damages exist)
   - "Document Reconciliation Required" (when cross-validation mismatches are detected)
5. Surveyor Action Items: Specific, actionable physical inspection steps for the surveyor.

OUTPUT FORMAT REQUIREMENTS:
Output ONLY valid, raw JSON. Do not include markdown fences (```json) or introductory chit-chat.
Return a JSON object with this exact structure:
{
  "executive_summary": "...",
  "surface_deformation_narrative": "...",
  "explainable_reasoning": "...",
  "surveyor_recommendation": "Detailed Teardown Audit Required" | "Physical Verification Recommended" | "Cosmetic Survey Verification" | "Document Reconciliation Required",
  "surveyor_action_items": [ "action 1", "action 2" ]
}
"""


def _prepare_compact_pipeline_payload(
    claim_data: ClaimInspectionResponse,
    documents: Optional[ClaimDocuments] = None,
) -> Dict[str, Any]:
    """
    Summarize pipeline outputs into a compact, numerical payload.
    Avoids passing raw mask arrays, polygon coordinates, or depth colormaps
    to prevent context window overflow and 400 Bad Request errors.
    """
    # 1. Metadata
    payload_metadata = {
        "claim_id": claim_data.claim_id,
        "vehicle_reg_number": claim_data.vehicle_reg_number or "Not Provided",
        "surveyor_notes": claim_data.surveyor_notes or "None provided",
        "timestamp": claim_data.created_at,
    }

    # 2. Document Verification Summary
    docs_summary: Dict[str, Any] = {"status": "No documents submitted"}
    if documents:
        docs_summary = {
            "cross_validation_passed": documents.cross_validation_passed,
            "cross_validation_notes": documents.cross_validation_notes,
            "rc_book": {
                name: item.value
                for name, item in (documents.rc_book.fields.items() if documents.rc_book else {}.items())
                if item.value
            },
            "insurance_policy": {
                name: item.value
                for name, item in (documents.insurance_policy.fields.items() if documents.insurance_policy else {}.items())
                if item.value
            },
            "driving_licence": {
                name: item.value
                for name, item in (documents.driving_licence.fields.items() if documents.driving_licence else {}.items())
                if item.value
            },
        }

    # 3. Compact Visual Damage Instances (No Polygon Arrays or Pixels)
    compact_damages: List[Dict[str, Any]] = []
    all_instances = []
    for img in claim_data.images:
        for d in img.detections:
            all_instances.append(d)

    for i, d in enumerate(all_instances):
        damage_entry: Dict[str, Any] = {
            "id": i + 1,
            "damage_type": d.damage_type,
            "view_angle": d.view_angle or "unknown",
            "panel": d.detected_panel or "unknown",
            "confidence": round(d.confidence, 3),
        }

        # Compact SAM2 area metrics (Numbers ONLY, zero masks)
        if d.segmentation:
            damage_entry["sam2_area_pct"] = round(d.segmentation.area_percentage, 2)
            damage_entry["sam2_area_px"] = d.segmentation.area_pixels
            if d.segmentation.mask_confidence:
                damage_entry["sam2_mask_conf"] = round(d.segmentation.mask_confidence, 2)

        # Compact Depth Anything V2 Deformation metrics
        if d.deformation:
            damage_entry["relative_deformation"] = (
                round(d.deformation.relative_deformation_score, 3)
                if d.deformation.relative_deformation_score is not None
                else None
            )
            damage_entry["max_relative_deformation"] = (
                round(d.deformation.max_relative_deformation, 3)
                if d.deformation.max_relative_deformation is not None
                else None
            )
            damage_entry["surface_irregularity"] = (
                round(d.deformation.surface_irregularity, 3)
                if d.deformation.surface_irregularity is not None
                else None
            )
            damage_entry["deformation_severity"] = d.deformation.severity_tier or "minor"
            damage_entry["deformation_type"] = d.deformation.deformation_type or "planar"

        compact_damages.append(damage_entry)

    # 4. Aggregated Damage Summary
    damage_summary_dict = {}
    if claim_data.damage_summary:
        damage_summary_dict = {
            "total_damages_count": claim_data.damage_summary.total_damages_count,
            "damage_counts_by_type": claim_data.damage_summary.damage_counts_by_type,
            "severity_assessment": claim_data.damage_summary.severity_assessment,
            "damages_by_view": claim_data.damage_summary.damages_by_view,
        }

    # 5. Compact Knowledge Graph Internal Component Risk Matrix
    compact_kg_recs: List[Dict[str, Any]] = []
    for r in claim_data.structural_risk_matrix:
        load_path_str = " -> ".join(r.load_path) if isinstance(r.load_path, list) else str(r.load_path)
        compact_kg_recs.append({
            "component_name": r.component_name,
            "source_panel": r.source_panel,
            "impact_zone": r.impact_zone,
            "risk_score": round(r.risk_score, 3),
            "safety_risk_level": r.safety_risk_level,
            "recommended_action": r.recommended_action,
            "estimated_labor_hours": r.estimated_labor_hours,
            "load_path": load_path_str,
            "rationale": r.rationale,
        })

    return {
        "claim_metadata": payload_metadata,
        "verified_documents": docs_summary,
        "damage_summary": damage_summary_dict,
        "classified_damage_instances": compact_damages,
        "knowledge_graph_structural_recommendations": compact_kg_recs,
    }


def _strip_json(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON object from text."""
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        cleaned = cleaned[first_brace:last_brace + 1]

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


def _generate_dynamic_mock_narrative(
    claim_data: ClaimInspectionResponse,
    documents: Optional[ClaimDocuments] = None,
) -> Dict[str, Any]:
    """Generate deterministic dynamic mock narrative adapted to actual claim data."""
    reg = claim_data.vehicle_reg_number or "Not Provided"
    total_damages = claim_data.damage_summary.total_damages_count if claim_data.damage_summary else 0
    severity = claim_data.damage_summary.severity_assessment if claim_data.damage_summary else "Minor"

    top_components = [r.component_name for r in (claim_data.structural_risk_matrix or [])[:2]]
    comp_phrase = f", warranting targeted internal borescope inspection of {', '.join(top_components)}" if top_components else ""

    has_high = any(r.safety_risk_level == "HIGH" for r in (claim_data.structural_risk_matrix or []))
    rec = (
        "Detailed Teardown Audit Required"
        if has_high or total_damages >= 3
        else "Physical Verification Recommended" if total_damages > 0
        else "Cosmetic Survey Verification"
    )

    action_items = []
    if top_components:
        action_items.append(f"Perform physical borescope examination of {', '.join(top_components)}.")
    action_items.append("Verify structural mount tolerances and panel alignment during garage teardown.")
    if documents and documents.cross_validation_passed:
        action_items.append("Confirm active zero-depreciation policy endorsements with underwriter.")

    return {
        "executive_summary": (
            f"Intelligent automated survey synthesis for vehicle {reg} (Claim ID: {claim_data.claim_id}) "
            f"identifies {total_damages} classified damage instances across inspected exterior panels. "
            f"Overall assessed damage severity is {severity}{comp_phrase} prior to final settlement."
        ),
        "surface_deformation_narrative": (
            f"Monocular depth analysis via Depth Anything V2 detects measurable surface irregularity "
            f"and localized depression contours on impacted panels, indicating kinetic force absorption."
        ),
        "explainable_reasoning": (
            f"Exterior impact loads traverse connected structural load paths. For vehicle {reg}, "
            f"kinetic energy directed into impacted panels propagates along chassis frame paths, "
            f"necessitating physical surveyor audit of concealed load-bearing members."
        ),
        "surveyor_recommendation": rec,
        "surveyor_action_items": action_items,
    }


def _build_markdown_dossier(
    claim_data: ClaimInspectionResponse,
    documents: Optional[ClaimDocuments],
    narrative: Dict[str, Any],
    internal_inspection_plan: List[Dict[str, Any]],
    total_labor_hours: float,
) -> str:
    """
    Deterministically build publication-grade Markdown dossier in Python.
    Guarantees 100% factual accuracy of tables and numbers without LLM escaping glitches.
    """
    claim_id = claim_data.claim_id
    veh_reg = claim_data.vehicle_reg_number or "Not Provided"
    exec_summary = narrative.get("executive_summary", "Automated survey synthesis completed.")
    surface_narrative = narrative.get("surface_deformation_narrative", "Surface deformation analysis completed.")
    explainable_reasoning = narrative.get("explainable_reasoning", "Explainable reasoning generated from structural load paths.")
    recommendation = narrative.get("surveyor_recommendation", "Detailed Teardown Audit Required")
    action_items = narrative.get("surveyor_action_items") or [
        "Physically inspect flagged high-priority internal components.",
        "Verify panel alignment and gap tolerances during teardown.",
    ]

    # Section 2: Verified Document & Policy Credentials
    if documents and (documents.rc_book or documents.insurance_policy or documents.driving_licence):
        rc_num = documents.rc_book.fields.get("registration_number", {}).value if (documents.rc_book and "registration_number" in documents.rc_book.fields) else (veh_reg if veh_reg != "Not Provided" else "N/A")
        rc_owner = documents.rc_book.fields.get("owner_name", {}).value if (documents.rc_book and "owner_name" in documents.rc_book.fields) else "N/A"
        rc_date = documents.rc_book.fields.get("registration_date", {}).value if (documents.rc_book and "registration_date" in documents.rc_book.fields) else "N/A"
        rc_status = "✅ Match Confirmed" if documents.cross_validation_passed else ("⚠️ Review Required" if (documents.rc_book and documents.rc_book.needs_manual_review) else "✅ Active")

        pol_num = documents.insurance_policy.fields.get("policy_number", {}).value if (documents.insurance_policy and "policy_number" in documents.insurance_policy.fields) else "N/A"
        pol_insured = documents.insurance_policy.fields.get("insured_name", {}).value if (documents.insurance_policy and "insured_name" in documents.insurance_policy.fields) else rc_owner
        pol_exp = documents.insurance_policy.fields.get("policy_expiry_date", {}).value if (documents.insurance_policy and "policy_expiry_date" in documents.insurance_policy.fields) else "N/A"
        pol_status = "✅ Active Policy" if (documents.insurance_policy and not documents.insurance_policy.needs_manual_review) else "⚠️ Verification Needed"

        dl_num = documents.driving_licence.fields.get("licence_number", {}).value if (documents.driving_licence and "licence_number" in documents.driving_licence.fields) else "N/A"
        dl_name = documents.driving_licence.fields.get("holder_name", {}).value if (documents.driving_licence and "holder_name" in documents.driving_licence.fields) else rc_owner
        dl_val = documents.driving_licence.fields.get("validity_expiry", {}).value if (documents.driving_licence and "validity_expiry" in documents.driving_licence.fields) else "N/A"
        dl_status = "✅ Valid Licence" if (documents.driving_licence and not documents.driving_licence.needs_manual_review) else "⚠️ Verification Needed"

        notes_str = "; ".join(documents.cross_validation_notes) if documents.cross_validation_notes else (
            "Registration numbers match across Registration Certificate and Policy." if documents.cross_validation_passed else "No cross-validation flags."
        )

        doc_section = f"""| Document | Reference Number | Holder / Insured | Validity / Expiry | Verification Status |
| :--- | :--- | :--- | :--- | :--- |
| **RC Book** | `{rc_num}` | {rc_owner} | {rc_date} | {rc_status} |
| **Insurance Policy** | `{pol_num}` | {pol_insured} | {pol_exp} | {pol_status} |
| **Driving Licence** | `{dl_num}` | {dl_name} | {dl_val} | {dl_status} |

*Cross-Validation Audit:* {notes_str}"""
    else:
        doc_section = "*Document Verification:* No digital document credentials submitted for automated cross-validation."

    # Section 3: Damage Breakdown
    dmg_summary = claim_data.damage_summary
    total_damages = dmg_summary.total_damages_count if dmg_summary else 0
    severity_tier = dmg_summary.severity_assessment if dmg_summary else "Unassessed"
    counts_by_type = dmg_summary.damage_counts_by_type if dmg_summary else {}

    if counts_by_type:
        type_lines = "\n".join([f"  - **{k.replace('_', ' ').title()}:** {v} instance{'s' if v > 1 else ''}" for k, v in counts_by_type.items()])
    else:
        type_lines = "  - None detected"

    panels = set()
    for img in claim_data.images:
        for d in img.detections:
            if d.detected_panel:
                panels.add(d.detected_panel.replace("_", " ").title())
    panels_str = ", ".join(sorted(panels)) if panels else "None recorded"

    # Section 5: Knowledge Graph Table
    if internal_inspection_plan:
        kg_rows = []
        for item in internal_inspection_plan:
            prio = item.get("priority", "MEDIUM")
            prio_badge = "🔴 **HIGH**" if prio == "HIGH" else "🟡 **MEDIUM**" if prio == "MEDIUM" else "🟢 **LOW**"
            comp = item.get("component_name", "Component").title()
            lp = item.get("load_path", [])
            lp_str = " -> ".join(lp) if isinstance(lp, list) else str(lp)
            act = item.get("recommended_action", "Physical inspection")
            hrs = item.get("estimated_labor_hours", 0.5)
            kg_rows.append(f"| {prio_badge} | **{comp}** | `{lp_str}` | {act} | {hrs:.1f} hrs |")
        kg_table = "\n".join(kg_rows)
        kg_section = f"""| Priority | Component | Load Path Propagation | Rec. Action | Est. Labor |
| :--- | :--- | :--- | :--- | :--- |
{kg_table}

*Total Estimated Teardown / Borescope Inspection Labor:* **{total_labor_hours:.1f} hrs**"""
    else:
        kg_section = "*Knowledge Graph Assessment:* No internal concealed components flagged for teardown inspection based on external damage trajectory."

    # Section 7: Action Checklist
    action_items_str = "\n".join([f"- [ ] {item}" for item in action_items])

    dossier = f"""# 📋 Intelligent Vehicle Damage Assessment & Survey Dossier
**AI-Assisted Claim Survey Report (Chapter 4.6.5 Compliance)**
*Status: DRAFT — Pending Surveyor Final Sign-Off*

---

### 1. Executive Summary
{exec_summary}

---

### 2. Verified Document & Policy Credentials
{doc_section}

---

### 3. Comprehensive Damage Breakdown (Vision + SAM2)
* **Overall Assessed Severity:** `{severity_tier}`
* **Total Damage Count:** {total_damages} instances
* **Damage Class Distribution:**
{type_lines}
* **Damaged Panels:** {panels_str}

---

### 4. Surface Deformation Analysis (Depth Anything V2)
{surface_narrative}

---

### 5. Probabilistic Internal Component Inspection Plan (Knowledge Graph)
*Concealed safety-critical components flagged via neuro-symbolic force propagation traversal:*

{kg_section}

---

### 6. Explainable Reasoning & Engineering Justification
{explainable_reasoning}

---

### 7. Surveyor Action Directive & Recommendation
* **Surveyor Audit Recommendation:** **{recommendation}**
* **Teardown Inspection Labor Scope:** {total_labor_hours:.1f} hrs (Concealed Component Borescope & Mount Audit)
* **Repair Cost Note:** Detailed parts replacement and labor quotation requires physical surveyor assessment and garage repair estimate (OEM pricing catalog not linked).
* **Surveyor Action Checklist:**
{action_items_str}

---

### 8. Surveyor Sign-Off & Verification
* **Surveyor Review Status:** `PENDING REVIEW`
* **Sign-Off Remarks:** `________________________________________`
* **Date & Signature:** `________________________________________`

> **Notice:** This AI-assisted survey dossier is an assistive decision-support tool (Chapter 4.6.5). The licensed insurance surveyor retains sole and independent authority for final claim adjudication and repair authorization.
"""
    return dossier


def generate_survey_report(
    claim_data: ClaimInspectionResponse,
    documents: Optional[ClaimDocuments] = None,
    allow_mock: bool = False,
) -> LLMSurveyReport:
    """
    Generate an explainable, audit-ready insurance survey report using Groq Text LLM.
    Implements Chapter 4.6.5 with non-fabrication constraints, deterministic Python
    assembly of markdown dossiers, and direct Knowledge Graph risk matrix preservation.
    """
    groq = get_groq_client()
    report_uid = f"REP-{claim_data.claim_id}-{uuid.uuid4().hex[:6].upper()}"

    # 1. Authoritative Internal Inspection Plan & Labor Hours from Knowledge Graph
    internal_inspection_plan = [
        {
            "component_name": r.component_name,
            "priority": r.safety_risk_level,
            "risk_score": round(r.risk_score, 3),
            "recommended_action": r.recommended_action,
            "estimated_labor_hours": r.estimated_labor_hours,
            "source_panel": r.source_panel,
            "impact_zone": r.impact_zone,
            "load_path": r.load_path if isinstance(r.load_path, list) else [str(r.load_path)],
            "rationale": r.rationale,
        }
        for r in (claim_data.structural_risk_matrix or [])
    ]
    total_labor_hours = sum(r.get("estimated_labor_hours", 0.0) for r in internal_inspection_plan)

    # 2. Check Mock Mode vs Real Execution
    if groq.is_mock_mode():
        if not allow_mock:
            raise ReportGenerationError(
                "Groq API key not configured or mock mode active, but allow_mock=False. "
                "Cannot synthesize live LLM report without Groq service."
            )
        # Explicit mock mode requested (e.g. offline testing)
        parsed_narrative = _generate_dynamic_mock_narrative(claim_data, documents)
        model_name = "deterministic-mock-llama-3.3-70b"
        latency_ms = 12.0
    else:
        # Prepare summarized input payload (NO raw image/mask tensors!)
        compact_payload = _prepare_compact_pipeline_payload(claim_data, documents)
        payload_json_str = json.dumps(compact_payload, indent=2)

        messages = [
            {"role": "system", "content": REPORT_GENERATOR_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Synthesize the official insurance claim survey report narrative for this inspected vehicle.\n\nInput Pipeline Data:\n{payload_json_str}",
            },
        ]

        try:
            raw_text, latency_ms = groq.text_chat_completion(
                messages=messages,
                temperature=0.25,
                response_format={"type": "json_object"},
            )
            parsed_narrative = _strip_json(raw_text)

            # Retry once if initial JSON parse failed
            if parsed_narrative is None:
                logger.warning("[!] Report JSON parse failed. Retrying with stricter JSON instruction...")
                stricter_messages = [
                    {"role": "system", "content": REPORT_GENERATOR_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Input Pipeline Data:\n{payload_json_str}"},
                    {"role": "assistant", "content": raw_text},
                    {
                        "role": "user",
                        "content": "Your previous response was not valid JSON. Please return ONLY a strictly valid JSON object matching the requested schema.",
                    },
                ]
                raw_text, retry_lat = groq.text_chat_completion(
                    messages=stricter_messages,
                    temperature=0.1,
                    response_format={"type": "json_object"},
                )
                latency_ms += retry_lat
                parsed_narrative = _strip_json(raw_text)

            if parsed_narrative is None:
                if not allow_mock:
                    raise ReportGenerationError(
                        f"Failed to parse valid JSON report response from Groq after retry. "
                        f"Raw snippet: {raw_text[:200] if raw_text else 'Empty response'}"
                    )
                logger.warning("[!] Fallback to dynamic mock narrative due to JSON parse error in mock mode.")
                parsed_narrative = _generate_dynamic_mock_narrative(claim_data, documents)

            model_name = groq.text_model

        except GroqClientError as e:
            if not allow_mock:
                raise ReportGenerationError(f"Groq Report Generation error: {e}")
            logger.warning(f"[!] Groq error in mock mode: {e}. Generating dynamic mock narrative.")
            parsed_narrative = _generate_dynamic_mock_narrative(claim_data, documents)
            model_name = "deterministic-fallback"
            latency_ms = 0.0

    # 3. Post-Process Narrative & Normalize Recommendation
    recommendation = parsed_narrative.get("surveyor_recommendation") or parsed_narrative.get("claim_disposition")
    if not recommendation or recommendation in ("Approve", "Reject"):
        has_high_risk = any(r.get("priority") == "HIGH" for r in internal_inspection_plan)
        total_dmg = claim_data.damage_summary.total_damages_count if claim_data.damage_summary else 0
        recommendation = (
            "Detailed Teardown Audit Required"
            if has_high_risk or total_dmg >= 3
            else "Physical Verification Recommended" if total_dmg > 0
            else "Cosmetic Survey Verification"
        )

    action_items = parsed_narrative.get("surveyor_action_items")
    if not isinstance(action_items, list) or not action_items:
        action_items = [
            "Physically inspect flagged high-priority internal components during teardown.",
            "Verify panel alignment and mounting tolerances against OEM specifications.",
        ]

    # 4. Build Deterministic Publication-Grade Markdown Dossier
    markdown_dossier = _build_markdown_dossier(
        claim_data=claim_data,
        documents=documents,
        narrative=parsed_narrative,
        internal_inspection_plan=internal_inspection_plan,
        total_labor_hours=total_labor_hours,
    )

    # 5. Return Clean Pydantic Model
    return LLMSurveyReport(
        report_id=report_uid,
        claim_id=claim_data.claim_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        model_used=model_name,
        latency_ms=latency_ms,
        executive_summary=parsed_narrative.get("executive_summary", "Automated survey synthesis completed."),
        damage_summary=parsed_narrative.get("damage_summary") or (
            claim_data.damage_summary.model_dump() if claim_data.damage_summary else {}
        ),
        surface_deformation_narrative=parsed_narrative.get("surface_deformation_narrative", ""),
        internal_inspection_plan=internal_inspection_plan,
        explainable_reasoning=parsed_narrative.get("explainable_reasoning", ""),
        surveyor_recommendation=recommendation,
        claim_disposition=recommendation,  # Backwards compatibility
        total_estimated_labor_hours=round(total_labor_hours, 1) if total_labor_hours > 0 else None,
        estimated_repair_cost_min=None,
        estimated_repair_cost_max=None,
        surveyor_action_items=action_items,
        markdown_dossier=markdown_dossier,
        status="DRAFT",
    )


# Backward compatibility alias
generate_claim_report = generate_survey_report
