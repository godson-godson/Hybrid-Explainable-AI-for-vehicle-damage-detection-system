"""
Vehicle Structural Knowledge Graph & Explainable Reasoning Engine.
Interfaces with Neo4j (with built-in Embedded Graph Store fallback) to model
automotive collision energy load paths and infer hidden structural risks.

GENERAL AUTOMOTIVE COLLISION INDUSTRY REFERENCES INFORMING THIS GRAPH:
1. I-CAR (Inter-Industry Conference on Auto Collision Repair):
   - "Collision Energy Management & Frontal Unibody Load Paths"
   - "Bumper System Impact Absorption and Sensor/Cooling Package Protection"
2. NHTSA & IIHS (Insurance Institute for Highway Safety):
   - "Frontal Crash Test Structural Performance & Energy Dissipation Protocols"
3. Thatcham Research (UK Motor Insurance Repair Research Centre):
   - "Structural Damage Assessment: Crash Box & Rail Force Propagation Rules"

NOTE ON METRICS & HEURISTICS:
- Edge weights (0.0 to 1.0) represent approximate, heuristically-assigned force
  propagation likelihoods along structural joints, not measured strain gauges.
- Risk scores are computed via multi-hop traversal with exponential distance decay
  and damage-severity scaling.
"""

from collections import deque
import logging
import os
from typing import Any, Dict, List, Optional, Set, Tuple

from ..models.schemas import (
    BoundingBox,
    DamageDetection,
    DamageSummary,
    InspectionRecommendation,
)

logger = logging.getLogger("knowledge_graph")

# ==============================================================================
# Step 0: Panel Localization Heuristic (3x3 Normalized Vehicle ROI Grid)
# ==============================================================================
# The damage detector outputs class type (dent, scratch, etc.) but not the panel.
# We localize damages by projecting the damage center into a normalized 3x3 grid
# of the vehicle ROI bounding box (top/mid/bottom x left/center/right), conditioned
# on the surveyor's viewpoint perspective tag.
#
# CAMERA-FACING CONVENTION:
# For "front" view: left = viewer's left (vehicle right-hand side), right = viewer's right.
# For "rear" view:  left = viewer's left (vehicle left-hand side), right = viewer's right.
# ==============================================================================

PANEL_GRID_LOOKUP: Dict[str, Dict[Tuple[str, str], str]] = {
    "front": {
        ("top", "left"): "left front fender",
        ("top", "center"): "hood",
        ("top", "right"): "right front fender",
        ("mid", "left"): "left headlight assembly",
        ("mid", "center"): "front grille",
        ("mid", "right"): "right headlight assembly",
        ("bottom", "left"): "front bumper cover",
        ("bottom", "center"): "front bumper cover",
        ("bottom", "right"): "front bumper cover",
    },
    "rear": {
        ("top", "left"): "rear windshield",
        ("top", "center"): "rear windshield",
        ("top", "right"): "rear windshield",
        ("mid", "left"): "left taillight assembly",
        ("mid", "center"): "trunk lid",
        ("mid", "right"): "right taillight assembly",
        ("bottom", "left"): "rear bumper cover",
        ("bottom", "center"): "rear bumper cover",
        ("bottom", "right"): "rear bumper cover",
    },
    "left": {
        ("top", "left"): "left front fender",
        ("top", "center"): "left front door",
        ("top", "right"): "left rear door",
        ("mid", "left"): "left front fender",
        ("mid", "center"): "left front door",
        ("mid", "right"): "left rear door",
        ("bottom", "left"): "left front fender",
        ("bottom", "center"): "left rocker panel",
        ("bottom", "right"): "left rocker panel",
    },
    "right": {
        ("top", "left"): "right rear door",
        ("top", "center"): "right front door",
        ("top", "right"): "right front fender",
        ("mid", "left"): "right rear door",
        ("mid", "center"): "right front door",
        ("mid", "right"): "right front fender",
        ("bottom", "left"): "right rocker panel",
        ("bottom", "center"): "right rocker panel",
        ("bottom", "right"): "right front fender",
    },
}


def localize_damage_panel_details(
    damage_bbox: BoundingBox,
    vehicle_roi_bbox: Optional[BoundingBox] = None,
    view_angle: Optional[str] = "front",
) -> Tuple[str, Optional[str], float, str]:
    """
    Detailed panel localization with geometric aspect-ratio disambiguation.

    Returns:
        (canonical_panel, fallback_panel, confidence, grid_cell)
    """
    view = (view_angle or "front").strip().lower()
    if view not in PANEL_GRID_LOOKUP:
        view = "front"

    # Compute damage center point and dimensions
    dc_x = (damage_bbox.x1 + damage_bbox.x2) / 2.0
    dc_y = (damage_bbox.y1 + damage_bbox.y2) / 2.0
    db_w = max(float(damage_bbox.width), 1.0)
    db_h = max(float(damage_bbox.height), 1.0)
    aspect_ratio = db_w / db_h

    if vehicle_roi_bbox and vehicle_roi_bbox.width > 0 and vehicle_roi_bbox.height > 0:
        vx1, vy1 = vehicle_roi_bbox.x1, vehicle_roi_bbox.y1
        vw, vh = float(vehicle_roi_bbox.width), float(vehicle_roi_bbox.height)
    else:
        vx1, vy1 = 0.0, 0.0
        vw = max(float(damage_bbox.x2), 100.0)
        vh = max(float(damage_bbox.y2), 100.0)

    # Normalized relative coordinate in [0.0, 1.0]
    norm_x = min(max((dc_x - vx1) / vw, 0.0), 1.0)
    norm_y = min(max((dc_y - vy1) / vh, 0.0), 1.0)
    rel_w = db_w / vw
    rel_h = db_h / vh

    # Determine vertical grid partition (top / mid / bottom)
    if norm_y < 0.35:
        row = "top"
    elif norm_y < 0.65:
        row = "mid"
    else:
        row = "bottom"

    # Determine horizontal grid partition (left / center / right)
    if norm_x < 0.35:
        col = "left"
    elif norm_x < 0.65:
        col = "center"
    else:
        col = "right"

    grid_cell = f"{row}-{col}"
    grid_panel = PANEL_GRID_LOOKUP[view].get((row, col), "front bumper cover")
    confidence = 0.75
    fallback_panel: Optional[str] = None

    # --------------------------------------------------------------------------
    # Refinement 2: Geometric & Aspect-Ratio Disambiguation
    # --------------------------------------------------------------------------
    if view == "front":
        # Rule A: Wide horizontal span at lower half -> prioritize bumper cover
        if norm_y >= 0.50 and (rel_w > 0.40 or aspect_ratio >= 2.2):
            fallback_panel = grid_panel
            grid_panel = "front bumper cover"
            confidence = 0.90
        # Rule B: Vertical elongation along outer boundary -> prioritize front fender
        elif (norm_x < 0.28 or norm_x > 0.72) and (rel_h > 0.22 or aspect_ratio <= 0.75):
            fallback_panel = grid_panel
            grid_panel = "left front fender" if norm_x < 0.50 else "right front fender"
            confidence = 0.85
        # Rule C: Upper center horizontal area -> Hood
        elif norm_y < 0.45 and 0.25 <= norm_x <= 0.75:
            fallback_panel = grid_panel
            grid_panel = "hood"
            confidence = 0.88
    elif view == "rear":
        # Rule A: Wide lower horizontal span -> rear bumper cover
        if norm_y >= 0.55 and (rel_w > 0.40 or aspect_ratio >= 2.0):
            fallback_panel = grid_panel
            grid_panel = "rear bumper cover"
            confidence = 0.90
        # Rule B: Upper half glass area
        elif norm_y < 0.40:
            fallback_panel = grid_panel
            grid_panel = "rear windshield"
            confidence = 0.85

    return grid_panel, fallback_panel, confidence, grid_cell


def localize_damage_panel(
    damage_bbox: BoundingBox,
    vehicle_roi_bbox: Optional[BoundingBox] = None,
    view_angle: Optional[str] = "front",
) -> str:
    """
    Map a damage bounding box to the most plausible external vehicle panel name
    using a 3x3 normalized spatial grid over the vehicle ROI crop with aspect-ratio disambiguation.

    Args:
        damage_bbox: Bounding box of the detected damage instance.
        vehicle_roi_bbox: Detected vehicle ROI bounding box (from YOLO11n).
        view_angle: Surveyor view tag ('front', 'rear', 'left', 'right', 'close_up').

    Returns:
        Canonical external panel name (e.g. 'front bumper cover', 'hood').
    """
    panel, _, _, _ = localize_damage_panel_details(damage_bbox, vehicle_roi_bbox, view_angle)
    return panel


# ==============================================================================
# Step 1 & 2: Graph Schema, Seed Data & In-Memory Store
# ==============================================================================

SEED_NODES = [
    # Front Zone External Panels
    {"type": "ExternalPanel", "name": "front bumper cover", "zone": "front"},
    {"type": "ExternalPanel", "name": "hood", "zone": "front"},
    {"type": "ExternalPanel", "name": "front grille", "zone": "front"},
    {"type": "ExternalPanel", "name": "left headlight assembly", "zone": "front"},
    {"type": "ExternalPanel", "name": "right headlight assembly", "zone": "front"},
    {"type": "ExternalPanel", "name": "left front fender", "zone": "front"},
    {"type": "ExternalPanel", "name": "right front fender", "zone": "front"},
    # Front Zone Concealed Internal Components
    {"type": "InternalComponent", "name": "bumper reinforcement bar", "zone": "front", "criticality": "high"},
    {"type": "InternalComponent", "name": "front crush cans", "zone": "front", "criticality": "high"},
    {"type": "InternalComponent", "name": "radiator support assembly", "zone": "front", "criticality": "high"},
    {"type": "InternalComponent", "name": "radiator", "zone": "front", "criticality": "medium"},
    {"type": "InternalComponent", "name": "air conditioning condenser", "zone": "front", "criticality": "medium"},
    {"type": "InternalComponent", "name": "cooling fan assembly", "zone": "front", "criticality": "medium"},
    {"type": "InternalComponent", "name": "front subframe crossmember", "zone": "front", "criticality": "high"},
    {"type": "InternalComponent", "name": "hood latch and release cable", "zone": "front", "criticality": "high"},
    {"type": "InternalComponent", "name": "hood hinge assembly", "zone": "front", "criticality": "medium"},
    {"type": "InternalComponent", "name": "engine bay wiring harness", "zone": "front", "criticality": "high"},
    {"type": "InternalComponent", "name": "windshield washer fluid reservoir", "zone": "front", "criticality": "low"},
    {"type": "InternalComponent", "name": "front longitudinal frame rails", "zone": "front", "criticality": "high"},
    # Rear Zone External Panels
    {"type": "ExternalPanel", "name": "rear bumper cover", "zone": "rear"},
    {"type": "ExternalPanel", "name": "trunk lid", "zone": "rear"},
    {"type": "ExternalPanel", "name": "rear windshield", "zone": "rear"},
    {"type": "ExternalPanel", "name": "left taillight assembly", "zone": "rear"},
    {"type": "ExternalPanel", "name": "right taillight assembly", "zone": "rear"},
    # Rear Zone Internal Components (Placeholder)
    {"type": "InternalComponent", "name": "rear impact reinforcement bar", "zone": "rear", "criticality": "high"},
    {"type": "InternalComponent", "name": "rear body panel sheetmetal", "zone": "rear", "criticality": "medium"},
    {"type": "InternalComponent", "name": "trunk floor pan and spare tire well", "zone": "rear", "criticality": "high"},
    {"type": "InternalComponent", "name": "rear longitudinal frame rails", "zone": "rear", "criticality": "high"},
    {"type": "InternalComponent", "name": "trunk latch and power closer", "zone": "rear", "criticality": "medium"},
    {"type": "InternalComponent", "name": "rear wiper motor and defroster harness", "zone": "rear", "criticality": "low"},
    # Left / Right Side Panels & Components (Placeholder)
    {"type": "ExternalPanel", "name": "left front door", "zone": "left"},
    {"type": "ExternalPanel", "name": "left rear door", "zone": "left"},
    {"type": "ExternalPanel", "name": "left rocker panel", "zone": "left"},
    {"type": "InternalComponent", "name": "left front door anti-intrusion beam", "zone": "left", "criticality": "high"},
    {"type": "InternalComponent", "name": "left center B-pillar structural post", "zone": "left", "criticality": "high"},
    {"type": "InternalComponent", "name": "left curtain airbag deployment sensors", "zone": "left", "criticality": "high"},
    {"type": "ExternalPanel", "name": "right front door", "zone": "right"},
    {"type": "ExternalPanel", "name": "right rear door", "zone": "right"},
    {"type": "ExternalPanel", "name": "right rocker panel", "zone": "right"},
    {"type": "InternalComponent", "name": "right front door anti-intrusion beam", "zone": "right", "criticality": "high"},
    {"type": "InternalComponent", "name": "right center B-pillar structural post", "zone": "right", "criticality": "high"},
    {"type": "InternalComponent", "name": "right curtain airbag deployment sensors", "zone": "right", "criticality": "high"},
]

SEED_RELATIONSHIPS = [
    # Front Zone Load Paths
    {"src": "front bumper cover", "rel": "PROTECTS", "dst": "bumper reinforcement bar", "weight": 0.85},
    {"src": "front bumper cover", "rel": "TRANSMITS_FORCE_TO", "dst": "bumper reinforcement bar", "weight": 0.90},
    {"src": "front bumper cover", "rel": "ATTACHED_TO", "dst": "radiator support assembly", "weight": 0.60},
    {"src": "front grille", "rel": "PROTECTS", "dst": "air conditioning condenser", "weight": 0.70},
    {"src": "front grille", "rel": "TRANSMITS_FORCE_TO", "dst": "radiator support assembly", "weight": 0.75},
    {"src": "bumper reinforcement bar", "rel": "TRANSMITS_FORCE_TO", "dst": "front crush cans", "weight": 0.95},
    {"src": "bumper reinforcement bar", "rel": "ATTACHED_TO", "dst": "radiator support assembly", "weight": 0.80},
    {"src": "front crush cans", "rel": "TRANSMITS_FORCE_TO", "dst": "front longitudinal frame rails", "weight": 0.88},
    {"src": "front crush cans", "rel": "TRANSMITS_FORCE_TO", "dst": "front subframe crossmember", "weight": 0.75},
    {"src": "radiator support assembly", "rel": "TRANSMITS_FORCE_TO", "dst": "air conditioning condenser", "weight": 0.80},
    {"src": "radiator support assembly", "rel": "TRANSMITS_FORCE_TO", "dst": "radiator", "weight": 0.75},
    {"src": "air conditioning condenser", "rel": "TRANSMITS_FORCE_TO", "dst": "radiator", "weight": 0.70},
    {"src": "radiator", "rel": "TRANSMITS_FORCE_TO", "dst": "cooling fan assembly", "weight": 0.65},
    {"src": "hood", "rel": "ATTACHED_TO", "dst": "hood latch and release cable", "weight": 0.85},
    {"src": "hood", "rel": "ATTACHED_TO", "dst": "hood hinge assembly", "weight": 0.75},
    {"src": "hood", "rel": "TRANSMITS_FORCE_TO", "dst": "radiator support assembly", "weight": 0.60},
    {"src": "hood latch and release cable", "rel": "ATTACHED_TO", "dst": "radiator support assembly", "weight": 0.70},
    {"src": "left headlight assembly", "rel": "ATTACHED_TO", "dst": "radiator support assembly", "weight": 0.70},
    {"src": "left headlight assembly", "rel": "TRANSMITS_FORCE_TO", "dst": "engine bay wiring harness", "weight": 0.55},
    {"src": "left headlight assembly", "rel": "ADJACENT_TO", "dst": "left front fender", "weight": 0.50},
    {"src": "right headlight assembly", "rel": "ATTACHED_TO", "dst": "radiator support assembly", "weight": 0.70},
    {"src": "right headlight assembly", "rel": "TRANSMITS_FORCE_TO", "dst": "engine bay wiring harness", "weight": 0.55},
    {"src": "right headlight assembly", "rel": "ADJACENT_TO", "dst": "right front fender", "weight": 0.50},
    {"src": "left front fender", "rel": "PROTECTS", "dst": "windshield washer fluid reservoir", "weight": 0.65},
    {"src": "left front fender", "rel": "TRANSMITS_FORCE_TO", "dst": "front longitudinal frame rails", "weight": 0.50},
    {"src": "right front fender", "rel": "TRANSMITS_FORCE_TO", "dst": "front longitudinal frame rails", "weight": 0.50},
    # Rear Zone Load Paths
    {"src": "rear bumper cover", "rel": "PROTECTS", "dst": "rear impact reinforcement bar", "weight": 0.85},
    {"src": "rear bumper cover", "rel": "TRANSMITS_FORCE_TO", "dst": "rear impact reinforcement bar", "weight": 0.90},
    {"src": "rear impact reinforcement bar", "rel": "TRANSMITS_FORCE_TO", "dst": "rear body panel sheetmetal", "weight": 0.80},
    {"src": "rear impact reinforcement bar", "rel": "TRANSMITS_FORCE_TO", "dst": "rear longitudinal frame rails", "weight": 0.75},
    {"src": "rear body panel sheetmetal", "rel": "TRANSMITS_FORCE_TO", "dst": "trunk floor pan and spare tire well", "weight": 0.70},
    {"src": "trunk lid", "rel": "ATTACHED_TO", "dst": "trunk latch and power closer", "weight": 0.80},
    {"src": "rear windshield", "rel": "PROTECTS", "dst": "rear wiper motor and defroster harness", "weight": 0.60},
    # Left / Right Side Load Paths
    {"src": "left front door", "rel": "PROTECTS", "dst": "left front door anti-intrusion beam", "weight": 0.90},
    {"src": "left front door", "rel": "TRANSMITS_FORCE_TO", "dst": "left center B-pillar structural post", "weight": 0.85},
    {"src": "left rear door", "rel": "TRANSMITS_FORCE_TO", "dst": "left center B-pillar structural post", "weight": 0.80},
    {"src": "left center B-pillar structural post", "rel": "TRANSMITS_FORCE_TO", "dst": "left curtain airbag deployment sensors", "weight": 0.75},
    {"src": "left rocker panel", "rel": "ATTACHED_TO", "dst": "left center B-pillar structural post", "weight": 0.80},
    {"src": "right front door", "rel": "PROTECTS", "dst": "right front door anti-intrusion beam", "weight": 0.90},
    {"src": "right front door", "rel": "TRANSMITS_FORCE_TO", "dst": "right center B-pillar structural post", "weight": 0.85},
    {"src": "right rear door", "rel": "TRANSMITS_FORCE_TO", "dst": "right center B-pillar structural post", "weight": 0.80},
    {"src": "right center B-pillar structural post", "rel": "TRANSMITS_FORCE_TO", "dst": "right curtain airbag deployment sensors", "weight": 0.75},
    {"src": "right rocker panel", "rel": "ATTACHED_TO", "dst": "right center B-pillar structural post", "weight": 0.80},
]


class EmbeddedGraphStore:
    """
    In-Memory Vehicle Structural Graph with exact Neo4j Cypher schema, nodes,
    weighted directed relationships, and multi-hop traversal reasoning.
    Used for instant unit testing and zero-downtime fallback when a standalone
    Neo4j daemon is not active.
    """

    def __init__(self):
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.adjacency: Dict[str, List[Dict[str, Any]]] = {}
        self.seed_count = 0
        self._load_seed_data()

    def _load_seed_data(self):
        """Idempotent seed data loading."""
        for n in SEED_NODES:
            self.nodes[n["name"]] = n
            if n["name"] not in self.adjacency:
                self.adjacency[n["name"]] = []

        for r in SEED_RELATIONSHIPS:
            src, dst = r["src"], r["dst"]
            if src not in self.adjacency:
                self.adjacency[src] = []
            # Avoid duplicate edges on re-runs (MERGE behavior)
            existing = [
                edge for edge in self.adjacency[src]
                if edge["dst"] == dst and edge["rel"] == r["rel"]
            ]
            if not existing:
                self.adjacency[src].append({
                    "dst": dst,
                    "rel": r["rel"],
                    "weight": float(r["weight"]),
                })
        self.seed_count = len(self.nodes)
        logger.info(f"[+] Embedded Knowledge Graph loaded with {len(self.nodes)} nodes and {len(SEED_RELATIONSHIPS)} edges.")

    @property
    def relationships(self) -> List[Dict[str, Any]]:
        rels = []
        for src, edges in self.adjacency.items():
            for e in edges:
                rels.append({"src": src, **e})
        return rels

    def seed_schema(self):
        self._load_seed_data()

    def find_node(self, name: str) -> Optional[Dict[str, Any]]:
        clean = (name or "").strip().lower()
        if clean in self.nodes:
            return self.nodes[clean]
        clean_space = clean.replace("_", " ")
        if clean_space in self.nodes:
            return self.nodes[clean_space]
        clean_underscore = clean.replace(" ", "_")
        return self.nodes.get(clean_underscore)

    def traverse_load_paths(
        self,
        start_panel: str,
        max_depth: int = 3,
        allowed_rels: Optional[Set[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Traverse outgoing structural load paths from a starting panel up to max_depth hops.

        Returns:
            List of path records: {dst_name, node_info, path_nodes, path_weights, hop_count}
        """
        start = (start_panel or "").strip().lower()
        if start not in self.nodes:
            start_space = start.replace("_", " ")
            if start_space in self.nodes:
                start = start_space
            else:
                return []

        if allowed_rels is None:
            allowed_rels = {"TRANSMITS_FORCE_TO", "PROTECTS", "ATTACHED_TO"}

        # BFS queue elements: (current_node, path_nodes, path_weights, depth)
        queue = deque([(start, [start], [], 0)])
        visited_paths: List[Dict[str, Any]] = []

        while queue:
            curr, path, weights, depth = queue.popleft()

            if depth >= max_depth:
                continue

            for edge in self.adjacency.get(curr, []):
                if edge["rel"] not in allowed_rels:
                    continue

                nxt = edge["dst"]
                # Prevent cycles along the current branch
                if nxt in path:
                    continue

                new_path = path + [nxt]
                new_weights = weights + [edge["weight"]]
                new_depth = depth + 1

                nxt_node = self.nodes.get(nxt, {})
                # We are interested in concealed internal components for physical inspection
                if nxt_node.get("type") == "InternalComponent":
                    visited_paths.append({
                        "dst_name": nxt,
                        "node_info": nxt_node,
                        "path_nodes": new_path,
                        "path_weights": new_weights,
                        "hop_count": new_depth,
                    })

                queue.append((nxt, new_path, new_weights, new_depth))

        return visited_paths


# ==============================================================================
# Step 3: Vehicle Structural Knowledge Graph Client & Reasoning Engine
# ==============================================================================

class VehicleStructuralKnowledgeGraph:
    """
    Manages Neo4j connectivity and executes explainable structural reasoning.
    Gracefully falls back to EmbeddedGraphStore if Neo4j is offline.
    """

    def __init__(self):
        self.uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = os.getenv("NEO4J_USER", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD", "password")
        self.database = os.getenv("NEO4J_DATABASE", "neo4j")

        self.driver = None
        self.is_neo4j_connected = False
        self.embedded_store = EmbeddedGraphStore()

        self._try_connect_neo4j()

    def _try_connect_neo4j(self):
        """Attempt connection to external Neo4j instance if available."""
        try:
            from neo4j import GraphDatabase
            driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            # Test connectivity with ping
            driver.verify_connectivity()
            self.driver = driver
            self.is_neo4j_connected = True
            logger.info(f"[+] Connected to Neo4j database at {self.uri}")
            self._seed_neo4j_graph()
        except Exception as e:
            self.driver = None
            self.is_neo4j_connected = False
            logger.info(
                f"[*] Neo4j daemon not reachable at {self.uri} ({type(e).__name__}). "
                f"Active mode: High-Performance Embedded Structural Knowledge Graph."
            )

    def _seed_neo4j_graph(self):
        """Seed external Neo4j instance idempotently using MERGE."""
        if not self.is_neo4j_connected or not self.driver:
            return
        try:
            cypher_file = os.path.join(os.path.dirname(__file__), "seed_graph.cypher")
            if os.path.exists(cypher_file):
                with open(cypher_file, "r", encoding="utf-8") as f:
                    content = f.read()
                # Split statements by semicolon
                statements = [s.strip() for s in content.split(";") if s.strip() and not s.strip().startswith("//")]
                with self.driver.session(database=self.database) as session:
                    for stmt in statements:
                        session.run(stmt)
                logger.info("[+] Neo4j seed graph verified idempotently via Cypher.")
        except Exception as err:
            logger.warning(f"[!] Could not seed Neo4j graph: {err}")

    @property
    def nodes(self):
        return self.embedded_store.nodes

    @property
    def relationships(self):
        return self.embedded_store.relationships

    def seed_schema(self):
        if self.is_neo4j_connected:
            self._seed_neo4j_graph()
        self.embedded_store.seed_schema()

    def recommend_inspections(
        self,
        panel_name: str,
        deformation_score: Optional[float] = None,
        damaged_area_pct: Optional[float] = None,
        damage_type: Optional[str] = "dent",
        max_depth: int = 3,
        decay_factor: float = 0.75,
        max_hops: Optional[int] = None,
        **kwargs,
    ) -> List[InspectionRecommendation]:
        """
        Traverse outward from an external damaged panel to recommend concealed
        internal components requiring physical inspection.

        HEURISTIC RISK SCORE FORMULATION:
        1. Accumulated edge weights with hop decay (further = less force transmitted):
           T(Path) = (Product of edge weights along path) * (decay_factor ^ (hops - 1))
        2. Damage severity scaling factor:
           - Deformation score delta (Phase 3 relative indentation):
             Positive delta (dent/recessed) increases risk by (1.0 + 3.0 * delta).
             Planar scratch (delta ~ 0.0) scales down to ~0.65.
           - Damaged area percentage (Phase 2 SAM2 mask):
             Higher area ratio increases load dissipation into backing structures.
        3. Component Risk = min(1.0, T(Path) * Severity_Factor)

        EXPLAINABLE REASONING LOG:
        Generates a transparent narrative showing the exact load propagation
        sequence (e.g. 'front bumper cover -> bumper reinforcement bar -> radiator support').
        """
        clean_panel = (panel_name or "").strip().lower()
        effective_depth = max_hops if max_hops is not None else max_depth

        # Step 3 requirement: Look up ExternalPanel in graph. If no match, return empty with reason.
        panel_node = self.embedded_store.find_node(clean_panel)
        if not panel_node:
            logger.info(f"[*] No structural mapping available for panel '{panel_name}'. Skipping KG reasoning.")
            return []

        # Multi-hop load path traversal
        paths = self.embedded_store.traverse_load_paths(
            start_panel=clean_panel,
            max_depth=effective_depth,
            allowed_rels={"TRANSMITS_FORCE_TO", "PROTECTS", "ATTACHED_TO"},
        )

        if not paths:
            return []

        # ----------------------------------------------------------------------
        # Refinement 1: Exact Heuristic Risk Formula
        # R(c) = min(1.0, S_damage * max_{p in P} (Prod_{e in p} w(e) * lambda^{hop(e)}))
        # S_damage = alpha * norm_deform + beta * norm_area normalized to [0.1, 1.0]
        # ----------------------------------------------------------------------
        deform = deformation_score if deformation_score is not None else 0.0
        area_pct = damaged_area_pct if damaged_area_pct is not None else 1.0
        dtype = (damage_type or "dent").lower()

        alpha = 0.65
        beta = 0.35

        # Normalize deformation score: recessed deformation (>0) increases severity
        norm_deform = min(1.0, max(0.0, (deform + 0.02) / 0.35))
        # Normalize damaged area percentage (SAM2 mask ratio)
        norm_area = min(1.0, max(0.0, area_pct / 10.0))

        s_damage = min(1.0, max(0.10, alpha * norm_deform + beta * norm_area))

        if deform > 0.02:
            deform_desc = f"recessed indentation (relative deformation: {deform:+.3f})"
        elif deform < -0.02:
            deform_desc = f"protruding buckle (relative deformation: {deform:+.3f})"
        else:
            deform_desc = "planar surface abrasion"

        # ----------------------------------------------------------------------
        # Traverse paths, compute attenuation Prod(w_e) * lambda^{hops - 1}
        # and take max_{p in P}
        # ----------------------------------------------------------------------
        best_by_component: Dict[str, Dict[str, Any]] = {}

        for p in paths:
            comp_name = p["dst_name"]
            weights = p["path_weights"]
            hops = p["hop_count"]

            # Edge weight product
            prod_weight = 1.0
            for w in weights:
                prod_weight *= w

            # Distance decay attenuation factor lambda = 0.75
            attenuation = prod_weight * (decay_factor ** (hops - 1))

            # Enforce exact formula: R(c) = min(1.0, S_damage * attenuation)
            risk_score = min(1.0, round(s_damage * attenuation, 3))

            if comp_name not in best_by_component or risk_score > best_by_component[comp_name]["risk_score"]:
                best_by_component[comp_name] = {
                    "component_name": comp_name,
                    "node_info": p["node_info"],
                    "path_nodes": p["path_nodes"],
                    "risk_score": risk_score,
                    "hops": hops,
                    "attenuation": attenuation,
                }

        # ----------------------------------------------------------------------
        # Build InspectionRecommendation Schemas with Explainable Rationale
        # ----------------------------------------------------------------------
        recommendations: List[InspectionRecommendation] = []

        for comp_name, item in best_by_component.items():
            r_score = item["risk_score"]
            node_info = item["node_info"]
            criticality = node_info.get("criticality", "medium")
            path_str = " -> ".join(item["path_nodes"])

            # Risk tier assignment
            if r_score >= 0.60:
                tier = "HIGH"
                action = "Physical Teardown & Dimensional Alignment Check"
                labor_hrs = 2.5 if criticality == "high" else 1.8
            elif r_score >= 0.35:
                tier = "MEDIUM"
                action = "Visual Borescope & Mount Integrity Inspection"
                labor_hrs = 1.2 if criticality == "high" else 0.8
            else:
                tier = "LOW"
                action = "Secondary Check during Reassembly"
                labor_hrs = 0.5

            # Explainable reasoning log narrative
            rationale = (
                f"{path_str}: {dtype.upper()} damage with {deform_desc} observed on '{clean_panel}'. "
                f"Collision force propagates through {item['hops']}-hop structural load path to '{comp_name}' "
                f"(Concealed Risk: {r_score * 100:.1f}%, Tier: {tier})."
            )

            rec = InspectionRecommendation(
                component_name=comp_name,
                source_panel=clean_panel,
                impact_zone=node_info.get("zone", panel_node.get("zone", "front")),
                risk_score=r_score,
                safety_risk_level=tier,
                recommended_action=action,
                estimated_labor_hours=labor_hrs,
                load_path=item["path_nodes"],
                rationale=rationale,
                detected_damage=dtype,
            )
            recommendations.append(rec)

        # Sort recommendations by risk score descending
        recommendations.sort(key=lambda x: x.risk_score, reverse=True)
        return recommendations

    def aggregate_claim_structural_risks(
        self,
        classified_detections: List[DamageDetection],
    ) -> List[InspectionRecommendation]:
        """
        Aggregate structural inspection recommendations across all damage instances
        in a claim using Noisy-OR probability aggregation:
            R_final(c) = 1 - Prod_i (1 - R_i(c))
        Compiles all contributing load paths into an explainable multi-vector log.
        """
        comp_sources: Dict[str, List[Tuple[str, float, str, List[str]]]] = {}
        comp_base_rec: Dict[str, InspectionRecommendation] = {}

        for det in classified_detections:
            if not det.classified or det.damage_type == "unclassified":
                continue

            # Auto-populate instance recommendations if not already computed
            if not det.recommendations and det.detected_panel:
                deform_val = det.deformation.relative_deformation_score if det.deformation else 0.5
                area_val = det.segmentation.area_percentage if det.segmentation else 1.0
                det.recommendations = self.recommend_inspections(
                    panel_name=det.detected_panel,
                    deformation_score=deform_val,
                    damaged_area_pct=area_val,
                    damage_type=det.damage_type,
                )

            panel_str = det.detected_panel or "unspecified panel"
            for rec in det.recommendations:
                c_name = rec.component_name
                if c_name not in comp_sources:
                    comp_sources[c_name] = []
                    comp_base_rec[c_name] = rec
                comp_sources[c_name].append((panel_str, rec.risk_score, det.damage_type, rec.load_path))

        aggregated_matrix: List[InspectionRecommendation] = []

        for c_name, sources in comp_sources.items():
            base_rec = comp_base_rec[c_name]

            # Refinement 3: Noisy-OR probability aggregation
            prod_complement = 1.0
            for _, r_score, _, _ in sources:
                prod_complement *= (1.0 - r_score)

            noisy_or_risk = round(min(1.0, max(0.0, 1.0 - prod_complement)), 3)

            # Assign risk tier based on combined cumulative risk
            if noisy_or_risk >= 0.60:
                tier = "HIGH"
                action = "Mandatory Physical Teardown & Frame Alignment Check"
            elif noisy_or_risk >= 0.35:
                tier = "MEDIUM"
                action = "Visual Borescope & Mount Tolerance Inspection"
            else:
                tier = "LOW"
                action = "Secondary Check during Reassembly"

            # Combine all distinct paths and panels into multi-vector rationale
            panel_summaries = [f"{p} ({dtype.upper()}: {r:.2f})" for p, r, dtype, _ in sources]
            unique_paths = []
            for _, _, _, path_nodes in sources:
                p_str = " -> ".join(path_nodes)
                if p_str not in unique_paths:
                    unique_paths.append(p_str)

            combined_paths_str = "; ".join([f"[{p}]" for p in unique_paths])
            multi_vector_rationale = (
                f"Flagged via {', '.join(panel_summaries)}. "
                f"Collision force load paths: {combined_paths_str}. "
                f"Cumulative Concealed Risk: {noisy_or_risk * 100:.1f}% (Tier: {tier})."
            )

            # All visited load path nodes (flat, deduplicated while preserving order)
            flat_path = []
            for _, _, _, p_nodes in sources:
                for node in p_nodes:
                    if node not in flat_path:
                        flat_path.append(node)

            agg_rec = InspectionRecommendation(
                component_name=c_name,
                source_panel=base_rec.source_panel if len(sources) == 1 else "multi-panel convergence",
                impact_zone=base_rec.impact_zone,
                risk_score=noisy_or_risk,
                safety_risk_level=tier,
                recommended_action=action,
                estimated_labor_hours=base_rec.estimated_labor_hours,
                load_path=flat_path,
                rationale=multi_vector_rationale,
                detected_damage=base_rec.detected_damage,
            )
            aggregated_matrix.append(agg_rec)

        aggregated_matrix.sort(key=lambda x: x.risk_score, reverse=True)
        return aggregated_matrix


# Global singleton instance
_knowledge_graph_instance: Optional[VehicleStructuralKnowledgeGraph] = None


def get_knowledge_graph() -> VehicleStructuralKnowledgeGraph:
    """Singleton getter for VehicleStructuralKnowledgeGraph."""
    global _knowledge_graph_instance
    if _knowledge_graph_instance is None:
        _knowledge_graph_instance = VehicleStructuralKnowledgeGraph()
    return _knowledge_graph_instance


# ------------------------------------------------------------------------------
# Universal & Backward Compatibility Interface Functions
# ------------------------------------------------------------------------------
def recommend_inspections(
    arg1: Any = None,
    arg2: Any = None,
    deformation_score: Optional[float] = None,
    damaged_area_pct: Optional[float] = None,
    damage_type: Optional[str] = None,
    detected_parts: Optional[List[str]] = None,
    detected_panel: Optional[str] = None,
    panel_name: Optional[str] = None,
    area_percentage: Optional[float] = None,
    max_hops: int = 3,
    **kwargs,
) -> List[InspectionRecommendation]:
    """
    Universal recommendation interface supporting:
    1. Legacy claim summary: recommend_inspections(damage_summary)
    2. Phase 4 panel-first: recommend_inspections("front bumper cover", deformation_score=0.85, ...)
    3. Phase 4 damage-first: recommend_inspections("dent", "front bumper cover", ...)
    4. Keyword arguments: recommend_inspections(damage_type="dent", detected_panel="front bumper cover", ...)
    """
    kg = get_knowledge_graph()

    if isinstance(arg1, DamageSummary):
        recs = []
        for view in arg1.damages_by_view.keys():
            panel = "front bumper cover" if view == "front" else f"{view} bumper cover"
            recs.extend(kg.recommend_inspections(panel, deformation_score=0.10, damaged_area_pct=2.0))
        seen = set()
        unique = []
        for r in recs:
            if r.component_name not in seen:
                seen.add(r.component_name)
                unique.append(r)
        unique.sort(key=lambda x: x.risk_score, reverse=True)
        return unique

    KNOWN_DAMAGE_TYPES = {"dent", "scratch", "crack", "glass_shatter", "broken_lamp", "flat_tire", "unclassified"}

    eff_damage_type = damage_type or "dent"
    eff_panel = panel_name or detected_panel
    eff_deform = deformation_score
    eff_area = area_percentage if area_percentage is not None else damaged_area_pct

    # Disambiguate positional arguments
    if isinstance(arg1, str):
        clean_arg1 = arg1.strip().lower()
        if clean_arg1 in KNOWN_DAMAGE_TYPES:
            eff_damage_type = clean_arg1
            if isinstance(arg2, str):
                eff_panel = arg2
            elif isinstance(arg2, (int, float)) and eff_deform is None:
                eff_deform = float(arg2)
        else:
            eff_panel = arg1
            if isinstance(arg2, (int, float)) and eff_deform is None:
                eff_deform = float(arg2)
            elif isinstance(arg2, str):
                eff_damage_type = arg2

    if not eff_panel:
        eff_panel = "front bumper cover"

    return kg.recommend_inspections(
        panel_name=eff_panel,
        deformation_score=eff_deform,
        damaged_area_pct=eff_area,
        damage_type=eff_damage_type,
        max_hops=max_hops,
    )


def aggregate_claim_structural_risks(
    classified_detections: List[DamageDetection],
) -> List[InspectionRecommendation]:
    """
    Module-level convenience wrapper for VehicleStructuralKnowledgeGraph.aggregate_claim_structural_risks.
    Combines multi-vector structural impact paths across all classified claim detections using Noisy-OR.
    """
    kg = get_knowledge_graph()
    return kg.aggregate_claim_structural_risks(classified_detections)


