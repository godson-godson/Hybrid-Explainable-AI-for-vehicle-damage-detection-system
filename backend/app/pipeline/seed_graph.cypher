// ==============================================================================
// VEHICLE STRUCTURAL KNOWLEDGE GRAPH (KG) SEED SCRIPT (CYPHER)
// ==============================================================================
// Automotive Collision Energy Load Path & Concealed Component Dependency Graph.
//
// GENERAL AUTOMOTIVE COLLISION INDUSTRY REFERENCES INFORMING THIS GRAPH:
// 1. I-CAR (Inter-Industry Conference on Auto Collision Repair):
//    - "Collision Energy Management & Frontal Unibody Load Paths"
//    - "Bumper System Impact Absorption and Sensor/Cooling Package Protection"
// 2. NHTSA & IIHS (Insurance Institute for Highway Safety):
//    - "Frontal Crash Test Structural Performance & Energy Dissipation Protocols"
// 3. Thatcham Research (UK Motor Insurance Repair Research Centre):
//    - "Structural Damage Assessment: Crash Box & Rail Force Propagation Rules"
//
// NOTE: Weights represent approximate, heuristically-assigned force propagation
// likelihood (0.0 to 1.0) along collision load paths, not empirical telemetry.
//
// IDEMPOTENCY: Uses MERGE to ensure safe, repeatable execution without duplicates.
// ==============================================================================

// ------------------------------------------------------------------------------
// 1. FRONT ZONE: External Panels
// ------------------------------------------------------------------------------
MERGE (p1:ExternalPanel {name: "front bumper cover", zone: "front"})
MERGE (p2:ExternalPanel {name: "hood", zone: "front"})
MERGE (p3:ExternalPanel {name: "front grille", zone: "front"})
MERGE (p4:ExternalPanel {name: "left headlight assembly", zone: "front"})
MERGE (p5:ExternalPanel {name: "right headlight assembly", zone: "front"})
MERGE (p6:ExternalPanel {name: "left front fender", zone: "front"})
MERGE (p7:ExternalPanel {name: "right front fender", zone: "front"})

// ------------------------------------------------------------------------------
// 2. FRONT ZONE: Concealed Internal Components
// ------------------------------------------------------------------------------
MERGE (c1:InternalComponent {name: "bumper reinforcement bar", zone: "front", criticality: "high"})
MERGE (c2:InternalComponent {name: "front crush cans", zone: "front", criticality: "high"})
MERGE (c3:InternalComponent {name: "radiator support assembly", zone: "front", criticality: "high"})
MERGE (c4:InternalComponent {name: "radiator", zone: "front", criticality: "medium"})
MERGE (c5:InternalComponent {name: "air conditioning condenser", zone: "front", criticality: "medium"})
MERGE (c6:InternalComponent {name: "cooling fan assembly", zone: "front", criticality: "medium"})
MERGE (c7:InternalComponent {name: "front subframe crossmember", zone: "front", criticality: "high"})
MERGE (c8:InternalComponent {name: "hood latch and release cable", zone: "front", criticality: "high"})
MERGE (c9:InternalComponent {name: "hood hinge assembly", zone: "front", criticality: "medium"})
MERGE (c10:InternalComponent {name: "engine bay wiring harness", zone: "front", criticality: "high"})
MERGE (c11:InternalComponent {name: "windshield washer fluid reservoir", zone: "front", criticality: "low"})
MERGE (c12:InternalComponent {name: "front longitudinal frame rails", zone: "front", criticality: "high"})

// ------------------------------------------------------------------------------
// 3. FRONT ZONE: Structural Relationships & Force Propagation Load Paths
// ------------------------------------------------------------------------------
// Front bumper cover connects to bumper reinforcement bar
MERGE (p1)-[r1:PROTECTS {weight: 0.85, relationship_type: "PROTECTS"}]->(c1)
MERGE (p1)-[r2:TRANSMITS_FORCE_TO {weight: 0.90, relationship_type: "TRANSMITS_FORCE_TO"}]->(c1)
MERGE (p1)-[r3:ATTACHED_TO {weight: 0.60, relationship_type: "ATTACHED_TO"}]->(c3)

// Front grille directly protects cooling pack
MERGE (p3)-[r4:PROTECTS {weight: 0.70, relationship_type: "PROTECTS"}]->(c5)
MERGE (p3)-[r5:TRANSMITS_FORCE_TO {weight: 0.75, relationship_type: "TRANSMITS_FORCE_TO"}]->(c3)

// Bumper reinforcement bar transmits to crush cans & frame rails
MERGE (c1)-[r6:TRANSMITS_FORCE_TO {weight: 0.95, relationship_type: "TRANSMITS_FORCE_TO"}]->(c2)
MERGE (c1)-[r7:ATTACHED_TO {weight: 0.80, relationship_type: "ATTACHED_TO"}]->(c3)
MERGE (c2)-[r8:TRANSMITS_FORCE_TO {weight: 0.88, relationship_type: "TRANSMITS_FORCE_TO"}]->(c12)
MERGE (c2)-[r9:TRANSMITS_FORCE_TO {weight: 0.75, relationship_type: "TRANSMITS_FORCE_TO"}]->(c7)

// Radiator support transmits to cooling pack components
MERGE (c3)-[r10:TRANSMITS_FORCE_TO {weight: 0.80, relationship_type: "TRANSMITS_FORCE_TO"}]->(c5)
MERGE (c3)-[r11:TRANSMITS_FORCE_TO {weight: 0.75, relationship_type: "TRANSMITS_FORCE_TO"}]->(c4)
MERGE (c5)-[r12:TRANSMITS_FORCE_TO {weight: 0.70, relationship_type: "TRANSMITS_FORCE_TO"}]->(c4)
MERGE (c4)-[r13:TRANSMITS_FORCE_TO {weight: 0.65, relationship_type: "TRANSMITS_FORCE_TO"}]->(c6)

// Hood load path (Latch and hinges)
MERGE (p2)-[r14:ATTACHED_TO {weight: 0.85, relationship_type: "ATTACHED_TO"}]->(c8)
MERGE (p2)-[r15:ATTACHED_TO {weight: 0.75, relationship_type: "ATTACHED_TO"}]->(c9)
MERGE (p2)-[r16:TRANSMITS_FORCE_TO {weight: 0.60, relationship_type: "TRANSMITS_FORCE_TO"}]->(c3)
MERGE (c8)-[r17:ATTACHED_TO {weight: 0.70, relationship_type: "ATTACHED_TO"}]->(c3)

// Headlamp load path
MERGE (p4)-[r18:ATTACHED_TO {weight: 0.70, relationship_type: "ATTACHED_TO"}]->(c3)
MERGE (p4)-[r19:TRANSMITS_FORCE_TO {weight: 0.55, relationship_type: "TRANSMITS_FORCE_TO"}]->(c10)
MERGE (p4)-[r20:ADJACENT_TO {weight: 0.50, relationship_type: "ADJACENT_TO"}]->(p6)

MERGE (p5)-[r21:ATTACHED_TO {weight: 0.70, relationship_type: "ATTACHED_TO"}]->(c3)
MERGE (p5)-[r22:TRANSMITS_FORCE_TO {weight: 0.55, relationship_type: "TRANSMITS_FORCE_TO"}]->(c10)
MERGE (p5)-[r23:ADJACENT_TO {weight: 0.50, relationship_type: "ADJACENT_TO"}]->(p7)

// Fender corners protect reservoir & wiring
MERGE (p6)-[r24:PROTECTS {weight: 0.65, relationship_type: "PROTECTS"}]->(c11)
MERGE (p6)-[r25:TRANSMITS_FORCE_TO {weight: 0.50, relationship_type: "TRANSMITS_FORCE_TO"}]->(c12)

// ------------------------------------------------------------------------------
// 4. REAR ZONE: External Panels & Internal Components (Placeholder Subgraph)
// ------------------------------------------------------------------------------
MERGE (rp1:ExternalPanel {name: "rear bumper cover", zone: "rear"})
MERGE (rp2:ExternalPanel {name: "trunk lid", zone: "rear"})
MERGE (rp3:ExternalPanel {name: "rear windshield", zone: "rear"})
MERGE (rp4:ExternalPanel {name: "left taillight assembly", zone: "rear"})
MERGE (rp5:ExternalPanel {name: "right taillight assembly", zone: "rear"})

MERGE (rc1:InternalComponent {name: "rear impact reinforcement bar", zone: "rear", criticality: "high"})
MERGE (rc2:InternalComponent {name: "rear body panel sheetmetal", zone: "rear", criticality: "medium"})
MERGE (rc3:InternalComponent {name: "trunk floor pan and spare tire well", zone: "rear", criticality: "high"})
MERGE (rc4:InternalComponent {name: "rear longitudinal frame rails", zone: "rear", criticality: "high"})
MERGE (rc5:InternalComponent {name: "trunk latch and power closer", zone: "rear", criticality: "medium"})
MERGE (rc6:InternalComponent {name: "rear wiper motor and defroster harness", zone: "rear", criticality: "low"})

MERGE (rp1)-[:PROTECTS {weight: 0.85, relationship_type: "PROTECTS"}]->(rc1)
MERGE (rp1)-[:TRANSMITS_FORCE_TO {weight: 0.90, relationship_type: "TRANSMITS_FORCE_TO"}]->(rc1)
MERGE (rc1)-[:TRANSMITS_FORCE_TO {weight: 0.80, relationship_type: "TRANSMITS_FORCE_TO"}]->(rc2)
MERGE (rc1)-[:TRANSMITS_FORCE_TO {weight: 0.75, relationship_type: "TRANSMITS_FORCE_TO"}]->(rc4)
MERGE (rc2)-[:TRANSMITS_FORCE_TO {weight: 0.70, relationship_type: "TRANSMITS_FORCE_TO"}]->(rc3)
MERGE (rp2)-[:ATTACHED_TO {weight: 0.80, relationship_type: "ATTACHED_TO"}]->(rc5)
MERGE (rp3)-[:PROTECTS {weight: 0.60, relationship_type: "PROTECTS"}]->(rc6)

// ------------------------------------------------------------------------------
// 5. LEFT & RIGHT SIDE ZONES: Placeholder Subgraphs
// ------------------------------------------------------------------------------
MERGE (lp1:ExternalPanel {name: "left front door", zone: "left"})
MERGE (lp2:ExternalPanel {name: "left rear door", zone: "left"})
MERGE (lp3:ExternalPanel {name: "left rocker panel", zone: "left"})
MERGE (lc1:InternalComponent {name: "left front door anti-intrusion beam", zone: "left", criticality: "high"})
MERGE (lc2:InternalComponent {name: "left center B-pillar structural post", zone: "left", criticality: "high"})
MERGE (lc3:InternalComponent {name: "left curtain airbag deployment sensors", zone: "left", criticality: "high"})

MERGE (lp1)-[:PROTECTS {weight: 0.90, relationship_type: "PROTECTS"}]->(lc1)
MERGE (lp1)-[:TRANSMITS_FORCE_TO {weight: 0.85, relationship_type: "TRANSMITS_FORCE_TO"}]->(lc2)
MERGE (lp2)-[:TRANSMITS_FORCE_TO {weight: 0.80, relationship_type: "TRANSMITS_FORCE_TO"}]->(lc2)
MERGE (lc2)-[:TRANSMITS_FORCE_TO {weight: 0.75, relationship_type: "TRANSMITS_FORCE_TO"}]->(lc3)
MERGE (lp3)-[:ATTACHED_TO {weight: 0.80, relationship_type: "ATTACHED_TO"}]->(lc2)

MERGE (rp_r1:ExternalPanel {name: "right front door", zone: "right"})
MERGE (rp_r2:ExternalPanel {name: "right rear door", zone: "right"})
MERGE (rp_r3:ExternalPanel {name: "right rocker panel", zone: "right"})
MERGE (rc_r1:InternalComponent {name: "right front door anti-intrusion beam", zone: "right", criticality: "high"})
MERGE (rc_r2:InternalComponent {name: "right center B-pillar structural post", zone: "right", criticality: "high"})
MERGE (rc_r3:InternalComponent {name: "right curtain airbag deployment sensors", zone: "right", criticality: "high"})

MERGE (rp_r1)-[:PROTECTS {weight: 0.90, relationship_type: "PROTECTS"}]->(rc_r1)
MERGE (rp_r1)-[:TRANSMITS_FORCE_TO {weight: 0.85, relationship_type: "TRANSMITS_FORCE_TO"}]->(rc_r2)
MERGE (rp_r2)-[:TRANSMITS_FORCE_TO {weight: 0.80, relationship_type: "TRANSMITS_FORCE_TO"}]->(rc_r2)
MERGE (rc_r2)-[:TRANSMITS_FORCE_TO {weight: 0.75, relationship_type: "TRANSMITS_FORCE_TO"}]->(rc_r3)
MERGE (rp_r3)-[:ATTACHED_TO {weight: 0.80, relationship_type: "ATTACHED_TO"}]->(rc_r2)
