/**
 * Surveyor Portal Frontend Application (Phase 1)
 * AutoClaim AI - Vehicle Damage Assessment
 */

const API_BASE = ""; // Relative path to FastAPI backend

// Application State
let currentFiles = []; // Array of { file: File, view: string, previewUrl: string }
let latestClaimResult = null;

// DOM Elements
const claimIdInput = document.getElementById("claimIdInput");
const btnGenClaimId = document.getElementById("btnGenClaimId");
const regNumberInput = document.getElementById("regNumberInput");
const confThresholdInput = document.getElementById("confThresholdInput");
const confValueLabel = document.getElementById("confValueLabel");
const surveyorNotesInput = document.getElementById("surveyorNotesInput");
const uploadDropzone = document.getElementById("uploadDropzone");
const fileInput = document.getElementById("fileInput");
const fileQueueSection = document.getElementById("fileQueueSection");
const fileGrid = document.getElementById("fileGrid");
const fileCountBadge = document.getElementById("fileCountBadge");
const btnClearFiles = document.getElementById("btnClearFiles");
const btnAnalyze = document.getElementById("btnAnalyze");
const btnLoadSamples = document.getElementById("btnLoadSamples");
const inspectionForm = document.getElementById("inspectionForm");

const loadingState = document.getElementById("loadingState");
const resultsContainer = document.getElementById("resultsContainer");
const systemStatusPill = document.getElementById("systemStatusPill");
const systemStatusText = document.getElementById("systemStatusText");
const btnRefreshHealth = document.getElementById("btnRefreshHealth");

// Modal Elements
const imageModal = document.getElementById("imageModal");
const modalImage = document.getElementById("modalImage");
const modalTitle = document.getElementById("modalTitle");
const btnCloseModal = document.getElementById("btnCloseModal");

const VIEW_OPTIONS = [
  { value: "front", label: "Front View" },
  { value: "rear", label: "Rear View" },
  { value: "left", label: "Left Side" },
  { value: "right", label: "Right Side" },
  { value: "close_up", label: "Close-Up Inspection" },
  { value: "overview", label: "Overview Angle" },
];

// Initialize on page load
document.addEventListener("DOMContentLoaded", () => {
  generateNewClaimId();
  checkHealth();
  loadStoredClaims();
  bindEvents();
});

function generateNewClaimId() {
  const randomSuffix = Math.floor(1000 + Math.random() * 9000);
  claimIdInput.value = `CLM-2026-${randomSuffix}`;
}

function bindEvents() {
  btnGenClaimId.addEventListener("click", generateNewClaimId);

  confThresholdInput.addEventListener("input", (e) => {
    confValueLabel.textContent = parseFloat(e.target.value).toFixed(2);
  });

  btnRefreshHealth.addEventListener("click", checkHealth);

  // Dropzone Events
  uploadDropzone.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", handleFileInputChange);

  uploadDropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadDropzone.classList.add("dragover");
  });

  uploadDropzone.addEventListener("dragleave", () => {
    uploadDropzone.classList.remove("dragover");
  });

  uploadDropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadDropzone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      addFiles(Array.from(e.dataTransfer.files));
    }
  });

  btnClearFiles.addEventListener("click", clearAllFiles);
  btnLoadSamples.addEventListener("click", loadSampleImages);

  inspectionForm.addEventListener("submit", handleFormSubmit);

  btnCloseModal.addEventListener("click", () => {
    imageModal.style.display = "none";
  });

  imageModal.addEventListener("click", (e) => {
    if (e.target === imageModal) {
      imageModal.style.display = "none";
    }
  });

  document.getElementById("btnExportJson").addEventListener("click", exportClaimJson);
  document.getElementById("btnRefreshClaims").addEventListener("click", loadStoredClaims);

  // Phase 5: Initialize Document Intake & Report Dossier handlers
  initDocumentSection();
  initReportSection();
}


// Health Check API
async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/api/health`);
    if (!res.ok) throw new Error("Health check failed");
    const data = await res.json();
    if (data.status === "HEALTHY") {
      const dev = data.vision_pipeline?.damage_segmenter_sam2?.device?.toUpperCase() || "CPU";
      systemStatusPill.className = "status-pill online";
      systemStatusText.textContent = `YOLO11m + SAM2 Ready (${dev})`;

      // Update Knowledge Graph Mode Badge
      const kgInfo = data.vision_pipeline?.structural_knowledge_graph;
      const kgBadge = document.getElementById("kgModeBadge");
      if (kgBadge && kgInfo) {
        if (kgInfo.status === "connected_live") {
          kgBadge.textContent = `Neo4j Live (${kgInfo.nodes} nodes)`;
          kgBadge.className = "badge-kg-mode";
        } else {
          kgBadge.textContent = `Embedded Graph (${kgInfo.nodes} nodes)`;
          kgBadge.className = "badge-kg-mode embedded";
        }
      }
    } else {
      systemStatusPill.className = "status-pill";
      systemStatusText.textContent = "Model Degraded";
    }
  } catch (err) {
    systemStatusPill.className = "status-pill";
    systemStatusText.textContent = "Backend Offline";
  }
}

// File Selection & Queue Handling
function handleFileInputChange(e) {
  if (e.target.files && e.target.files.length > 0) {
    addFiles(Array.from(e.target.files));
  }
  fileInput.value = "";
}

function addFiles(filesList) {
  const validFiles = filesList.filter((f) =>
    ["image/jpeg", "image/png", "image/webp", "image/jpg"].includes(f.type)
  );

  if (validFiles.length === 0) {
    alert("Please select valid image files (JPEG, PNG, WebP).");
    return;
  }

  validFiles.forEach((file) => {
    const previewUrl = URL.createObjectURL(file);
    const defaultView = VIEW_OPTIONS[Math.min(currentFiles.length, VIEW_OPTIONS.length - 1)].value;
    currentFiles.push({
      file,
      view: defaultView,
      previewUrl,
    });
  });

  renderFileQueue();
}

function removeFile(index) {
  if (currentFiles[index]) {
    URL.revokeObjectURL(currentFiles[index].previewUrl);
    currentFiles.splice(index, 1);
    renderFileQueue();
  }
}

function clearAllFiles() {
  currentFiles.forEach((f) => URL.revokeObjectURL(f.previewUrl));
  currentFiles = [];
  renderFileQueue();
}

function renderFileQueue() {
  if (currentFiles.length === 0) {
    fileQueueSection.style.display = "none";
    btnAnalyze.disabled = true;
    fileCountBadge.textContent = "0";
    return;
  }

  fileQueueSection.style.display = "flex";
  btnAnalyze.disabled = false;
  fileCountBadge.textContent = currentFiles.length;
  fileGrid.innerHTML = "";

  currentFiles.forEach((item, idx) => {
    const card = document.createElement("div");
    card.className = "file-card";

    const optionsHtml = VIEW_OPTIONS.map(
      (opt) => `<option value="${opt.value}" ${item.view === opt.value ? "selected" : ""}>${opt.label}</option>`
    ).join("");

    card.innerHTML = `
      <img src="${item.previewUrl}" class="file-thumb" alt="Preview">
      <button type="button" class="btn-remove-file" title="Remove" onclick="removeFile(${idx})">&times;</button>
      <div class="file-card-body">
        <span class="file-name" title="${item.file.name}">${item.file.name}</span>
        <select class="view-select" onchange="updateViewTag(${idx}, this.value)">
          ${optionsHtml}
        </select>
      </div>
    `;

    fileGrid.appendChild(card);
  });
}

window.removeFile = removeFile;
window.updateViewTag = function (index, viewValue) {
  if (currentFiles[index]) {
    currentFiles[index].view = viewValue;
  }
};

// Load Sample Repo Images helper
async function loadSampleImages() {
  try {
    btnLoadSamples.disabled = true;
    btnLoadSamples.textContent = "Loading sample images...";

    // Fetch sample files from public repo
    const sampleUrls = [
      { name: "sample_car_1.png", url: "/static/samples/1.png", view: "front" },
      { name: "sample_car_2.png", url: "/static/samples/2.png", view: "rear" },
      { name: "sample_car_3.png", url: "/static/samples/3.png", view: "left" },
    ];

    clearAllFiles();

    for (const sample of sampleUrls) {
      const res = await fetch(sample.url);
      if (!res.ok) continue;
      const blob = await res.blob();
      const file = new File([blob], sample.name, { type: "image/png" });
      currentFiles.push({
        file,
        view: sample.view,
        previewUrl: URL.createObjectURL(file),
      });
    }

    renderFileQueue();
  } catch (err) {
    console.error("Failed to load sample images:", err);
    alert("Could not load sample images automatically. Please select photos using the browse button.");
  } finally {
    btnLoadSamples.disabled = false;
    btnLoadSamples.innerHTML = `
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M7 10l5 5 5-5M12 15V3"/>
      </svg>
      Load Sample Repo Images
    `;
  }
}

// Form Submission & API Call
async function handleFormSubmit(e) {
  e.preventDefault();

  const claimId = claimIdInput.value.trim();
  if (!claimId) {
    alert("Please enter or generate a Claim ID.");
    return;
  }

  if (currentFiles.length === 0) {
    alert("Please upload at least one vehicle photo before running inspection.");
    return;
  }

  // Build FormData payload
  const formData = new FormData();
  formData.append("vehicle_reg_number", regNumberInput.value.trim());
  formData.append("surveyor_notes", surveyorNotesInput.value.trim());
  formData.append("confidence_threshold", confThresholdInput.value);

  const amgToggle = document.getElementById("amgToggleInput");
  formData.append("enable_amg", amgToggle && amgToggle.checked ? "true" : "false");

  currentFiles.forEach((item) => {
    formData.append("files", item.file);
    formData.append("view_angles", item.view);
  });

  // UI state: Loading
  btnAnalyze.disabled = true;
  loadingState.style.display = "flex";
  resultsContainer.style.display = "none";

  try {
    const response = await fetch(`${API_BASE}/api/claims/${encodeURIComponent(claimId)}/inspect`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const errData = await response.json().catch(() => ({}));
      throw new Error(errData.detail || `Server error: ${response.status}`);
    }

    const claimResult = await response.json();
    latestClaimResult = claimResult;
    renderClaimResults(claimResult);
    loadStoredClaims();
  } catch (error) {
    console.error("Damage assessment error:", error);
    alert(`Damage assessment failed: ${error.message}`);
  } finally {
    loadingState.style.display = "none";
    btnAnalyze.disabled = false;
  }
}

// Render Results Dashboard
function renderClaimResults(result) {
  resultsContainer.style.display = "flex";
  resultsContainer.scrollIntoView({ behavior: "smooth" });

  const meta = document.getElementById("claimMetaText");
  const dt = new Date(result.created_at).toLocaleString();
  meta.textContent = `Claim ID: ${result.claim_id} | Reg No: ${result.vehicle_reg_number || "N/A"} | Assessed: ${dt}`;

  // Severity Badge
  const badge = document.getElementById("severityBadge");
  const sev = result.damage_summary.severity_assessment || "Moderate";
  badge.textContent = sev;
  badge.className = "severity-badge";
  if (sev.toLowerCase().includes("minor") || sev.toLowerCase().includes("no damage")) {
    badge.classList.add("minor");
  } else if (sev.toLowerCase().includes("severe")) {
    badge.classList.add("severe");
  } else {
    badge.classList.add("moderate");
  }

  // KPIs
  const counts = result.damage_summary.damage_counts_by_type || {};
  document.getElementById("kpiTotalDamages").textContent = result.damage_summary.total_damages_count || 0;
  document.getElementById("kpiDents").textContent = counts["dent"] || 0;
  document.getElementById("kpiScratches").textContent = counts["scratch"] || 0;
  document.getElementById("kpiCracks").textContent = counts["crack"] || 0;
  document.getElementById("kpiBrokenLamps").textContent = counts["broken_lamp"] || 0;
  document.getElementById("kpiShatteredGlass").textContent = counts["shattered_glass"] || 0;
  document.getElementById("kpiFlatTires").textContent = counts["flat_tire"] || 0;

  const unclassifiedKpi = document.getElementById("kpiUnclassified");
  if (unclassifiedKpi) {
    unclassifiedKpi.textContent = result.damage_summary.unclassified_candidates_count || 0;
  }

  // Latency Profiling Display
  const latencyBanner = document.getElementById("latencyBanner");
  const latencyPills = document.getElementById("latencyPills");
  if (latencyBanner && latencyPills) {
    let totalYolo = 0, totalBoxSam2 = 0, totalCrop = 0, totalAmg = 0, totalDepth = 0, totalKg = 0, totalPipeline = 0;
    let countWithLatency = 0;

    result.images.forEach((img) => {
      if (img.latency_metrics) {
        totalYolo += img.latency_metrics.yolo_detection_ms;
        totalBoxSam2 += img.latency_metrics.box_prompted_sam2_ms;
        totalCrop += img.latency_metrics.vehicle_crop_ms;
        totalAmg += img.latency_metrics.amg_sam2_ms;
        totalDepth += (img.latency_metrics.depth_estimation_ms || 0);
        totalKg += (img.latency_metrics.knowledge_graph_ms || 0);
        totalPipeline += img.latency_metrics.total_pipeline_ms;
        countWithLatency++;
      }
    });

    if (countWithLatency > 0) {
      latencyBanner.style.display = "flex";
      latencyPills.innerHTML = `
        <span class="latency-pill">YOLO11m: <strong>${Math.round(totalYolo)} ms</strong></span>
        <span class="latency-pill">Padded SAM2: <strong>${Math.round(totalBoxSam2)} ms</strong></span>
        <span class="latency-pill">Vehicle ROI: <strong>${Math.round(totalCrop)} ms</strong></span>
        <span class="latency-pill">SAM2 AMG: <strong>${Math.round(totalAmg)} ms</strong></span>
        <span class="latency-pill">Depth V2: <strong>${Math.round(totalDepth)} ms</strong></span>
        <span class="latency-pill">KG Traversal: <strong>${Math.round(totalKg)} ms</strong></span>
        <span class="latency-pill total">Total: <strong>${(totalPipeline / 1000).toFixed(2)} s</strong></span>
      `;
    } else {
      latencyBanner.style.display = "none";
    }
  }

  // Gallery
  const gallery = document.getElementById("inspectionGallery");
  gallery.innerHTML = "";

  result.images.forEach((img, idx) => {
    const card = document.createElement("div");
    card.className = "gallery-card";

    const segmentedUrl = img.segmented_image_url || img.annotated_image_url;
    const annotatedUrl = img.annotated_image_url;
    const originalUrl = img.original_image_url;
    const depthUrl = img.depth_colormap_url;

    // Classified tags with tier & deformation indicator
    const classifiedTags = (img.detections || []).map((d) => {
      const areaStr = d.segmentation ? ` | ${d.segmentation.area_percentage.toFixed(2)}%` : "";
      const isLow = (d.confidence_tier === "low");
      const tierTag = isLow ? " [LOW]" : "";
      let deformTag = "";
      if (d.deformation && d.deformation.severity_tier) {
        deformTag = ` | ${d.deformation.severity_tier.toUpperCase()}`;
      }
      return `<span class="damage-tag ${d.damage_type} ${isLow ? 'tier-low' : ''}">${d.damage_type.replace("_", " ")} (${(d.confidence * 100).toFixed(0)}%${tierTag}${areaStr}${deformTag})</span>`;
    });

    // Unclassified candidate tags (capped at top N by area descending)
    const TOP_N_UNCLASSIFIED = 5;
    const rawUnclass = img.unclassified_regions || [];
    const sortedUnclass = [...rawUnclass].sort((a, b) => {
      const aArea = a.segmentation ? a.segmentation.area_percentage : 0;
      const bArea = b.segmentation ? b.segmentation.area_percentage : 0;
      return bArea - aArea;
    });

    const displayedUnclass = sortedUnclass.slice(0, TOP_N_UNCLASSIFIED);
    const collapsedCount = sortedUnclass.length - displayedUnclass.length;

    const unclassifiedTags = displayedUnclass.map((u) => {
      const areaStr = u.segmentation ? ` | ${u.segmentation.area_percentage.toFixed(2)}%` : "";
      let deformTag = "";
      if (u.deformation && u.deformation.severity_tier) {
        deformTag = ` | ${u.deformation.severity_tier.toUpperCase()}`;
      }
      return `<span class="damage-tag unclassified">Unclassified Fragment${areaStr}${deformTag}</span>`;
    });

    if (collapsedCount > 0) {
      unclassifiedTags.push(
        `<span class="damage-tag unclassified-collapsed" title="${collapsedCount} additional minor fragments filtered from primary display">+${collapsedCount} additional minor fragment${collapsedCount === 1 ? '' : 's'} filtered out</span>`
      );
    }

    const allTags = [...classifiedTags, ...unclassifiedTags];
    const unclassCount = rawUnclass.length;
    const unclassBadge = unclassCount > 0 ? ` + ${unclassCount} Fragment${unclassCount === 1 ? '' : 's'}` : "";

    const depthBtnHtml = depthUrl
      ? `<button type="button" class="btn-toggle-view" id="btnDepth_${idx}" onclick="switchImageView(${idx}, 'depth', '${depthUrl}')">Depth Map</button>`
      : '';

    card.innerHTML = `
      <div class="gallery-card-header">
        <span class="view-badge">${img.view_angle}</span>
        <span class="damage-count-badge ${img.damage_count === 0 && unclassCount === 0 ? "zero" : ""}">
          ${img.damage_count} Damage${img.damage_count === 1 ? "" : "s"}${unclassBadge}
        </span>
      </div>
      <div class="gallery-image-wrapper" onclick="openActiveModal(${idx})">
        <img src="${segmentedUrl}" id="imgDisplay_${idx}" class="gallery-img" alt="Inspection Visual" data-active-url="${segmentedUrl}">
      </div>
      <div class="image-toggle-bar">
        <button type="button" class="btn-toggle-view active" id="btnMask_${idx}" onclick="switchImageView(${idx}, 'mask', '${segmentedUrl}')">SAM2 Masks</button>
        ${depthBtnHtml}
        <button type="button" class="btn-toggle-view" id="btnBbox_${idx}" onclick="switchImageView(${idx}, 'bbox', '${annotatedUrl}')">Boxes</button>
        <button type="button" class="btn-toggle-view" id="btnOrig_${idx}" onclick="switchImageView(${idx}, 'orig', '${originalUrl}')">Original</button>
      </div>
      <div class="gallery-detections-list">
        ${allTags.length > 0 ? allTags.join("") : '<span class="text-muted" style="font-size:0.75rem;">No damages or fragments detected on this angle</span>'}
      </div>
    `;

    gallery.appendChild(card);
  });

  // Helper function to render deformation table cells
  function formatDeformationCells(item) {
    const def = item.deformation;
    if (!def) {
      return `<td><span class="text-muted">N/A</span></td><td><span class="text-muted">-</span></td>`;
    }
    if (def.deformation_status === "reference_unavailable" || def.deformation_status === "reference_unreliable_edge_boundary") {
      return `
        <td><span class="badge-sev-unavail" title="Reference ring excluded due to adjacent damage or image borders">Ref Unavail</span></td>
        <td><span class="text-muted">-</span></td>
      `;
    }
    if (def.deformation_status === "too_small_for_reliable_severity_estimate" || def.deformation_status === "too_small") {
      return `
        <td><span class="badge-sev-too-small" title="Fragment area is too small for reliable 3D depth deformation estimate">Too Small</span></td>
        <td><span class="text-muted">-</span></td>
      `;
    }
    const score = def.relative_deformation_score;
    const scoreFormatted = (score !== null && score !== undefined)
      ? `${score >= 0 ? '+' : ''}${score.toFixed(3)}`
      : '0.000';
    const typeLabel = def.deformation_type || 'relative';
    const sevTier = def.severity_tier;
    const sevBadge = sevTier
      ? `<span class="badge-sev-${sevTier}">${sevTier.toUpperCase()}</span>`
      : `<span class="text-muted">-</span>`;
    return `
      <td>
        <div class="deform-cell">
          <span class="deform-val">${scoreFormatted}</span>
          <span class="deform-sub">${typeLabel}</span>
        </div>
      </td>
      <td>${sevBadge}</td>
    `;
  }

  // Detections Table
  const tbody = document.getElementById("detectionsTableBody");
  tbody.innerHTML = "";

  let rowCounter = 1;
  result.images.forEach((img) => {
    // 1. Render Classified Detections
    (img.detections || []).forEach((det) => {
      const tr = document.createElement("tr");
      const confPct = (det.confidence * 100).toFixed(1);
      const b = det.bbox;
      const seg = det.segmentation;
      const areaHtml = seg
        ? `<span class="area-badge"><strong>${seg.area_percentage.toFixed(2)}%</strong> (${seg.area_pixels.toLocaleString()} px)</span>`
        : `<span class="text-muted">N/A</span>`;

      const tierBadge = det.confidence_tier === "low"
        ? `<span class="badge-tier low">LOW (REVIEW)</span>`
        : `<span class="badge-tier high">HIGH</span>`;

      const deformCells = formatDeformationCells(det);
      const panelName = det.detected_panel ? det.detected_panel.replace(/_/g, " ").toUpperCase() : "UNKNOWN";
      const panelBadge = `<span class="detected-panel-badge">${panelName}</span>`;

      tr.innerHTML = `
        <td><strong>#${rowCounter++}</strong></td>
        <td><span class="view-badge">${img.view_angle}</span></td>
        <td>${panelBadge}</td>
        <td><span class="damage-tag ${det.damage_type}">${det.damage_type.replace("_", " ").toUpperCase()}</span></td>
        <td>
          <div class="confidence-bar-container">
            <div class="confidence-bar-bg">
              <div class="confidence-bar-fill" style="width: ${confPct}%"></div>
            </div>
            <strong>${confPct}%</strong>
            ${tierBadge}
          </div>
        </td>
        <td>${areaHtml}</td>
        ${deformCells}
        <td class="coord-cell">[${b.x1}, ${b.y1}, ${b.x2}, ${b.y2}]</td>
        <td class="coord-cell">${b.width} × ${b.height} px</td>
        <td><span class="badge-segmented-ready">SAM2 Segmented</span></td>
      `;

      tbody.appendChild(tr);
    });

    // 2. Render Unclassified Candidate Regions (AMG) - Top N by area descending
    const rawUnclassTable = img.unclassified_regions || [];
    const sortedUnclassTable = [...rawUnclassTable].sort((a, b) => {
      const aArea = a.segmentation ? a.segmentation.area_percentage : 0;
      const bArea = b.segmentation ? b.segmentation.area_percentage : 0;
      return bArea - aArea;
    });

    const TOP_N_TABLE = 5;
    const displayedUnclassTable = sortedUnclassTable.slice(0, TOP_N_TABLE);
    const collapsedCountTable = sortedUnclassTable.length - displayedUnclassTable.length;

    displayedUnclassTable.forEach((unclass) => {
      const tr = document.createElement("tr");
      tr.style.backgroundColor = "rgba(100, 116, 139, 0.05)";
      const confPct = (unclass.confidence * 100).toFixed(1);
      const b = unclass.bbox;
      const seg = unclass.segmentation;
      const areaHtml = seg
        ? `<span class="area-badge"><strong>${seg.area_percentage.toFixed(2)}%</strong> (${seg.area_pixels.toLocaleString()} px)</span>`
        : `<span class="text-muted">N/A</span>`;

      const deformCells = formatDeformationCells(unclass);
      const panelBadge = `<span class="detected-panel-badge fallback">Needs Review</span>`;

      tr.innerHTML = `
        <td><strong>#${rowCounter++}</strong></td>
        <td><span class="view-badge">${img.view_angle}</span></td>
        <td>${panelBadge}</td>
        <td><span class="damage-tag unclassified">UNCLASSIFIED</span></td>
        <td>
          <div class="confidence-bar-container">
            <div class="confidence-bar-bg">
              <div class="confidence-bar-fill" style="width: ${confPct}%; background: #94A3B8;"></div>
            </div>
            <strong>${confPct}%</strong>
            <span class="badge-tier low">REVIEW</span>
          </div>
        </td>
        <td>${areaHtml}</td>
        ${deformCells}
        <td class="coord-cell">[${b.x1}, ${b.y1}, ${b.x2}, ${b.y2}]</td>
        <td class="coord-cell">${b.width} × ${b.height} px</td>
        <td><span class="badge-unclass-status">Needs Review (AMG)</span></td>
      `;

      tbody.appendChild(tr);
    });

    if (collapsedCountTable > 0) {
      const trSummary = document.createElement("tr");
      trSummary.className = "unclass-collapsed-row";
      trSummary.innerHTML = `
        <td colspan="11">
          <div class="unclass-collapsed-wrapper">
            <span class="badge-unclass-collapsed">+${collapsedCountTable} additional minor fragment${collapsedCountTable === 1 ? '' : 's'} filtered out</span>
            <span class="unclass-collapsed-desc text-muted">Filtered from primary list (below top ${TOP_N_TABLE} largest fragments).</span>
          </div>
        </td>
      `;
      tbody.appendChild(trSummary);
    }
  });

  if (rowCounter === 1) {
    tbody.innerHTML = `<tr><td colspan="11" class="text-muted" style="text-align:center; padding: 20px;">No damage detections or fragments registered for this claim.</td></tr>`;
  }

  // 3. Render Probabilistic Structural Risk Matrix (Phase 4)
  const riskTbody = document.getElementById("riskMatrixTableBody");
  if (riskTbody) {
    riskTbody.innerHTML = "";
    const matrix = result.structural_risk_matrix || [];
    if (matrix.length === 0) {
      riskTbody.innerHTML = `<tr><td colspan="7" class="text-muted" style="text-align:center; padding: 24px;">No internal structural component risks flagged for this claim.</td></tr>`;
    } else {
      matrix.forEach((rec) => {
        const tr = document.createElement("tr");
        const compName = rec.component_name ? rec.component_name.replace(/_/g, " ").toUpperCase() : "UNKNOWN";
        const zone = rec.impact_zone || "VEHICLE";
        const riskPct = Math.round((rec.risk_score || 0) * 100);
        const tierClass = (rec.priority_tier || "low").toLowerCase();
        const meterFillClass = riskPct >= 70 ? "risk-high" : riskPct >= 40 ? "risk-med" : "risk-low";

        const pathChips = (rec.load_path || []).map((step, sIdx, arr) => {
          const isSource = sIdx === 0;
          const isTarget = sIdx === arr.length - 1;
          const stepClass = isSource ? 'source' : isTarget ? 'target' : '';
          const nameClean = step.replace(/_/g, ' ');
          const arrow = sIdx < arr.length - 1 ? '<span class="load-path-arrow">➔</span>' : '';
          return `<span class="load-path-step ${stepClass}">${nameClean}</span>${arrow}`;
        }).join(" ");

        tr.innerHTML = `
          <td>
            <strong>${compName}</strong>
          </td>
          <td>
            <span class="risk-zone-badge">${zone}</span>
          </td>
          <td>
            <div class="risk-meter-wrapper">
              <div class="risk-meter-header">
                <span class="risk-score-val">${riskPct}%</span>
                <span class="text-muted" style="font-size:0.7rem;">(${(rec.risk_score || 0).toFixed(3)})</span>
              </div>
              <div class="risk-meter-bg">
                <div class="risk-meter-fill ${meterFillClass}" style="width: ${riskPct}%"></div>
              </div>
            </div>
          </td>
          <td>
            <span class="badge-risk ${tierClass}">${(rec.priority_tier || 'LOW').toUpperCase()}</span>
          </td>
          <td>
            <span style="font-weight: 600; font-size: 0.85rem;">${rec.recommended_action || 'Inspect'}</span>
          </td>
          <td>
            <span class="labor-tag">⏱ ${rec.estimated_labor_hours || '1.0 - 2.0 hrs'}</span>
          </td>
          <td>
            <div class="load-path-flow">${pathChips}</div>
            <p class="risk-rationale-text">${rec.rationale || ''}</p>
          </td>
        `;
        riskTbody.appendChild(tr);
      });
    }
  }

  // Phase 5: Populate Documents and Survey Report if present on claim
  if (result.documents) {
    renderClaimDocuments(result.documents);
  }
  if (result.report) {
    renderSurveyReport(result.report);
  }
}


// Toggle between segmented mask, depth map, bounding box, and original image
window.switchImageView = function (idx, mode, url) {
  const imgEl = document.getElementById(`imgDisplay_${idx}`);
  const btnMask = document.getElementById(`btnMask_${idx}`);
  const btnDepth = document.getElementById(`btnDepth_${idx}`);
  const btnBbox = document.getElementById(`btnBbox_${idx}`);
  const btnOrig = document.getElementById(`btnOrig_${idx}`);

  if (imgEl) {
    imgEl.src = url;
    imgEl.dataset.activeUrl = url;
  }
  if (btnMask) btnMask.classList.toggle("active", mode === "mask");
  if (btnDepth) btnDepth.classList.toggle("active", mode === "depth");
  if (btnBbox) btnBbox.classList.toggle("active", mode === "bbox");
  if (btnOrig) btnOrig.classList.toggle("active", mode === "orig");
};

window.openActiveModal = function (idx) {
  const imgEl = document.getElementById(`imgDisplay_${idx}`);
  if (imgEl && imgEl.dataset.activeUrl) {
    openModal(imgEl.dataset.activeUrl, `Inspection Visual - Image #${idx + 1}`);
  }
};

// Modal Preview
window.openModal = function (imageUrl, title) {
  modalImage.src = imageUrl;
  modalTitle.textContent = title;
  imageModal.style.display = "flex";
};

// Export JSON
function exportClaimJson() {
  if (!latestClaimResult) return;
  const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(latestClaimResult, null, 2));
  const downloadAnchor = document.createElement("a");
  downloadAnchor.setAttribute("href", dataStr);
  downloadAnchor.setAttribute("download", `${latestClaimResult.claim_id}_damage_assessment.json`);
  document.body.appendChild(downloadAnchor);
  downloadAnchor.click();
  downloadAnchor.remove();
}

// Stored Claims List
async function loadStoredClaims() {
  const listEl = document.getElementById("claimsList");
  try {
    const res = await fetch(`${API_BASE}/api/claims?limit=10`);
    if (!res.ok) throw new Error("Failed to load claims");
    const claims = await res.json();

    if (!claims || claims.length === 0) {
      listEl.innerHTML = `<p class="text-muted" style="font-size:0.85rem;">No saved claims yet. Run a damage assessment above to populate the repository.</p>`;
      return;
    }

    listEl.innerHTML = claims
      .map(
        (c) => `
        <div class="claim-item" onclick="loadSingleClaim('${c.claim_id}')">
          <div>
            <span class="claim-item-id">${c.claim_id}</span>
            <span class="claim-item-meta" style="margin-left: 10px;">${c.images.length} Photos | ${c.damage_summary.total_damages_count} Damages (${c.damage_summary.severity_assessment})</span>
          </div>
          <span class="badge-stub-ready">${new Date(c.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
        </div>
      `
      )
      .join("");
  } catch (err) {
    listEl.innerHTML = `<p class="text-muted" style="font-size:0.85rem;">Error loading claims repository.</p>`;
  }
}

window.loadSingleClaim = async function (claimId) {
  try {
    const res = await fetch(`${API_BASE}/api/claims/${encodeURIComponent(claimId)}`);
    if (!res.ok) throw new Error("Claim not found");
    const claim = await res.json();
    latestClaimResult = claim;
    renderClaimResults(claim);
  } catch (err) {
    alert(`Could not load claim: ${err.message}`);
  }
};

/* ==========================================================================
   Phase 5: Document Intake & KYC Verification Functions
   ========================================================================== */

let activeDocType = "rc_book";
let activeDocData = {
  rc_book: null,
  insurance_policy: null,
  driving_licence: null,
};
let activeSurveyReport = null;

const DOC_LABELS = {
  rc_book: "Registration Certificate (RC)",
  insurance_policy: "Motor Insurance Policy",
  driving_licence: "Driving Licence",
};

function initDocumentSection() {
  const docTabBtns = document.querySelectorAll(".doc-tab-btn");
  const docUploadZone = document.getElementById("docUploadZone");
  const docFileInput = document.getElementById("docFileInput");
  const btnSaveDocOverrides = document.getElementById("btnSaveDocOverrides");

  if (!docUploadZone || !docFileInput) return;

  // Tab switching
  docTabBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      docTabBtns.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      activeDocType = btn.getAttribute("data-doc-type");

      const labelEl = document.getElementById("currentDocLabel");
      if (labelEl) labelEl.textContent = DOC_LABELS[activeDocType] || "Document";

      renderActiveDocFields();
    });
  });

  // Dropzone click & drag
  docUploadZone.addEventListener("click", () => docFileInput.click());
  docFileInput.addEventListener("change", handleDocFileInputChange);

  docUploadZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    docUploadZone.classList.add("drag-over");
  });

  docUploadZone.addEventListener("dragleave", () => {
    docUploadZone.classList.remove("drag-over");
  });

  docUploadZone.addEventListener("drop", (e) => {
    e.preventDefault();
    docUploadZone.classList.remove("drag-over");
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      uploadDocumentFile(e.dataTransfer.files[0]);
    }
  });

  if (btnSaveDocOverrides) {
    btnSaveDocOverrides.addEventListener("click", saveActiveDocOverrides);
  }
}

function handleDocFileInputChange(e) {
  if (e.target.files && e.target.files.length > 0) {
    uploadDocumentFile(e.target.files[0]);
  }
}

async function uploadDocumentFile(file) {
  const claimId = claimIdInput.value.trim();
  if (!claimId) {
    alert("Please enter or generate a Claim / Session ID before uploading documents.");
    return;
  }

  const spinner = document.getElementById("docSpinnerBox");
  const reviewPanel = document.getElementById("docFieldsReview");
  if (spinner) spinner.style.display = "flex";
  if (reviewPanel) reviewPanel.style.display = "none";

  const formData = new FormData();
  formData.append("file", file);
  formData.append("document_type", activeDocType);

  try {
    const res = await fetch(`${API_BASE}/api/claims/${encodeURIComponent(claimId)}/documents`, {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: "Upload failed" }));
      throw new Error(err.detail || "Failed to upload document");
    }

    const extractionResult = await res.json();
    activeDocData[activeDocType] = extractionResult;
    renderActiveDocFields();
    updateCrossCheckBadge();

    // Auto-populate registration number input if RC provides it
    if (activeDocType === "rc_book" && extractionResult.fields && extractionResult.fields.registration_number) {
      const regVal = extractionResult.fields.registration_number.value;
      if (regVal && regNumberInput && !regNumberInput.value) {
        regNumberInput.value = regVal;
      }
    }
  } catch (err) {
    alert(`Document OCR Error: ${err.message}`);
  } finally {
    if (spinner) spinner.style.display = "none";
  }
}

function renderActiveDocFields() {
  const reviewPanel = document.getElementById("docFieldsReview");
  const tableBody = document.getElementById("docFieldsTableBody");
  const badgesContainer = document.getElementById("docReviewBadges");
  const titleEl = document.getElementById("docReviewTitle");

  if (!reviewPanel || !tableBody) return;

  const currentData = activeDocData[activeDocType];
  if (!currentData || !currentData.fields || Object.keys(currentData.fields).length === 0) {
    reviewPanel.style.display = "none";
    return;
  }

  reviewPanel.style.display = "flex";
  if (titleEl) titleEl.textContent = `${DOC_LABELS[activeDocType]} — Extracted Credentials`;

  // Status badges
  if (badgesContainer) {
    const isNeedsReview = currentData.needs_manual_review;
    const latency = currentData.latency_ms ? `${currentData.latency_ms.toFixed(0)}ms` : "";
    const method = currentData.extraction_method === "vision_fallback"
      ? "👁️ Vision Fallback (<40 chars)"
      : "📄 PaddleOCR + Llama-3.3-70B";
    badgesContainer.innerHTML = `
      <span class="${isNeedsReview ? 'status-pill-review' : 'status-pill-valid'}">
        ${isNeedsReview ? 'Needs Surveyor Review' : 'Verified Valid'}
      </span>
      <span class="badge-stub-ready" style="font-size:0.75rem;">${method}</span>
      ${latency ? `<span class="badge-stub-ready" style="font-size:0.75rem;">⚡ Latency: ${latency}</span>` : ''}
    `;
  }

  tableBody.innerHTML = "";
  const fields = currentData.fields;

  for (const [key, item] of Object.entries(fields)) {
    const tr = document.createElement("tr");
    const friendlyName = key.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
    const val = item.value !== null && item.value !== undefined ? item.value : "";
    const snippet = item.source_snippet ? item.source_snippet : "No snippet";
    const confPct = (item.confidence * 100).toFixed(0);

    const isFieldValid = item.is_valid !== false && val !== "";
    const statusPill = isFieldValid
      ? `<span class="status-pill-valid">${confPct}% Valid</span>`
      : `<span class="status-pill-review" title="${item.validation_error || 'Check value'}">Review (${confPct}%)</span>`;

    tr.innerHTML = `
      <td><strong>${friendlyName}</strong></td>
      <td>
        <input type="text" class="doc-field-input" data-field-key="${key}" value="${val}" placeholder="Not detected (null)">
      </td>
      <td>
        <span class="doc-snippet-tag" title="${snippet}">"${snippet}"</span>
      </td>
      <td>
        ${statusPill}
      </td>
    `;
    tableBody.appendChild(tr);
  }
}

async function saveActiveDocOverrides() {
  const claimId = claimIdInput.value.trim();
  const currentData = activeDocData[activeDocType];
  const statusMsg = document.getElementById("docSaveStatusMsg");

  if (!claimId || !currentData) return;

  const inputs = document.querySelectorAll("#docFieldsTableBody .doc-field-input");
  const updates = {};
  inputs.forEach((inp) => {
    const key = inp.getAttribute("data-field-key");
    if (key) updates[key] = inp.value.trim() || null;
  });

  try {
    if (statusMsg) statusMsg.textContent = "Saving changes...";
    const res = await fetch(`${API_BASE}/api/claims/${encodeURIComponent(claimId)}/documents/${encodeURIComponent(activeDocType)}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(updates),
    });

    if (!res.ok) throw new Error("Failed to save changes");
    const updated = await res.json();
    activeDocData[activeDocType] = updated;
    renderActiveDocFields();
    updateCrossCheckBadge();

    if (statusMsg) {
      statusMsg.textContent = "✅ Saved & Confirmed!";
      setTimeout(() => { statusMsg.textContent = ""; }, 3000);
    }
  } catch (err) {
    if (statusMsg) statusMsg.textContent = `❌ Error: ${err.message}`;
  }
}

function updateCrossCheckBadge() {
  const badge = document.getElementById("docCrossCheckBadge");
  if (!badge) return;

  const rc = activeDocData.rc_book;
  const pol = activeDocData.insurance_policy;

  if (rc && pol && rc.fields && pol.fields) {
    const rcReg = (rc.fields.registration_number?.value || "").replace(/\s/g, "").toUpperCase();
    const polReg = (pol.fields.vehicle_reg_number?.value || "").replace(/\s/g, "").toUpperCase();

    if (rcReg && polReg) {
      if (rcReg === polReg) {
        badge.textContent = `✅ Reg No Match: ${rc.fields.registration_number.value}`;
        badge.style.background = "rgba(16, 185, 129, 0.2)";
        badge.style.color = "#10B981";
        return;
      } else {
        badge.textContent = `⚠️ Mismatch: RC(${rcReg}) vs Policy(${polReg})`;
        badge.style.background = "rgba(239, 68, 68, 0.2)";
        badge.style.color = "#EF4444";
        return;
      }
    }
  }

  badge.textContent = "Cross-Check: Pending 2+ Docs";
  badge.style.background = "rgba(255, 255, 255, 0.08)";
  badge.style.color = "#94A3B8";
}

function renderClaimDocuments(docs) {
  if (!docs) return;
  activeDocData.rc_book = docs.rc_book || null;
  activeDocData.insurance_policy = docs.insurance_policy || null;
  activeDocData.driving_licence = docs.driving_licence || null;
  renderActiveDocFields();
  updateCrossCheckBadge();
}

/* ==========================================================================
   Phase 5: AI-Assisted Explainable Survey Report Dossier Functions
   ========================================================================== */

function initReportSection() {
  const btnGenerateReport = document.getElementById("btnGenerateReport");
  const btnFinalizeReport = document.getElementById("btnFinalizeReport");
  const btnExportMarkdown = document.getElementById("btnExportMarkdown");
  const tabViewMarkdown = document.getElementById("tabViewMarkdown");
  const tabViewJson = document.getElementById("tabViewJson");

  if (btnGenerateReport) {
    btnGenerateReport.addEventListener("click", generateSurveyReportAction);
  }
  if (btnFinalizeReport) {
    btnFinalizeReport.addEventListener("click", finalizeSurveyReportAction);
  }
  if (btnExportMarkdown) {
    btnExportMarkdown.addEventListener("click", exportMarkdownReportAction);
  }

  if (tabViewMarkdown && tabViewJson) {
    tabViewMarkdown.addEventListener("click", () => {
      tabViewMarkdown.classList.add("active");
      tabViewJson.classList.remove("active");
      document.getElementById("reportMarkdownBody").style.display = "block";
      document.getElementById("reportJsonBody").style.display = "none";
    });

    tabViewJson.addEventListener("click", () => {
      tabViewJson.classList.add("active");
      tabViewMarkdown.classList.remove("active");
      document.getElementById("reportMarkdownBody").style.display = "none";
      document.getElementById("reportJsonBody").style.display = "block";
    });
  }
}

async function generateSurveyReportAction() {
  const claimId = claimIdInput.value.trim();
  if (!claimId) {
    alert("Please enter or load a valid Claim ID first.");
    return;
  }

  const spinner = document.getElementById("reportSpinnerBox");
  const emptyPlaceholder = document.getElementById("reportEmptyPlaceholder");
  const activeView = document.getElementById("reportActiveView");

  if (spinner) spinner.style.display = "flex";
  if (emptyPlaceholder) emptyPlaceholder.style.display = "none";

  try {
    const res = await fetch(`${API_BASE}/api/claims/${encodeURIComponent(claimId)}/generate-report`, {
      method: "POST",
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: "Failed to generate report" }));
      throw new Error(err.detail || "Report synthesis failed");
    }

    const report = await res.json();
    activeSurveyReport = report;
    renderSurveyReport(report);
  } catch (err) {
    alert(`Report Generation Error: ${err.message}`);
    if (emptyPlaceholder && !activeSurveyReport) emptyPlaceholder.style.display = "block";
  } finally {
    if (spinner) spinner.style.display = "none";
  }
}

function renderSurveyReport(report) {
  if (!report) return;
  activeSurveyReport = report;

  const emptyPlaceholder = document.getElementById("reportEmptyPlaceholder");
  const activeView = document.getElementById("reportActiveView");
  const statusBadge = document.getElementById("reportStatusBadge");
  const dispositionEl = document.getElementById("reportDisposition");
  const costRangeEl = document.getElementById("reportCostRange");
  const modelUsedEl = document.getElementById("reportModelUsed");
  const latencyEl = document.getElementById("reportLatency");
  const mdBody = document.getElementById("reportMarkdownBody");
  const jsonCode = document.getElementById("reportJsonCode");
  const signoffBadge = document.getElementById("signoffStatusBadge");
  const signoffNotesInput = document.getElementById("surveyorSignoffNotes");

  if (emptyPlaceholder) emptyPlaceholder.style.display = "none";
  if (activeView) activeView.style.display = "block";

  const isFinalized = (report.status === "FINALIZED");

  if (statusBadge) {
    statusBadge.textContent = isFinalized ? "Finalized Assessment" : "AI-Assisted Draft Dossier";
    statusBadge.className = isFinalized ? "badge-finalized-status" : "badge-draft-status";
  }

  if (signoffBadge) {
    signoffBadge.textContent = isFinalized ? `Signed Off (${new Date(report.signed_off_at || Date.now()).toLocaleDateString()})` : "Pending Human Signature";
    signoffBadge.style.background = isFinalized ? "rgba(16, 185, 129, 0.2)" : "rgba(245, 158, 11, 0.2)";
    signoffBadge.style.color = isFinalized ? "#10B981" : "#F59E0B";
  }

  if (dispositionEl) dispositionEl.textContent = report.claim_disposition || "Conditional Approval";
  if (costRangeEl) {
    costRangeEl.textContent = `₹${(report.estimated_repair_cost_min || 0).toLocaleString()} - ₹${(report.estimated_repair_cost_max || 0).toLocaleString()}`;
  }
  if (modelUsedEl) modelUsedEl.textContent = report.model_used || "Groq LLM";
  if (latencyEl) latencyEl.textContent = report.latency_ms ? `${report.latency_ms.toFixed(0)}ms` : "Fast";

  if (signoffNotesInput && report.surveyor_signoff_notes) {
    signoffNotesInput.value = report.surveyor_signoff_notes;
  }

  // Render Markdown
  if (mdBody) {
    mdBody.innerHTML = renderSimpleMarkdown(report.markdown_dossier || "");
  }

  // Render JSON
  if (jsonCode) {
    jsonCode.textContent = JSON.stringify(report, null, 2);
  }
}

async function finalizeSurveyReportAction() {
  const claimId = claimIdInput.value.trim();
  if (!claimId) return;

  const signoffNotesInput = document.getElementById("surveyorSignoffNotes");
  const notes = signoffNotesInput ? signoffNotesInput.value.trim() : "";

  try {
    const res = await fetch(`${API_BASE}/api/claims/${encodeURIComponent(claimId)}/finalize-report`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ signoff_notes: notes }),
    });

    if (!res.ok) throw new Error("Failed to finalize report");
    const finalizedReport = await res.json();
    renderSurveyReport(finalizedReport);
    alert("✅ Official Claim Survey Dossier has been signed off and finalized!");
  } catch (err) {
    alert(`Could not finalize report: ${err.message}`);
  }
}

function exportMarkdownReportAction() {
  if (!activeSurveyReport || !activeSurveyReport.markdown_dossier) {
    alert("No generated report to export.");
    return;
  }

  const claimId = activeSurveyReport.claim_id || "claim";
  const blob = new Blob([activeSurveyReport.markdown_dossier], { type: "text/markdown;charset=utf-8" });
  const downloadUrl = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = downloadUrl;
  a.download = `${claimId}_explainable_survey_dossier.md`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(downloadUrl);
}

/**
 * Lightweight, robust Markdown to HTML Converter for Dossier Rendering.
 */
function renderSimpleMarkdown(mdText) {
  if (!mdText) return "<p>No report text available.</p>";

  const lines = mdText.split("\n");
  const output = [];
  let inTable = false;
  let tableRows = [];

  function flushTable() {
    if (!inTable || tableRows.length === 0) return;
    let html = "<table>";
    tableRows.forEach((row, rIdx) => {
      // Skip separator rows like |---|---|
      if (row.every((cell) => cell.replace(/-/g, "").trim() === "")) return;
      const tag = (rIdx === 0) ? "th" : "td";
      html += "<tr>";
      row.forEach((cell) => {
        html += `<${tag}>${formatInline(cell)}</${tag}>`;
      });
      html += "</tr>";
    });
    html += "</table>";
    output.push(html);
    tableRows = [];
    inTable = false;
  }

  function formatInline(str) {
    return str
      .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
      .replace(/\*(.*?)\*/g, "<em>$1</em>")
      .replace(/`([^`]+)`/g, "<code>$1</code>");
  }

  for (let i = 0; i < lines.length; i++) {
    const rawLine = lines[i];
    const line = rawLine.trim();

    // Table detection
    if (line.startsWith("|") && line.endsWith("|")) {
      inTable = true;
      const cells = line
        .slice(1, -1)
        .split("|")
        .map((c) => c.trim());
      tableRows.push(cells);
      continue;
    } else if (inTable) {
      flushTable();
    }

    // Headings
    if (line.startsWith("### ")) {
      output.push(`<h3>${formatInline(line.slice(4))}</h3>`);
    } else if (line.startsWith("## ")) {
      output.push(`<h2>${formatInline(line.slice(3))}</h2>`);
    } else if (line.startsWith("# ")) {
      output.push(`<h1>${formatInline(line.slice(2))}</h1>`);
    } else if (line.startsWith("---")) {
      output.push("<hr>");
    } else if (line.startsWith("* ") || line.startsWith("- ")) {
      output.push(`<ul><li>${formatInline(line.slice(2))}</li></ul>`);
    } else if (line.length > 0) {
      output.push(`<p>${formatInline(line)}</p>`);
    }
  }

  flushTable();
  return output.join("\n");
}

