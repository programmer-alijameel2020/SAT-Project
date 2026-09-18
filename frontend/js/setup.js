let selectedDatasetSource = "demo";
let selectedMode = "homography";
let selectedDetectorMode = "placeholder";
let motionInjectionOn = true;
let foundSequences = [];
let browseCurrentPath = "";
let browseCurrentParent = null;

const LAST_PATH_KEY = (src) => `satmot_last_path_${src}`;

function loadRememberedPath(src) {
  try {
    return localStorage.getItem(LAST_PATH_KEY(src)) || "";
  } catch (e) {
    return "";
  }
}

function rememberPath(src, path) {
  try {
    localStorage.setItem(LAST_PATH_KEY(src), path);
  } catch (e) { /* ignore (private browsing, etc.) */ }
}

// ---------------------------------------------------------------------- //
// Generic field persistence -- every parameter field remembers its last
// value across sessions. This exists because a silently-reset parameter
// (e.g. det_diff_thresh defaulting back to 18 on a fresh page load) can
// invalidate an ablation run without any error being raised -- the run
// completes "successfully," just against the wrong settings. Restoring
// on load and saving on every change closes that gap for good rather
// than relying on remembering to retype values each time.
// ---------------------------------------------------------------------- //
const PERSISTED_FIELD_IDS = [
  "predictor_model", "window_length", "kmax_occlusion", "min_hits_to_confirm",
  "max_frames", "min_inliers", "min_inlier_ratio", "max_reproj_error",
  "det_min_area", "det_max_area", "det_diff_thresh", "det_max_detections",
  "save_frame_every", "step_delay_ms", "synth_motion_px_per_frame", "synth_motion_seed",
];
const FIELD_KEY = (id) => `satmot_field_${id}`;
const MODE_KEY = "satmot_trajectory_mode";
const SOURCE_KEY = "satmot_dataset_source";

function restorePersistedFields() {
  PERSISTED_FIELD_IDS.forEach((id) => {
    const saved = localStorage.getItem(FIELD_KEY(id));
    if (saved !== null) {
      const el = document.getElementById(id);
      if (el) el.value = saved;
    }
  });
}

function wirePersistence() {
  PERSISTED_FIELD_IDS.forEach((id) => {
    const el = document.getElementById(id);
    if (!el) return;
    el.addEventListener("change", () => {
      try { localStorage.setItem(FIELD_KEY(id), el.value); } catch (e) { /* ignore */ }
    });
  });
}

function initChipGroup(groupId, onSelect, valueAttr = "data-value") {
  const group = document.getElementById(groupId);
  group.querySelectorAll(".radio-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      group.querySelectorAll(".radio-chip").forEach((c) => c.classList.remove("selected"));
      chip.classList.add("selected");
      onSelect(chip.getAttribute(valueAttr));
    });
  });
}

initChipGroup("dataset-source-group", (val) => {
  selectedDatasetSource = val;
  document.getElementById("dataset-root-field").style.display = val === "demo" ? "none" : "block";
  if (val !== "demo") {
    const remembered = loadRememberedPath(val);
    if (remembered) document.getElementById("dataset_root").value = remembered;
  }
  try { localStorage.setItem(SOURCE_KEY, val); } catch (e) { /* ignore */ }
});

// Restore last-used dataset source chip (defaults to "demo" if none saved)
const savedSource = (() => { try { return localStorage.getItem(SOURCE_KEY); } catch (e) { return null; } })();
const initialSourceValue = savedSource || "demo";
document.querySelector(`#dataset-source-group [data-value="${initialSourceValue}"]`)?.click();
if (!document.querySelector(`#dataset-source-group [data-value="${initialSourceValue}"].selected`)) {
  // fallback if the saved value somehow doesn't match a chip
  document.querySelector('#dataset-source-group [data-value="demo"]').classList.add("selected");
  document.getElementById("dataset-root-field").style.display = "none";
}

initChipGroup("mode-group", (val) => {
  selectedMode = val;
  try { localStorage.setItem(MODE_KEY, val); } catch (e) { /* ignore */ }
});

// Restore last-used trajectory mode chip (defaults to "homography")
const savedMode = (() => { try { return localStorage.getItem(MODE_KEY); } catch (e) { return null; } })();
if (savedMode) {
  document.querySelector(`#mode-group [data-value="${savedMode}"]`)?.click();
}

initChipGroup("detector-mode-group", (val) => {
  selectedDetectorMode = val;
});

initChipGroup("motion-injection-group", (val) => {
  motionInjectionOn = (val === "on");
  document.getElementById("motion-injection-fields").style.display = motionInjectionOn ? "grid" : "none";
});

restorePersistedFields();
wirePersistence();

initChipGroup("preset-group", (val) => {
  const datasetGroup = document.getElementById("dataset-source-group");
  if (val === "demo") {
    datasetGroup.querySelector('[data-value="demo"]').click();
  } else if (val === "viso") {
    datasetGroup.querySelector('[data-value="VISO"]').click();
  } else if (val === "sdmcar") {
    datasetGroup.querySelector('[data-value="SDM-Car"]').click();
  }
  // Presets deliberately do NOT pre-fill dataset_root with a placeholder
  // path -- a placeholder that silently gets submitted as if it were
  // real (because the field "looked" non-empty) is worse than an empty
  // field with a visible required-looking prompt. The field's HTML
  // placeholder attribute (greyed-out example text) still shows the
  // expected format without being submittable by accident.
}, "data-preset");

document.getElementById("scan-btn").addEventListener("click", async () => {
  const statusEl = document.getElementById("scan-status");
  const root = document.getElementById("dataset_root").value.trim();
  if (selectedDatasetSource !== "demo" && !root) {
    statusEl.textContent = "Enter the dataset root directory path above first.";
    return;
  }
  statusEl.textContent = "Scanning...";
  try {
    const res = await apiPost("/api/datasets/scan", {
      dataset_source: selectedDatasetSource,
      dataset_root: root,
    });
    foundSequences = res.sequences;
    statusEl.textContent = `Found ${res.n_sequences} sequence(s).`;
    const seqField = document.getElementById("sequence-field");
    const seqSelect = document.getElementById("sequence_name");
    seqSelect.innerHTML = "";
    res.sequences.forEach((s) => {
      const opt = document.createElement("option");
      opt.value = s.name;
      opt.textContent = `${s.name}  (${s.n_frames} frames${s.has_annotations ? ", annotated" : ""})`;
      seqSelect.appendChild(opt);
    });
    seqField.style.display = res.sequences.length ? "block" : "none";
    if (res.n_sequences === 0) {
      statusEl.textContent += " Check the dataset root path.";
    } else if (selectedDatasetSource !== "demo") {
      rememberPath(selectedDatasetSource, root);
    }
  } catch (e) {
    statusEl.textContent = "Scan failed: " + e.message;
  }
});

// ---------------------------------------------------------------------- //
// Folder browser modal
// ---------------------------------------------------------------------- //
const browseModal = document.getElementById("browse-modal");

async function openBrowseModal() {
  const startPath = document.getElementById("dataset_root").value.trim()
    || loadRememberedPath(selectedDatasetSource) || "";
  browseModal.style.display = "flex";
  await browseNavigate(startPath);
}

async function browseNavigate(path) {
  const listEl = document.getElementById("browse-list");
  listEl.innerHTML = `<div class="field-help" style="padding:8px;">Loading&hellip;</div>`;
  try {
    const res = await apiGet(`/api/fs/list?path=${encodeURIComponent(path)}`);
    browseCurrentPath = res.path;
    browseCurrentParent = res.parent;
    document.getElementById("browse-current-path").textContent = res.path || "(drives)";
    listEl.innerHTML = "";

    if (res.parent !== null && res.parent !== undefined) {
      const up = document.createElement("div");
      up.className = "file-row";
      up.innerHTML = `<span class="name">&uarr; ..</span>`;
      up.addEventListener("click", () => browseNavigate(res.parent));
      listEl.appendChild(up);
    }

    if (res.entries.length === 0) {
      listEl.innerHTML += `<div class="field-help" style="padding:8px;">No subfolders here.</div>`;
    }
    res.entries.forEach((entry) => {
      const row = document.createElement("div");
      row.className = "file-row";
      const hint = entry.hint ? ` <span class="mono" style="color:var(--accent);">[${entry.hint}]</span>` : "";
      row.innerHTML = `<span class="name">${entry.type === "drive" ? "\u{1F4BE}" : "\u{1F4C1}"} ${entry.name}${hint}</span>`;
      row.addEventListener("dblclick", () => browseNavigate(entry.path));
      row.addEventListener("click", () => browseNavigate(entry.path));
      listEl.appendChild(row);
    });
  } catch (e) {
    listEl.innerHTML = `<div class="field-help" style="padding:8px; color: var(--danger);">Could not read this folder: ${e.message}</div>`;
  }
}

document.getElementById("browse-btn").addEventListener("click", openBrowseModal);
document.getElementById("browse-close-btn").addEventListener("click", () => browseModal.style.display = "none");
document.getElementById("browse-cancel-btn").addEventListener("click", () => browseModal.style.display = "none");
document.getElementById("browse-select-btn").addEventListener("click", () => {
  document.getElementById("dataset_root").value = browseCurrentPath;
  if (selectedDatasetSource !== "demo") rememberPath(selectedDatasetSource, browseCurrentPath);
  browseModal.style.display = "none";
});

document.getElementById("reset-defaults-btn").addEventListener("click", () => {
  if (!confirm("Reset all detector/tracker parameters to their defaults? This clears remembered values (dataset path is kept).")) return;
  PERSISTED_FIELD_IDS.forEach((id) => localStorage.removeItem(FIELD_KEY(id)));
  localStorage.removeItem(MODE_KEY);
  window.location.reload();
});

document.getElementById("experiment-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const submitBtn = document.getElementById("submit-btn");
  const statusEl = document.getElementById("submit-status");

  const datasetRoot = document.getElementById("dataset_root").value.trim();
  if (selectedDatasetSource !== "demo" && !datasetRoot) {
    statusEl.textContent = "Dataset root directory is required for VISO/SDM-Car -- enter it above.";
    return;
  }

  submitBtn.disabled = true;
  statusEl.textContent = "Creating experiment...";

  const seqSelect = document.getElementById("sequence_name");
  const sequenceName = seqSelect && seqSelect.value ? seqSelect.value : null;

  const params = {
    name: document.getElementById("name").value || "unnamed_experiment",
    dataset_source: selectedDatasetSource,
    dataset_root: document.getElementById("dataset_root").value,
    sequence_name: sequenceName,
    trajectory_mode: selectedMode,
    detector_mode: selectedDetectorMode,
    inject_synthetic_motion: motionInjectionOn,
    synth_motion_px_per_frame: parseFloat(document.getElementById("synth_motion_px_per_frame").value),
    synth_motion_seed: parseInt(document.getElementById("synth_motion_seed").value),
    predictor_model: document.getElementById("predictor_model").value,
    window_length: parseInt(document.getElementById("window_length").value),
    kmax_occlusion: parseInt(document.getElementById("kmax_occlusion").value),
    min_hits_to_confirm: parseInt(document.getElementById("min_hits_to_confirm").value),
    max_frames: parseInt(document.getElementById("max_frames").value),
    min_inliers: parseInt(document.getElementById("min_inliers").value),
    min_inlier_ratio: parseFloat(document.getElementById("min_inlier_ratio").value),
    max_reproj_error: parseFloat(document.getElementById("max_reproj_error").value),
    det_min_area: parseInt(document.getElementById("det_min_area").value),
    det_max_area: parseInt(document.getElementById("det_max_area").value),
    det_diff_thresh: parseInt(document.getElementById("det_diff_thresh").value),
    det_max_detections: parseInt(document.getElementById("det_max_detections").value),
    save_frame_every: parseInt(document.getElementById("save_frame_every").value),
    step_delay_ms: parseInt(document.getElementById("step_delay_ms").value),
  };

  try {
    const created = await apiPost("/api/experiments", params);
    if (selectedDatasetSource !== "demo") rememberPath(selectedDatasetSource, datasetRoot);
    statusEl.textContent = "Launching...";
    await apiPost(`/api/experiments/${created.experiment_uid}/start`);
    window.location.href = `dashboard.html?uid=${created.experiment_uid}`;
  } catch (e) {
    statusEl.textContent = "Failed: " + e.message;
    submitBtn.disabled = false;
  }
});
