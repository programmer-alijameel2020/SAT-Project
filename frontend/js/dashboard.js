const SETUP_STAGES = [
  { key: "initializing", label: "Initialize experiment" },
  { key: "dataset_scanning", label: "Scan dataset" },
  { key: "sequence_selected", label: "Select sequence" },
  { key: "video_to_frames", label: "Decode video \u2192 frame array" },
];
const CYCLE_STAGES = [
  { key: "frame_loaded", label: "Load frame" },
  { key: "detection", label: "Detect candidate targets" },
  { key: "homography_estimation", label: "Estimate platform motion" },
  { key: "motion_decomposition", label: "Compensate trajectories" },
  { key: "residual_prediction", label: "Residual + Kalman forecast" },
  { key: "association", label: "Associate & reactivate" },
];
const FINAL_STAGES = [
  { key: "computing_metrics", label: "Compute metrics" },
  { key: "saving_results", label: "Save results" },
  { key: "completed", label: "Completed" },
];

let uid = null;
let expConfig = null;
let maxFrames = 0;
let lastFrameShape = [320, 480, 3];

function qs(name) {
  return new URLSearchParams(window.location.search).get(name);
}

function renderRail(containerId, stages) {
  const el = document.getElementById(containerId);
  el.innerHTML = "";
  stages.forEach((s, i) => {
    const div = document.createElement("div");
    div.className = "stage-item";
    div.id = `stage-${s.key}`;
    div.innerHTML = `
      <div class="stage-dot">${i + 1}</div>
      <div>
        <div class="stage-label">${s.label}</div>
        <div class="stage-detail" id="detail-${s.key}"></div>
      </div>`;
    el.appendChild(div);
  });
}

function setStageState(key, state, detail) {
  const item = document.getElementById(`stage-${key}`);
  if (!item) return;
  item.classList.remove("done", "active");
  if (state) item.classList.add(state);
  if (detail !== undefined) {
    const d = document.getElementById(`detail-${key}`);
    if (d) d.textContent = detail;
  }
}

function markUpTo(stages, key, currentState) {
  const idx = stages.findIndex((s) => s.key === key);
  stages.forEach((s, i) => {
    if (i < idx) setStageState(s.key, "done");
    else if (i === idx) setStageState(s.key, currentState);
    else setStageState(s.key, null);
  });
  return idx;
}

function logLine(text, cls = "") {
  const stream = document.getElementById("log-stream");
  const line = document.createElement("div");
  line.className = "log-line " + cls;
  const t = new Date().toLocaleTimeString();
  line.innerHTML = `<span class="tag">${t}</span>${text}`;
  stream.appendChild(line);
  stream.scrollTop = stream.scrollHeight;
}

function ensureCanvas() {
  let canvas = document.getElementById("track-canvas");
  const viewer = document.getElementById("frame-viewer");
  if (!canvas) {
    viewer.innerHTML = "";
    canvas = document.createElement("canvas");
    canvas.id = "track-canvas";
    canvas.width = 640;
    canvas.height = 480;
    canvas.style.width = "100%";
    canvas.style.height = "100%";
    viewer.appendChild(canvas);
  }
  return canvas;
}

function drawTracks(tracks, frameShape) {
  const canvas = ensureCanvas();
  const ctx = canvas.getContext("2d");
  const [h, w] = frameShape;
  canvas.width = 640;
  canvas.height = Math.round(640 * (h / w));
  const scaleX = canvas.width / w;
  const scaleY = canvas.height / h;

  ctx.fillStyle = "#05070c";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // faint grid to suggest the compensated coordinate frame
  ctx.strokeStyle = "rgba(94,234,212,0.06)";
  ctx.lineWidth = 1;
  for (let x = 0; x < canvas.width; x += 32) {
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height); ctx.stroke();
  }
  for (let y = 0; y < canvas.height; y += 32) {
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvas.width, y); ctx.stroke();
  }

  (tracks || []).forEach((tr) => {
    if (!tr.box) return;
    const [x, y, bw, bh] = tr.box;
    const rx = x * scaleX, ry = y * scaleY, rw = Math.max(bw * scaleX, 4), rh = Math.max(bh * scaleY, 4);
    const color = tr.visible ? "#5eead4" : "#f2a65a";
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.strokeRect(rx - rw / 2, ry - rh / 2, rw, rh);
    ctx.fillStyle = color;
    ctx.font = "10px JetBrains Mono, monospace";
    const label = tr.visible ? `#${tr.id}` : `#${tr.id} occ:${tr.occlusion_age}`;
    ctx.fillText(label, rx - rw / 2, ry - rh / 2 - 3);
  });
}

async function loadOverlayImageIfSaved(frameIdx) {
  const saveEvery = (expConfig && expConfig.params_json && expConfig.params_json.save_frame_every) || 5;
  if (frameIdx % saveEvery !== 0) return;
  const padded = String(frameIdx).padStart(5, "0");
  const path = `frames/frame_${padded}.jpg`;
  const url = `/api/experiments/${uid}/file?path=${encodeURIComponent(path)}&t=${Date.now()}`;
  const img = new Image();
  img.onload = () => {
    const viewer = document.getElementById("frame-viewer");
    viewer.innerHTML = "";
    const el = document.createElement("img");
    el.src = url;
    el.style.width = "100%";
    viewer.appendChild(el);
  };
  img.src = url;
}

function updateTelemetry(payload) {
  if (payload.n_tracks_active !== undefined) document.getElementById("stat-tracks").textContent = payload.n_tracks_active;
  if (payload.n_tracks_tentative !== undefined) document.getElementById("stat-tentative").textContent = payload.n_tracks_tentative;
  if (payload.n_occluded !== undefined) document.getElementById("stat-occluded").textContent = payload.n_occluded;
  if (payload.n_matched !== undefined) document.getElementById("stat-matched").textContent = payload.n_matched;
  if (payload.n_new !== undefined) document.getElementById("stat-new").textContent = payload.n_new;
  if (payload.homography_mode !== undefined) document.getElementById("stat-hmode").textContent = payload.homography_mode;
  if (payload.homography_quality !== undefined) document.getElementById("stat-hquality").textContent = payload.homography_quality;
}

function handleEvent(ev) {
  // any live event arriving means the run is active -- keep the cancel
  // button visible regardless of what status the page loaded with
  if (!["completed", "cancelled", "failed"].includes(ev.stage)) {
    document.getElementById("status-badge").textContent = "running";
    document.getElementById("status-badge").className = "badge running";
    document.getElementById("cancel-btn").style.display = "inline-flex";
    document.getElementById("cancel-btn").disabled = false;
    document.getElementById("cancel-btn").textContent = "Cancel run";
  }

  const detail = ev.message || (ev.frame !== undefined ? `frame ${ev.frame}` : "");

  if (SETUP_STAGES.some((s) => s.key === ev.stage)) {
    markUpTo(SETUP_STAGES, ev.stage, "active");
    setStageState(ev.stage, "active", detail);
    logLine(`<b>${ev.stage}</b> ${detail || ""}`);
    if (ev.device) {
      const d = ev.device;
      document.getElementById("stat-device").textContent = d.gpu_available
        ? `${d.device}${d.gpu_name ? " (" + d.gpu_name + ")" : ""}`
        : "cpu";
      logLine(`<span class="mono">${d.note}</span>`);
    }
  }

  if (ev.stage === "sequence_selected") {
    maxFrames = ev.n_frames || 0;
    document.getElementById("frame-progress").textContent = `0 / ${maxFrames}`;
    document.getElementById("exp-meta").textContent =
      `${expConfig ? expConfig.dataset_source : ""} \u00b7 ${ev.sequence_name} \u00b7 mode=${expConfig ? expConfig.trajectory_mode : ""} \u00b7 detector=${expConfig && expConfig.params_json ? expConfig.params_json.detector_mode : "placeholder"}`;
    if (ev.fallback_used) {
      logLine(`<b>WARNING</b> ${ev.message}`, "warn");
    } else {
      logLine(`sequence <b>${ev.sequence_name}</b>: ${ev.n_frames} frames, annotations=${ev.has_annotations}`);
    }
  }

  if (CYCLE_STAGES.some((s) => s.key === ev.stage)) {
    SETUP_STAGES.forEach((s) => setStageState(s.key, "done"));
    markUpTo(CYCLE_STAGES, ev.stage, "active");
    const f = ev.frame !== undefined ? ev.frame : "?";
    setStageState(ev.stage, "active", `frame ${f}`);

    if (ev.stage === "frame_loaded") {
      lastFrameShape = ev.shape || lastFrameShape;
      const pct = maxFrames ? Math.min(100, (ev.frame / maxFrames) * 100) : 0;
      document.getElementById("sweep-fill").style.width = pct + "%";
      document.getElementById("frame-progress").textContent = `${ev.frame} / ${maxFrames}`;
      document.getElementById("frame-caption").textContent = `Frame ${ev.frame} decoded from source video/image-sequence`;
    }
    if (ev.stage === "detection") {
      document.getElementById("frame-caption").textContent = `${ev.n_detections} candidate detection(s) found`;
    }
    if (ev.stage === "homography_estimation") {
      updateTelemetry({ homography_mode: ev.mode, homography_quality: ev.quality });
    }
    if (ev.stage === "association") {
      updateTelemetry(ev);
      drawTracks(ev.tracks, lastFrameShape);
      loadOverlayImageIfSaved(ev.frame);
    }
  }

  if (ev.stage === "frame_complete") {
    updateTelemetry(ev);
    logLine(`frame ${ev.frame}: matched=${ev.n_matched} new=${ev.n_new} occluded=${ev.n_occluded} dead=${ev.n_dead}`);
  }

  if (FINAL_STAGES.some((s) => s.key === ev.stage)) {
    CYCLE_STAGES.forEach((s) => setStageState(s.key, "done"));
    markUpTo(FINAL_STAGES, ev.stage, ev.stage === "completed" ? "done" : "active");
    logLine(`<b>${ev.stage}</b> ${detail || ""}`);
  }

  if (ev.stage === "completed") {
    document.getElementById("status-badge").textContent = "completed";
    document.getElementById("status-badge").className = "badge completed";
    document.getElementById("sweep-fill").style.width = "100%";
    document.getElementById("cancel-btn").style.display = "none";
    showResults(ev.summary, ev.output_dir);
    logLine(`<b>run completed</b> \u2192 ${ev.output_dir}`);
  }

  if (ev.stage === "cancelled") {
    document.getElementById("status-badge").textContent = "cancelled";
    document.getElementById("status-badge").className = "badge cancelled";
    document.getElementById("cancel-btn").style.display = "none";
    showResults(ev.summary, ev.output_dir);
    logLine(`<b>run cancelled</b> \u2014 partial results saved to ${ev.output_dir}`, "warn");
  }

  if (ev.stage === "failed") {
    document.getElementById("status-badge").textContent = "failed";
    document.getElementById("status-badge").className = "badge failed";
    document.getElementById("cancel-btn").style.display = "none";
    logLine(`<b>ERROR</b> ${ev.error}`, "err");
  }
}

function showResults(summary, outputDir) {
  const panel = document.getElementById("results-panel");
  panel.style.display = "block";
  document.getElementById("results-body").textContent = JSON.stringify(summary, null, 2);
  document.getElementById("open-explorer-link").href = `explorer.html?uid=${uid}`;
  document.getElementById("stat-recovery").textContent = summary.n_occlusion_events ?? 0;
}

async function init() {
  uid = qs("uid");
  if (!uid) {
    document.getElementById("no-run").style.display = "block";
    document.getElementById("run-content").style.display = "none";
    return;
  }

  renderRail("setup-rail", SETUP_STAGES);
  renderRail("cycle-rail", CYCLE_STAGES);
  renderRail("final-rail", FINAL_STAGES);

  document.getElementById("cancel-btn").addEventListener("click", async () => {
    const btn = document.getElementById("cancel-btn");
    btn.disabled = true;
    btn.textContent = "Cancelling...";
    try {
      const res = await apiPost(`/api/experiments/${uid}/cancel`);
      logLine(res.message);
    } catch (e) {
      logLine("Cancel request failed: " + e.message, "err");
      btn.disabled = false;
      btn.textContent = "Cancel run";
    }
  });

  try {
    expConfig = await apiGet(`/api/experiments/${uid}`);
    document.getElementById("exp-name").textContent = expConfig.name;
    document.getElementById("exp-meta").textContent =
      `${expConfig.dataset_source} \u00b7 ${expConfig.sequence_name || "(auto)"} \u00b7 mode=${expConfig.trajectory_mode} \u00b7 detector=${expConfig.params_json ? expConfig.params_json.detector_mode : "placeholder"} \u00b7 created ${fmtTime(expConfig.created_at)}`;
    document.getElementById("status-badge").textContent = expConfig.status;
    document.getElementById("status-badge").className = "badge " + expConfig.status;
    if (expConfig.status === "running" || expConfig.status === "pending") {
      document.getElementById("cancel-btn").style.display = "inline-flex";
    }
    if (expConfig.status === "completed" && expConfig.results_json) {
      showResults(expConfig.results_json, expConfig.output_dir);
      SETUP_STAGES.forEach((s) => setStageState(s.key, "done"));
      CYCLE_STAGES.forEach((s) => setStageState(s.key, "done"));
      FINAL_STAGES.forEach((s) => setStageState(s.key, "done"));
      document.getElementById("sweep-fill").style.width = "100%";
    }
  } catch (e) {
    logLine("Failed to load experiment config: " + e.message, "err");
  }

  const ws = new WebSocket(wsURL(`/api/experiments/ws/${uid}`));
  ws.onopen = () => logLine("WebSocket connected \u2014 streaming live pipeline events");
  ws.onclose = () => logLine("WebSocket closed", "warn");
  ws.onerror = () => logLine("WebSocket error", "err");
  ws.onmessage = (msg) => {
    try {
      const ev = JSON.parse(msg.data);
      handleEvent(ev);
    } catch (e) {
      console.error("bad event", e, msg.data);
    }
  };
}

init();
